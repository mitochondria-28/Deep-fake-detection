# webapp/detector/views.py
"""
Upload and result views with:
  - Full file validation (format + size) before saving to disk
  - PredictionRecord saved to database after every inference
  - Result page served from DB record (no session dependency)
  - Recent predictions history page
"""

import os
import uuid
import time
import logging
from pathlib import Path
from typing import Optional

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.views.decorators.http import require_http_methods
from django.utils import timezone

from detector.inference import run_inference
from detector.models import PredictionRecord

logger = logging.getLogger(__name__)


# ── File validation ───────────────────────────────────────────────────────────

def _validate_upload(uploaded_file) -> Optional[str]:
    """
    Validate file BEFORE writing to disk.
    Returns an error string if invalid, None if valid.

    Checks:
      1. File was actually submitted
      2. File size within limit
      3. File extension is allowed
    """
    if not uploaded_file:
        return "No file selected. Please choose a video file."

    # ── Size check ────────────────────────────────────────────────────────────
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if uploaded_file.size > max_bytes:
        mb = uploaded_file.size / (1024 * 1024)
        return (
            f"File too large: {mb:.1f} MB. "
            f"Maximum allowed size is {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    if uploaded_file.size == 0:
        return "Uploaded file is empty. Please select a valid video file."

    # ── Extension check ───────────────────────────────────────────────────────
    name = uploaded_file.name.lower()
    ext  = Path(name).suffix
    if ext not in settings.ALLOWED_VIDEO_EXTS:
        allowed = ", ".join(sorted(settings.ALLOWED_VIDEO_EXTS))
        return (
            f"Unsupported file type '{ext}'. "
            f"Allowed formats: {allowed}"
        )

    return None   # valid


# ── Upload view ───────────────────────────────────────────────────────────────

@require_http_methods(["GET", "POST"])
def upload_view(request):
    """
    GET  — render upload form with optional recent history
    POST — validate, save temp file, run inference, store in DB, redirect
    """
    if request.method == "GET":
        recent = PredictionRecord.objects.all()[:5]
        return render(request, "detector/upload.html", {"recent": recent})

    # ── POST ──────────────────────────────────────────────────────────────────
    uploaded = request.FILES.get("video")

    # Validate before touching disk
    error = _validate_upload(uploaded)
    if error:
        recent = PredictionRecord.objects.all()[:5]
        return render(request, "detector/upload.html", {
            "error":  error,
            "recent": recent,
        })

    # ── Save to temp file ─────────────────────────────────────────────────────
    upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext       = Path(uploaded.name).suffix.lower()
    temp_name = f"upload_{uuid.uuid4().hex}{ext}"
    temp_path = upload_dir / temp_name

    try:
        with open(temp_path, "wb") as f:
            for chunk in uploaded.chunks(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Saved upload: {temp_path} ({uploaded.size:,} bytes)")
    except OSError as e:
        logger.error(f"File save failed: {e}")
        recent = PredictionRecord.objects.all()[:5]
        return render(request, "detector/upload.html", {
            "error":  "File upload failed — disk may be full. Please try again.",
            "recent": recent,
        })

    # ── Run inference (temp file deleted inside run_inference) ────────────────
    t0     = time.time()
    result = run_inference(
        video_path=str(temp_path),
        media_root=str(settings.MEDIA_ROOT),
    )
    elapsed = round(time.time() - t0, 2)

    # ── Save prediction to database ───────────────────────────────────────────
    record = PredictionRecord.objects.create(
        filename         = uploaded.name,
        label            = result.get("label",           "UNCERTAIN"),
        confidence       = result.get("confidence",      0.0),
        real_probability = result.get("real_prob",       0.0),
        fake_probability = result.get("fake_prob",       0.0),
        frames_analyzed  = result.get("frames_analyzed", 0),
        face_grid_path   = result.get("face_grid_path",  None),
        processing_time  = elapsed,
        file_size_bytes  = uploaded.size,
        error_message    = result.get("error",           None),
        created_at       = timezone.now(),
    )
    logger.info(
        f"Prediction saved: id={record.pk} | {record.label} | "
        f"{record.confidence:.1f}% | {elapsed}s"
    )

    return redirect("detector:result", pk=record.pk)


# ── Result view ───────────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def result_view(request, pk: int):
    """
    Serve prediction result from database record.
    Each result has a permanent URL: /result/<pk>/
    """
    record = get_object_or_404(PredictionRecord, pk=pk)

    if record.error_message:
        return render(request, "detector/upload.html", {
            "error":  record.error_message,
            "recent": PredictionRecord.objects.all()[:5],
        })

    context = {
        "record":          record,
        "label":           record.label,
        "confidence":      record.confidence,
        "real_prob":       record.real_probability,
        "fake_prob":       record.fake_probability,
        "frames_analyzed": record.frames_analyzed,
        "face_grid_url":   record.face_grid_url,
        "filename":        record.filename,
        "processing_time": record.processing_time,
        "created_at":      record.created_at,
        "record_id":       record.pk,
    }
    return render(request, "detector/result.html", context)


# ── History view ──────────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def history_view(request):
    """Show all past predictions from the database."""
    records = PredictionRecord.objects.all()
    counts  = {
        "total":     records.count(),
        "real":      records.filter(label="REAL").count(),
        "fake":      records.filter(label="FAKE").count(),
        "uncertain": records.filter(label="UNCERTAIN").count(),
    }
    return render(request, "detector/history.html", {
        "records": records[:50],
        "counts":  counts,
    })
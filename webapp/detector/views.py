# webapp/detector/views.py
"""
Upload and result views with:
  - Full file validation (format + size) before saving to disk
  - PredictionRecord saved to database after every inference
  - Result page served from DB record (no session dependency)
  - Recent predictions history page
  - Separate routes for video vs single-image detection
  - PDF report download for any saved result
"""

import os
import uuid
import time
import logging
from pathlib import Path
from typing import Optional

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.utils import timezone

from detector.inference import run_inference, run_image_inference
from detector.models import PredictionRecord
from detector.report import render_report_pdf

logger = logging.getLogger(__name__)


# ── File validation ───────────────────────────────────────────────────────────

def _validate_video_upload(uploaded_file) -> Optional[str]:
    """
    Validate a video file BEFORE writing to disk.
    Returns an error string if invalid, None if valid.
    """
    if not uploaded_file:
        return "No file selected. Please choose a video file."

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if uploaded_file.size > max_bytes:
        mb = uploaded_file.size / (1024 * 1024)
        return (
            f"File too large: {mb:.1f} MB. "
            f"Maximum allowed size is {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    if uploaded_file.size == 0:
        return "Uploaded file is empty. Please select a valid video file."

    name = uploaded_file.name.lower()
    ext  = Path(name).suffix
    if ext not in settings.ALLOWED_VIDEO_EXTS:
        allowed = ", ".join(sorted(settings.ALLOWED_VIDEO_EXTS))
        return (
            f"Unsupported file type '{ext}'. "
            f"Allowed formats: {allowed}"
        )

    return None   # valid


def _validate_image_upload(uploaded_file) -> Optional[str]:
    """
    Validate an image file BEFORE writing to disk.
    Returns an error string if invalid, None if valid.
    """
    if not uploaded_file:
        return "No file selected. Please choose an image file."

    max_image_mb = getattr(settings, "MAX_IMAGE_UPLOAD_SIZE_MB", 10)
    max_bytes = max_image_mb * 1024 * 1024
    if uploaded_file.size > max_bytes:
        mb = uploaded_file.size / (1024 * 1024)
        return (
            f"File too large: {mb:.1f} MB. "
            f"Maximum allowed size is {max_image_mb} MB."
        )

    if uploaded_file.size == 0:
        return "Uploaded file is empty. Please select a valid image file."

    allowed_image_exts = {".jpg", ".jpeg", ".png"}
    name = uploaded_file.name.lower()
    ext  = Path(name).suffix
    if ext not in allowed_image_exts:
        allowed = ", ".join(sorted(allowed_image_exts))
        return (
            f"Unsupported file type '{ext}'. "
            f"Allowed formats: {allowed}"
        )

    return None   # valid


# ── Video upload view ─────────────────────────────────────────────────────────

@require_http_methods(["GET", "POST"])
def upload_view(request):
    """
    GET  — render upload form with optional recent history
    POST — validate, save temp video file, run video inference, store in DB
    """
    if request.method == "GET":
        recent = PredictionRecord.objects.all()[:5]
        return render(request, "detector/upload.html", {"recent": recent})

    # ── POST ──────────────────────────────────────────────────────────────────
    uploaded = request.FILES.get("video")

    error = _validate_video_upload(uploaded)
    if error:
        recent = PredictionRecord.objects.all()[:5]
        return render(request, "detector/upload.html", {
            "error":  error,
            "recent": recent,
        })

    upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext       = Path(uploaded.name).suffix.lower()
    temp_name = f"upload_{uuid.uuid4().hex}{ext}"
    temp_path = upload_dir / temp_name

    try:
        with open(temp_path, "wb") as f:
            for chunk in uploaded.chunks(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Saved video upload: {temp_path} ({uploaded.size:,} bytes)")
    except OSError as e:
        logger.error(f"File save failed: {e}")
        recent = PredictionRecord.objects.all()[:5]
        return render(request, "detector/upload.html", {
            "error":  "File upload failed — disk may be full. Please try again.",
            "recent": recent,
        })

    t0     = time.time()
    result = run_inference(
        video_path=str(temp_path),
        media_root=str(settings.MEDIA_ROOT),
    )
    elapsed = round(time.time() - t0, 2)

    record = PredictionRecord.objects.create(
        filename          = uploaded.name,
        label             = result.get("label",             "UNCERTAIN"),
        confidence        = result.get("confidence",        0.0),
        real_probability  = result.get("real_prob",          0.0),
        fake_probability  = result.get("fake_prob",          0.0),
        frames_analyzed   = result.get("frames_analyzed",    0),
        face_grid_path    = result.get("face_grid_path",     None),
        gradcam_grid_path = result.get("gradcam_grid_path",  None),
        detection_mode    = result.get("detection_mode",     "video"),
        processing_time   = elapsed,
        file_size_bytes   = uploaded.size,
        error_message     = result.get("error",              None),
        created_at        = timezone.now(),
    )
    logger.info(
        f"Video prediction saved: id={record.pk} | {record.label} | "
        f"{record.confidence:.1f}% | {elapsed}s"
    )

    return redirect("detector:result", pk=record.pk)


# ── Image upload view ─────────────────────────────────────────────────────────

@require_http_methods(["GET", "POST"])
def upload_image_view(request):
    """
    GET  — render the same upload page (image tab active)
    POST — validate, save temp image file, run single-image inference, store in DB

    Reuses the SAME trained model and MTCNN instance as video upload.
    See run_image_inference() docstring for the temporal-analysis limitation.
    """
    if request.method == "GET":
        recent = PredictionRecord.objects.all()[:5]
        return render(request, "detector/upload.html", {
            "recent": recent,
            "active_tab": "image",
        })

    # ── POST ──────────────────────────────────────────────────────────────────
    uploaded = request.FILES.get("image")

    error = _validate_image_upload(uploaded)
    if error:
        recent = PredictionRecord.objects.all()[:5]
        return render(request, "detector/upload.html", {
            "error":      error,
            "recent":     recent,
            "active_tab": "image",
        })

    upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext       = Path(uploaded.name).suffix.lower()
    temp_name = f"upload_{uuid.uuid4().hex}{ext}"
    temp_path = upload_dir / temp_name

    try:
        with open(temp_path, "wb") as f:
            for chunk in uploaded.chunks(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Saved image upload: {temp_path} ({uploaded.size:,} bytes)")
    except OSError as e:
        logger.error(f"File save failed: {e}")
        recent = PredictionRecord.objects.all()[:5]
        return render(request, "detector/upload.html", {
            "error":      "File upload failed — disk may be full. Please try again.",
            "recent":     recent,
            "active_tab": "image",
        })

    t0     = time.time()
    result = run_image_inference(
        image_path=str(temp_path),
        media_root=str(settings.MEDIA_ROOT),
    )
    elapsed = round(time.time() - t0, 2)

    if result.get("error"):
        recent = PredictionRecord.objects.all()[:5]
        return render(request, "detector/upload.html", {
            "error":      result["error"],
            "recent":     recent,
            "active_tab": "image",
        })

    record = PredictionRecord.objects.create(
        filename          = uploaded.name,
        label             = result.get("label",             "UNCERTAIN"),
        confidence        = result.get("confidence",        0.0),
        real_probability  = result.get("real_prob",          0.0),
        fake_probability  = result.get("fake_prob",          0.0),
        frames_analyzed   = result.get("frames_analyzed",    1),
        face_grid_path    = result.get("face_grid_path",     None),
        gradcam_grid_path = result.get("gradcam_grid_path",  None),
        detection_mode    = result.get("detection_mode",     "image"),
        processing_time   = elapsed,
        file_size_bytes   = uploaded.size,
        error_message     = result.get("error",              None),
        created_at        = timezone.now(),
    )
    logger.info(
        f"Image prediction saved: id={record.pk} | {record.label} | "
        f"{record.confidence:.1f}% | {elapsed}s"
    )

    return redirect("detector:result", pk=record.pk)


# ── Result view ───────────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def result_view(request, pk: int):
    """
    Serve prediction result from database record.
    Each result has a permanent URL: /result/<pk>/
    Works for both video and image detection modes.
    """
    record = get_object_or_404(PredictionRecord, pk=pk)

    if record.error_message:
        return render(request, "detector/upload.html", {
            "error":  record.error_message,
            "recent": PredictionRecord.objects.all()[:5],
        })

    context = {
        "record":           record,
        "label":            record.label,
        "confidence":       record.confidence,
        "real_prob":        record.real_probability,
        "fake_prob":        record.fake_probability,
        "frames_analyzed":  record.frames_analyzed,
        "face_grid_url":    record.face_grid_url,
        "gradcam_grid_url": record.gradcam_grid_url,
        "filename":         record.filename,
        "processing_time":  record.processing_time,
        "created_at":       record.created_at,
        "record_id":        record.pk,
        "detection_mode":   record.detection_mode,
    }
    return render(request, "detector/result.html", context)


# ── PDF report download view ──────────────────────────────────────────────────

@require_http_methods(["GET"])
def download_report_view(request, pk: int):
    """
    Generate and return a downloadable PDF report for a PredictionRecord.

    Reuses existing saved images (face_grid_path, gradcam_grid_path) from
    disk — nothing is recomputed from the model. If an image is missing
    (e.g. Grad-CAM failed silently during inference, as already handled
    elsewhere), the PDF still generates with whatever is available.
    """
    record = get_object_or_404(PredictionRecord, pk=pk)

    try:
        pdf_bytes = render_report_pdf(record)
    except Exception as e:
        # Don't let a PDF rendering issue take down the result page flow —
        # log it and send the user back to the result page with an error.
        logger.error(f"PDF report generation failed for record {pk}: {e}")
        return render(request, "detector/result.html", {
            "record":           record,
            "label":            record.label,
            "confidence":       record.confidence,
            "real_prob":        record.real_probability,
            "fake_prob":        record.fake_probability,
            "frames_analyzed":  record.frames_analyzed,
            "face_grid_url":    record.face_grid_url,
            "gradcam_grid_url": record.gradcam_grid_url,
            "filename":         record.filename,
            "processing_time":  record.processing_time,
            "created_at":       record.created_at,
            "record_id":        record.pk,
            "detection_mode":   record.detection_mode,
            "report_error":     "PDF report could not be generated. Please try again.",
        })

    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    safe_name = Path(record.filename).stem or "detectra_report"
    response["Content-Disposition"] = (
        f'attachment; filename="detectra_report_{safe_name}_{record.pk}.pdf"'
    )
    logger.info(f"PDF report downloaded for record id={record.pk}")
    return response


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
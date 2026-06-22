# webapp/detector/report.py
"""
PDF report generation for PredictionRecord results.

Design intent:
  - Reuses EXISTING saved images (face_grid_path, gradcam_grid_path) from disk.
    Nothing here touches the model, MTCNN, or Grad-CAM computation — those
    already ran during inference and wrote files to MEDIA_ROOT.
  - Built with WeasyPrint (HTML/CSS -> PDF) so the visual language can mirror
    result.html's existing palette without hand-positioning boxes.
  - Must never crash if an image is missing (e.g. Grad-CAM failed silently,
    as already handled elsewhere in the pipeline) — the PDF still generates
    with whatever is available.
"""

import logging
from pathlib import Path
from typing import Optional

from django.conf import settings
from django.template.loader import render_to_string

logger = logging.getLogger(__name__)


def _resolve_media_path(relative_path: str) -> Optional[str]:
    """
    Convert a path stored relative to MEDIA_ROOT into an absolute
    file:// -compatible path WeasyPrint can read directly from disk,
    but ONLY if the file actually exists.

    Returns None if relative_path is falsy or the file is missing,
    so the template can skip that section instead of breaking.
    """
    if not relative_path:
        return None

    abs_path = Path(settings.MEDIA_ROOT) / relative_path
    if not abs_path.is_file():
        logger.warning(f"Report: expected image not found on disk: {abs_path}")
        return None

    # WeasyPrint accepts plain absolute filesystem paths in <img src="...">
    # when base_url is set, but using as_uri() is the most reliable
    # cross-platform form (handles Windows drive letters too).
    return abs_path.resolve().as_uri()


def build_report_context(record) -> dict:
    """
    Assemble the template context for a single PredictionRecord.
    Pure data preparation — no PDF rendering happens here.
    """
    return {
        "record":            record,
        "label":             record.label,
        "confidence":        record.confidence,
        "real_prob":         record.real_probability,
        "fake_prob":         record.fake_probability,
        "frames_analyzed":   record.frames_analyzed,
        "filename":          record.filename,
        "detection_mode":    record.detection_mode,
        "created_at":        record.created_at,
        "processing_time":   record.processing_time,
        "face_grid_uri":     _resolve_media_path(record.face_grid_path),
        "gradcam_grid_uri":  _resolve_media_path(record.gradcam_grid_path),
    }


def render_report_pdf(record) -> bytes:
    """
    Render the PDF report for a PredictionRecord and return raw PDF bytes.
    Raises only on genuine rendering failure (e.g. WeasyPrint not installed
    correctly) — missing optional images are handled gracefully upstream.
    """
    # Imported lazily so the rest of the app (and `manage.py` commands that
    # never touch reports) doesn't hard-require WeasyPrint + its system libs.
    from weasyprint import HTML

    context = build_report_context(record)
    html_string = render_to_string("detector/report_pdf.html", context)

    pdf_bytes = HTML(string=html_string).write_pdf()
    return pdf_bytes
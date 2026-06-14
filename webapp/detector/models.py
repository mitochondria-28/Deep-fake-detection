# webapp/detector/models.py
"""
PredictionRecord — stores every inference result in the database.
Gives the system a full audit trail of all predictions made.
Works identically with SQLite (dev) and PostgreSQL (prod).
"""

from django.db import models
from django.utils import timezone


class PredictionRecord(models.Model):

    LABEL_CHOICES = [
        ("REAL",      "Real"),
        ("FAKE",      "Fake"),
        ("UNCERTAIN", "Uncertain"),
    ]

    # ── Core result fields ────────────────────────────────────────────────────
    filename         = models.CharField(max_length=255)
    label            = models.CharField(max_length=10, choices=LABEL_CHOICES)
    confidence       = models.FloatField(help_text="Confidence % of predicted class")
    real_probability = models.FloatField(help_text="P(Real) as percentage")
    fake_probability = models.FloatField(help_text="P(Fake) as percentage")
    frames_analyzed  = models.IntegerField(default=0)

    # ── Face grid image path ──────────────────────────────────────────────────
    # Stored as a relative path under MEDIA_ROOT
    face_grid_path   = models.CharField(max_length=500, blank=True, null=True)

    # ── Metadata ──────────────────────────────────────────────────────────────
    created_at       = models.DateTimeField(default=timezone.now, db_index=True)
    processing_time  = models.FloatField(
        null=True, blank=True,
        help_text="Inference time in seconds"
    )
    file_size_bytes  = models.BigIntegerField(null=True, blank=True)
    error_message    = models.TextField(blank=True, null=True)

    class Meta:
        ordering        = ["-created_at"]
        verbose_name    = "Prediction Record"
        verbose_name_plural = "Prediction Records"

    def __str__(self):
        return f"{self.filename} → {self.label} ({self.confidence:.1f}%)"

    @property
    def is_uncertain(self):
        return self.label == "UNCERTAIN"

    @property
    def face_grid_url(self):
        from django.conf import settings
        if self.face_grid_path:
            return settings.MEDIA_URL + self.face_grid_path
        return None
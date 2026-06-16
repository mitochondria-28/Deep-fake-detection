# webapp/detector/apps.py
"""
Loads the Deepfakedetection model and MTCNN into module-level globals
exactly once when Django starts. All requests share these objects.
"""

import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detector"

    def ready(self):
        """Called once by Django when the app is fully loaded."""
        # Avoid double-loading in Django's dev server auto-reloader
        import os
        if os.environ.get("RUN_MAIN") != "true" and \
           os.environ.get("DJANGO_SETTINGS_MODULE"):
            # Load on first real worker process only
            pass

        try:
            from detector import inference
            inference.load_models()
            logger.info("✓ Deepfakedetection model and MTCNN loaded at startup")
        except Exception as e:
            logger.error(f"✗ Failed to load model at startup: {e}")
            raise
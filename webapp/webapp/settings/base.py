# webapp/webapp/settings/base.py
"""
Shared settings used by both development and production.
Environment-specific values are loaded from .env via python-dotenv.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
DETECTRA_ROOT = BASE_DIR.parent

# Add detectra root so we can import model/, dataloader/, preprocessing/
if str(DETECTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(DETECTRA_ROOT))

# ── Security ──────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "fallback-insecure-key-set-env-var"
)
DEBUG        = os.environ.get("DEBUG", "True") == "True"
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "*").split(",")

# ── Apps ──────────────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "detector",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",   # serve static files
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF      = "webapp.urls"
WSGI_APPLICATION  = "webapp.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# ── Static and media files ────────────────────────────────────────────────────
STATIC_URL   = "/static/"
STATIC_ROOT  = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL    = "/media/"
MEDIA_ROOT   = BASE_DIR / "media"

# ── Upload and inference settings ─────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB   = 100
ALLOWED_VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv"}
CHECKPOINT_PATH      = os.environ.get(
    "CHECKPOINT_PATH",
    str(DETECTRA_ROOT / "checkpoints" / "best_acc.pt")
)
CONFIDENCE_THRESHOLD = 0.75
SEQUENCE_LENGTH      = 20

# ── Session ───────────────────────────────────────────────────────────────────
SESSION_ENGINE       = "django.contrib.sessions.backends.db"
SESSION_COOKIE_AGE   = 3600   # 1 hour

DEFAULT_AUTO_FIELD   = "django.db.models.BigAutoField"
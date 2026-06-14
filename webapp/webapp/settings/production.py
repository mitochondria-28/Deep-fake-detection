# webapp/webapp/settings/production.py
import os
import dj_database_url
from .base import *

DEBUG        = False
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "*").split(",")

# Railway provides DATABASE_URL automatically
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    DATABASES = {
        "default": dj_database_url.parse(
            DATABASE_URL,
            conn_max_age=60,
        )
    }
else:
    # Fallback to SQLite if no DATABASE_URL set
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME":   BASE_DIR / "db.sqlite3",
        }
    }

SESSION_COOKIE_SECURE  = True
CSRF_COOKIE_SECURE     = True
X_FRAME_OPTIONS        = "DENY"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level":    "INFO",
    },
}
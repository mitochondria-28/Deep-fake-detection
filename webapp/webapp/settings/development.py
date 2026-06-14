# webapp/webapp/settings/development.py
"""
Development settings — uses SQLite, full DEBUG, no SSL.
"""

from .base import *   # noqa

DEBUG = True

# ── SQLite database (development) ─────────────────────────────────────────────
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME":   BASE_DIR / "db.sqlite3",
    }
}

# ── Relaxed security for local dev ────────────────────────────────────────────
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE    = False
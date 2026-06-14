# webapp/webapp/wsgi.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from django.core.wsgi import get_wsgi_application

# Load env before Django settings
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

env = os.environ.get("DJANGO_ENV", "development")
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    f"webapp.settings.{env}"
)

application = get_wsgi_application()
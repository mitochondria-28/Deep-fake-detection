# webapp/manage.py  — replace the entire file with this
#!/usr/bin/env python
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def main():
    # Load .env so DJANGO_ENV is available
    load_dotenv(Path(__file__).resolve().parent / ".env")

    env = os.environ.get("DJANGO_ENV", "development")
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE",
        f"webapp.settings.{env}"
    )

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Could not import Django. Make sure it is installed."
        ) from exc

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
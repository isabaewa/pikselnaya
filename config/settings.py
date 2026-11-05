import os
from pathlib import Path

from django.core.management.utils import get_random_secret_key

BASE_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DIR = BASE_DIR / "runtime"
RUNTIME_DIR.mkdir(exist_ok=True)
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    key_file = RUNTIME_DIR / "django_secret_key"
    try:
        with key_file.open("x", encoding="utf-8") as file:
            file.write(get_random_secret_key())
    except FileExistsError:
        pass
    SECRET_KEY = key_file.read_text(encoding="utf-8").strip()
DEBUG = True
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]
INSTALLED_APPS = ["django.contrib.staticfiles", "editor"]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]
ROOT_URLCONF = "config.urls"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
}]
SESSION_ENGINE = "django.contrib.sessions.backends.file"
SESSION_FILE_PATH = str(RUNTIME_DIR / "sessions")
Path(SESSION_FILE_PATH).mkdir(parents=True, exist_ok=True)
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 52 * 1024 * 1024
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
MEDIA_URL = "/media/"
MEDIA_ROOT = RUNTIME_DIR / "images"
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
LANGUAGE_CODE = "ru"
TIME_ZONE = "Asia/Yekaterinburg"
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

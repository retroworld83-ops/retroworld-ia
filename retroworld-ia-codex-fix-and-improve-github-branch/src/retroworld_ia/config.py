import os
import secrets
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key, default) or "").strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


BASE_DIR = Path(__file__).resolve().parents[2]
APP_DATA_DIR = Path(os.getenv("APP_DATA_DIR", str(BASE_DIR / "data"))).resolve()
STATIC_DIR = BASE_DIR / "static"
CONV_DIR = APP_DATA_DIR / "conversations"
DB_PATH = Path(os.getenv("APP_DB_PATH", str(APP_DATA_DIR / "retroworld_ia.db"))).resolve()
KNOWLEDGE_PATH = BASE_DIR / "src" / "data" / "knowledge_base.json"

APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
CONV_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-5.2")
OPENAI_REASONING_EFFORT = _env("OPENAI_REASONING_EFFORT", "none")
OPENAI_TEMPERATURE = _env_float("OPENAI_TEMPERATURE", 0.3)
OPENAI_MAX_OUTPUT_TOKENS = _env_int("OPENAI_MAX_OUTPUT_TOKENS", 900)
CHAT_HISTORY_MESSAGES = max(_env_int("CHAT_HISTORY_MESSAGES", 10), 0)
DEBUG_LOGS = _env("DEBUG_LOGS").lower() in {"1", "true", "yes", "on"}
PUBLIC_BASE_URL = _env("PUBLIC_BASE_URL")
SERVER_MODE = _env("SERVER_MODE", "auto").lower()
LOG_BUFFER_MAX = max(_env_int("ADMIN_LOG_BUFFER_MAX", 300), 50)
ALLOW_LEGACY_ADMIN_TOKEN = _env("ADMIN_ENABLE_LEGACY_TOKENS", "false").lower() in {"1", "true", "yes", "on"}
LEGACY_ADMIN_TOKENS = {tok for tok in [_env("ADMIN_API_TOKEN"), _env("ADMIN_DASHBOARD_TOKEN")] if tok}
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in _env("ALLOWED_ORIGINS").split(",") if o.strip()]
PUBLIC_BRANDS_ENV = [b.strip() for b in _env("PUBLIC_BRANDS").split(",") if b.strip()]
FAQ_ENABLED_BRANDS_ENV = [b.strip() for b in _env("FAQ_ENABLED_BRANDS").split(",") if b.strip()]
ADMIN_USERNAME = _env("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = _env("ADMIN_PASSWORD") or _env("ADMIN_DASHBOARD_TOKEN") or _env("ADMIN_API_TOKEN")
ADMIN_PASSWORD_HASH = _env("ADMIN_PASSWORD_HASH")
SECRET_KEY = _env("SECRET_KEY") or secrets.token_hex(32)
SESSION_COOKIE_SECURE = _env("SESSION_COOKIE_SECURE", "").lower() in {"1", "true", "yes", "on"}
LEAD_WEBHOOK_URL = _env("LEAD_WEBHOOK_URL")


def running_on_render() -> bool:
    return _env("RENDER").lower() in {"1", "true", "yes", "on"}


def should_use_gunicorn() -> bool:
    if SERVER_MODE == "gunicorn":
        return True
    if SERVER_MODE == "flask":
        return False
    return running_on_render()


def gunicorn_cmd(port: int) -> list[str]:
    return ["gunicorn", "-w", "2", "-k", "gthread", "-b", f"0.0.0.0:{port}", "app:app"]

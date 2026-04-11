from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional
import traceback

from src.retroworld_ia import config


APP_LOGS = deque(maxlen=config.LOG_BUFFER_MAX)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def record_log(level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
    APP_LOGS.append(
        {
            "ts": now_str(),
            "level": level,
            "message": message,
            "context": context or {},
        }
    )


def log(*args: Any) -> None:
    message = " ".join(str(arg) for arg in args)
    record_log("debug", message)
    if config.DEBUG_LOGS:
        print("[DBG]", *args, flush=True)


def log_error(message: str, err: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None) -> None:
    payload = dict(context or {})
    if err is not None:
        payload["error"] = str(err)
        payload["traceback"] = traceback.format_exc(limit=5)
    record_log("error", message, payload)
    print("[ERR]", message, payload.get("error", ""), flush=True)

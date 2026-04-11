import secrets

from flask import jsonify, redirect, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from src.retroworld_ia import config
from src.retroworld_ia.services.conversations import get_db
from src.retroworld_ia.services.logging_store import now_str, record_log


def bootstrap_admin_user() -> None:
    password_hash = config.ADMIN_PASSWORD_HASH
    if not password_hash and config.ADMIN_PASSWORD:
        password_hash = generate_password_hash(config.ADMIN_PASSWORD)
    if not password_hash:
        record_log("warning", "Admin bootstrap skipped: no password configured")
        return
    with get_db() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO admin_users (username, password_hash, created_at, updated_at, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (config.ADMIN_USERNAME, password_hash, now_str(), now_str()),
        )
        conn.execute(
            "UPDATE admin_users SET password_hash = ?, updated_at = ? WHERE username = ?",
            (password_hash, now_str(), config.ADMIN_USERNAME),
        )


def ensure_csrf_token() -> str:
    if not session.get("csrf_token"):
        session["csrf_token"] = secrets.token_urlsafe(24)
    return session["csrf_token"]


def legacy_admin_token_valid() -> bool:
    if not config.ALLOW_LEGACY_ADMIN_TOKEN:
        return False
    token = ""
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    else:
        token = (request.args.get("token") or request.headers.get("X-Admin-Token") or "").strip()
    return bool(token and token in config.LEGACY_ADMIN_TOKENS)


def is_admin_authenticated() -> bool:
    if session.get("admin_user_id") and session.get("admin_username"):
        ensure_csrf_token()
        return True
    return legacy_admin_token_valid()


def require_admin_auth(api: bool = True):
    if is_admin_authenticated():
        return None
    if api:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    return redirect("/admin/login")


def require_csrf():
    if legacy_admin_token_valid():
        return None
    expected = session.get("csrf_token") or ""
    received = (request.headers.get("X-CSRF-Token") or "").strip()
    if not expected or not secrets.compare_digest(expected, received):
        return jsonify({"ok": False, "error": "csrf_failed"}), 403
    return None


def attempt_login(username: str, password: str):
    with get_db() as conn:
        user = conn.execute("SELECT id, username, password_hash, is_active FROM admin_users WHERE username = ?", (username,)).fetchone()
    if not user or not user["is_active"] or not check_password_hash(user["password_hash"], password):
        return None
    session.clear()
    session["admin_user_id"] = user["id"]
    session["admin_username"] = user["username"]
    record_log("info", "Admin login", {"username": user["username"]})
    return {"username": user["username"], "csrf_token": ensure_csrf_token()}


def create_admin_user(username: str, password: str) -> dict | None:
    username = (username or "").strip()
    if not username or not password:
        return None
    password_hash = generate_password_hash(password)
    with get_db() as conn:
        exists = conn.execute("SELECT id FROM admin_users WHERE username = ?", (username,)).fetchone()
        if exists:
            return None
        conn.execute(
            "INSERT INTO admin_users (username, password_hash, created_at, updated_at, is_active) VALUES (?, ?, ?, ?, 1)",
            (username, password_hash, now_str(), now_str()),
        )
    record_log("info", "Admin user created", {"username": username})
    return {"username": username}


def update_admin_user(user_id: int, password: str | None = None, is_active: int | None = None) -> bool:
    updates = []
    params = []
    if password:
        updates.append("password_hash = ?")
        params.append(generate_password_hash(password))
    if is_active is not None:
        updates.append("is_active = ?")
        params.append(1 if is_active else 0)
    if not updates:
        return False
    updates.append("updated_at = ?")
    params.append(now_str())
    params.append(user_id)
    with get_db() as conn:
        conn.execute(f"UPDATE admin_users SET {', '.join(updates)} WHERE id = ?", params)
    record_log("info", "Admin user updated", {"user_id": user_id})
    return True

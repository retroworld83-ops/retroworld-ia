import json
from flask import Blueprint, jsonify, redirect, request, send_from_directory, session

from src.retroworld_ia import config
from src.retroworld_ia.services.auth import (
    attempt_login,
    create_admin_user,
    ensure_csrf_token,
    is_admin_authenticated,
    require_admin_auth,
    require_csrf,
    update_admin_user,
)
from src.retroworld_ia.services.conversations import analytics_snapshot, fetch_conversation, get_db, list_admin_users, list_conversations, list_leads
from src.retroworld_ia.services.knowledge import BRANDS, FAQ_ENABLED_BRANDS, PUBLIC_BRANDS, get_knowledge_editor_payload, load_public_faq, normalize_brand, save_knowledge_brand
from src.retroworld_ia.services.logging_store import APP_LOGS, now_str


admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/admin/login", methods=["GET"])
def admin_login_page():
    if is_admin_authenticated():
        return redirect("/admin")
    return send_from_directory(str(config.STATIC_DIR), "admin-login.html")


@admin_bp.route("/admin", methods=["GET"])
def admin_page():
    auth = require_admin_auth(api=False)
    if auth is not None:
        return auth
    ensure_csrf_token()
    return send_from_directory(str(config.STATIC_DIR), "admin.html")


@admin_bp.route("/admin/faq", methods=["GET"])
def admin_faq_page():
    auth = require_admin_auth(api=False)
    if auth is not None:
        return auth
    ensure_csrf_token()
    return send_from_directory(str(config.STATIC_DIR), "admin-faq.html")


@admin_bp.route("/admin/knowledge", methods=["GET"])
def admin_knowledge_page():
    auth = require_admin_auth(api=False)
    if auth is not None:
        return auth
    ensure_csrf_token()
    return send_from_directory(str(config.STATIC_DIR), "admin-knowledge.html")


@admin_bp.route("/admin/api/auth/login", methods=["POST"])
def admin_api_login():
    payload = request.get_json(silent=True) or {}
    result = attempt_login((payload.get("username") or "").strip(), payload.get("password") or "")
    if not result:
        return jsonify({"ok": False, "error": "invalid_credentials"}), 401
    return jsonify({"ok": True, **result})


@admin_bp.route("/admin/api/auth/logout", methods=["POST"])
def admin_api_logout():
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    session.clear()
    return jsonify({"ok": True, "logged_out": True})


@admin_bp.route("/admin/api/session", methods=["GET"])
def admin_api_session():
    if not is_admin_authenticated():
        return jsonify({"ok": True, "authenticated": False, "login_url": "/admin/login"})
    return jsonify({"ok": True, "authenticated": True, "username": session.get("admin_username", "legacy-token"), "csrf_token": ensure_csrf_token()})


@admin_bp.route("/admin/api/brands", methods=["GET"])
def admin_api_brands():
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    faq_only = (request.args.get("faq_only") or "").strip() == "1"
    brand_ids = FAQ_ENABLED_BRANDS if faq_only else list(BRANDS.keys())
    items = []
    for bid in brand_ids:
        cfg = BRANDS.get(bid, {})
        items.append({"id": bid, "name": cfg.get("name", bid), "display_name": cfg.get("short", cfg.get("name", bid))})
    return jsonify({"ok": True, "brands": items})


@admin_bp.route("/admin/api/diag", methods=["GET"])
def admin_diag():
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    faq_files = []
    for bid in BRANDS.keys():
        path = config.STATIC_DIR / f"faq_{bid}.json"
        faq_files.append({"brand_id": bid, "file": str(path.relative_to(config.BASE_DIR)), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0})
    with get_db() as conn:
        conversations_count = conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
        users_count = conn.execute("SELECT COUNT(*) AS n FROM admin_users WHERE is_active = 1").fetchone()["n"]
    return jsonify(
        {
            "ok": True,
            "time": now_str(),
            "openai_configured": bool(config.OPENAI_API_KEY),
            "brands": list(BRANDS.keys()),
            "faq_enabled_brands": FAQ_ENABLED_BRANDS,
            "public_brands": PUBLIC_BRANDS,
            "server_mode": config.SERVER_MODE,
            "running_on_render": config.running_on_render(),
            "allowed_origins": config.ALLOWED_ORIGINS,
            "conversations_count": conversations_count,
            "faq_files": faq_files,
            "recent_logs": list(APP_LOGS)[-80:],
            "users_count": users_count,
            "storage": "sqlite",
        }
    )


@admin_bp.route("/admin/api/conversations", methods=["GET"])
def admin_list_conversations():
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    return jsonify({"ok": True, "items": list_conversations()})


@admin_bp.route("/admin/api/conversation/<conv_id>", methods=["GET"])
def admin_get_conversation(conv_id: str):
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    return jsonify({"ok": True, "conversation": fetch_conversation(conv_id)})


@admin_bp.route("/admin/api/analytics", methods=["GET"])
def admin_analytics():
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    return jsonify({"ok": True, "analytics": analytics_snapshot()})


@admin_bp.route("/admin/api/leads", methods=["GET"])
def admin_leads():
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    return jsonify({"ok": True, "items": list_leads(100)})


@admin_bp.route("/admin/api/users", methods=["GET", "POST"])
def admin_users():
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    if request.method == "GET":
        return jsonify({"ok": True, "items": list_admin_users()})
    csrf = require_csrf()
    if csrf is not None:
        return csrf
    payload = request.get_json(silent=True) or {}
    created = create_admin_user((payload.get("username") or "").strip(), payload.get("password") or "")
    if not created:
        return jsonify({"ok": False, "error": "user_create_failed"}), 400
    return jsonify({"ok": True, **created})


@admin_bp.route("/admin/api/users/<int:user_id>", methods=["PUT"])
def admin_user_update(user_id: int):
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    csrf = require_csrf()
    if csrf is not None:
        return csrf
    payload = request.get_json(silent=True) or {}
    changed = update_admin_user(user_id, password=(payload.get("password") or None), is_active=payload.get("is_active"))
    if not changed:
        return jsonify({"ok": False, "error": "user_update_failed"}), 400
    return jsonify({"ok": True, "updated": True})


@admin_bp.route("/admin/api/export.csv", methods=["GET"])
def admin_export_csv():
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    import csv
    import io
    from flask import make_response
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["conversation_id", "brand_id", "ts", "role", "content", "flags"])
    for conv in list_conversations():
        full = fetch_conversation(conv["id"])
        flags = "|".join(full.get("flags", []))
        brand_id = (full.get("meta") or {}).get("brand_id", "")
        for message in full.get("messages") or []:
            writer.writerow([full["id"], brand_id, message.get("ts", ""), message.get("role", ""), (message.get("content", "") or "").replace("\n", " "), flags])
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = 'attachment; filename="conversations.csv"'
    return response


@admin_bp.route("/admin/api/faq/<brand_id>", methods=["GET", "PUT"])
def admin_faq_by_brand(brand_id: str):
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    bid = normalize_brand(brand_id)
    if bid not in BRANDS:
        return jsonify({"ok": False, "error": "unknown_brand"}), 400
    if request.method == "GET":
        return jsonify(load_public_faq(bid))
    csrf = require_csrf()
    if csrf is not None:
        return csrf
    payload = request.get_json(silent=True) or {}
    items = payload.get("items")
    if not isinstance(items, list):
        return jsonify({"ok": False, "error": "items must be a list"}), 400
    clean = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = (item.get("question") or item.get("q") or "").strip()
        answer = (item.get("answer") or item.get("a") or "").strip()
        tags = item.get("tags") or []
        if question and answer:
            clean.append({"question": question, "answer": answer, "tags": tags if isinstance(tags, list) else []})
    out = {"brand": bid, "updated": now_str(), "items": clean}
    path = config.STATIC_DIR / f"faq_{bid}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    if bid == "runningman":
        legacy = config.STATIC_DIR / "static" / "faq_runningman.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    return jsonify(out)


@admin_bp.route("/admin/api/faq/get", methods=["GET"])
def admin_faq_get_compat():
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    bid = normalize_brand(request.args.get("brand_id") or "retroworld")
    if bid not in BRANDS:
        return jsonify({"ok": False, "error": "unknown_brand"}), 400
    return jsonify({"ok": True, "kb": load_public_faq(bid)})


@admin_bp.route("/admin/api/faq/save", methods=["POST"])
def admin_faq_save_compat():
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    csrf = require_csrf()
    if csrf is not None:
        return csrf
    payload = request.get_json(silent=True) or {}
    bid = normalize_brand(payload.get("brand") or payload.get("brand_id") or "retroworld")
    if bid not in BRANDS:
        return jsonify({"ok": False, "error": "unknown_brand"}), 400
    response = admin_faq_by_brand(bid)
    if isinstance(response, tuple):
        return response
    saved = response.get_json()
    return jsonify({"ok": True, "saved": True, "updated": saved.get("updated"), "count": len(saved.get("items", []))})


@admin_bp.route("/admin/api/knowledge/<brand_id>", methods=["GET", "PUT"])
def admin_knowledge_by_brand(brand_id: str):
    auth = require_admin_auth(api=True)
    if auth is not None:
        return auth
    bid = normalize_brand(brand_id)
    if bid not in BRANDS:
        return jsonify({"ok": False, "error": "unknown_brand"}), 400
    if request.method == "GET":
        return jsonify({"ok": True, "brand": get_knowledge_editor_payload(bid)})
    csrf = require_csrf()
    if csrf is not None:
        return csrf
    payload = request.get_json(silent=True) or {}
    saved = save_knowledge_brand(bid, payload)
    return jsonify({"ok": True, "brand": saved})

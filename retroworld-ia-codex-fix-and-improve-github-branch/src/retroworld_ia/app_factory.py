from flask import Flask, jsonify, make_response, request
from werkzeug.exceptions import HTTPException

from src.retroworld_ia import config
from src.retroworld_ia.routes.admin import admin_bp
from src.retroworld_ia.routes.public import public_bp
from src.retroworld_ia.services.auth import bootstrap_admin_user
from src.retroworld_ia.services.conversations import init_db, migrate_legacy_json_conversations
from src.retroworld_ia.services.logging_store import log_error, record_log


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(config.STATIC_DIR), static_url_path="/static")
    app.secret_key = config.SECRET_KEY
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=config.SESSION_COOKIE_SECURE,
    )

    @app.after_request
    def apply_cors(resp):
        origin = (request.headers.get("Origin") or "").strip().rstrip("/")
        if origin:
            same_origin = request.host_url.rstrip("/")
            allow = False
            if origin == same_origin:
                allow = True
            elif config.ALLOWED_ORIGINS:
                allow = origin in config.ALLOWED_ORIGINS
            else:
                allow = origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:")
            if allow:
                resp.headers["Access-Control-Allow-Origin"] = origin
                resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Brand-Id, X-CSRF-Token"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS"
        return resp

    @app.errorhandler(HTTPException)
    def handle_http_exception(err: HTTPException):
        status_code = err.code or 500
        record_log("warning", "HTTP exception", {"path": request.path, "method": request.method, "status": status_code, "error": err.name})
        if request.path.startswith("/admin/api/") or request.path.startswith("/chat"):
            payload = "http_error"
            if status_code == 404:
                payload = "not_found"
            elif status_code == 405:
                payload = "method_not_allowed"
            return jsonify({"ok": False, "error": payload}), status_code
        if status_code == 404:
            return make_response("not found", 404)
        return make_response(err.description or "http error", status_code)

    @app.errorhandler(Exception)
    def handle_unexpected_error(err: Exception):
        log_error("Unhandled server exception", err, {"path": request.path, "method": request.method})
        if request.path.startswith("/admin/api/") or request.path.startswith("/chat"):
            return jsonify({"ok": False, "error": "internal_error"}), 500
        return make_response("internal server error", 500)

    app.register_blueprint(public_bp)
    app.register_blueprint(admin_bp)

    init_db()
    bootstrap_admin_user()
    migrate_legacy_json_conversations()
    return app

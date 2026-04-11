from flask import Blueprint, current_app, jsonify, make_response, redirect, request, send_from_directory

from src.retroworld_ia import config
from src.retroworld_ia.services.ai import (
    add_disclaimer_if_needed,
    append_retroworld_links_if_missing,
    build_openai_messages,
    enforce_no_reservation_promises,
    openai_answer,
    openai_ready,
)
from src.retroworld_ia.services.conversations import append_message, create_or_load_conversation, new_conv_id, upsert_conversation
from src.retroworld_ia.services.knowledge import (
    BRAND_ID_DEFAULT,
    BRANDS,
    FAQ_ENABLED_BRANDS,
    PUBLIC_BRANDS,
    booking_intent,
    build_system_prompt,
    detect_brand_from_text,
    load_public_faq,
    normalize_brand,
    public_brand_payload,
)
from src.retroworld_ia.services.logging_store import now_str


public_bp = Blueprint("public", __name__)


def detect_brand_from_origin() -> str | None:
    candidates = [
        (request.headers.get("Origin") or "").strip().lower(),
        (request.headers.get("Referer") or "").strip().lower(),
        (request.host or "").strip().lower(),
    ]
    for candidate in candidates:
        for bid, cfg in BRANDS.items():
            for domain in cfg.get("domains", []) or []:
                if domain and domain.lower() in candidate:
                    return bid
    return None


def get_brand_id(payload: dict) -> str:
    for candidate in [
        normalize_brand(payload.get("brand_id") or ""),
        normalize_brand(request.headers.get("X-Brand-Id") or ""),
        normalize_brand(request.args.get("brand_id") or request.args.get("brand") or ""),
    ]:
        if candidate in BRANDS:
            return candidate
    from_text = detect_brand_from_text(payload.get("message") or "")
    if from_text in BRANDS:
        return from_text
    from_origin = detect_brand_from_origin()
    if from_origin in BRANDS:
        return from_origin
    return BRAND_ID_DEFAULT


@public_bp.route("/", methods=["GET"])
def index():
    return redirect("/static/chat-widget.html")


@public_bp.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "time": now_str(),
            "openai_configured": openai_ready(),
            "brands": list(BRANDS.keys()),
            "faq_enabled_brands": FAQ_ENABLED_BRANDS,
            "public_brands": PUBLIC_BRANDS,
            "server_mode": config.SERVER_MODE,
            "running_on_render": config.running_on_render(),
            "storage": "sqlite",
            "db_path": str(config.DB_PATH.name),
            "admin_auth_mode": "session_password",
        }
    )


@public_bp.route("/brands.json", methods=["GET"])
def brands_json():
    return jsonify({"items": [public_brand_payload(bid) for bid in PUBLIC_BRANDS], "base_url": config.PUBLIC_BASE_URL})


@public_bp.route("/knowledge.json", methods=["GET"])
def knowledge_json():
    bid = normalize_brand(request.args.get("brand_id") or request.args.get("brand") or BRAND_ID_DEFAULT)
    if bid not in BRANDS:
        return jsonify({"ok": False, "error": "unknown_brand"}), 404
    return jsonify({"ok": True, "brand": public_brand_payload(bid)})


@public_bp.route("/faq", methods=["GET"])
def faq_page():
    bid = normalize_brand(request.args.get("brand_id") or request.args.get("brand") or BRAND_ID_DEFAULT)
    if bid not in FAQ_ENABLED_BRANDS:
        return make_response("FAQ indisponible pour le moment.", 404)
    return redirect(f"/static/chat-widget.html?tab=faq&brand={bid}")


@public_bp.route("/faq/<brand_id>", methods=["GET"], strict_slashes=False)
def faq_page_by_brand(brand_id: str):
    bid = normalize_brand(brand_id)
    if bid not in FAQ_ENABLED_BRANDS:
        return jsonify({"brand": bid, "items": [], "updated": now_str()}), 404
    data = load_public_faq(bid)
    return jsonify({"brand": bid, "updated": data.get("updated", now_str()), "items": data.get("items", [])})


@public_bp.route("/faq.json", methods=["GET"])
def faq_json():
    bid = normalize_brand(request.args.get("brand_id") or request.args.get("brand") or BRAND_ID_DEFAULT)
    if bid == "all":
        payload = [{"brand": brand_id, "items": load_public_faq(brand_id).get("items", [])} for brand_id in FAQ_ENABLED_BRANDS]
        return jsonify({"items": payload, "updated": now_str()})
    if bid not in FAQ_ENABLED_BRANDS:
        return jsonify({"brand": bid, "items": [], "updated": now_str()}), 404
    data = load_public_faq(bid)
    return jsonify({"brand": bid, "updated": data.get("updated", now_str()), "items": data.get("items", [])})


@public_bp.route("/faq_retroworld.json", methods=["GET"])
def faq_retroworld_alias():
    return send_from_directory(str(config.STATIC_DIR), "faq_retroworld.json")


@public_bp.route("/faq_runningman.json", methods=["GET"])
def faq_runningman_alias():
    return send_from_directory(str(config.STATIC_DIR), "faq_runningman.json")


@public_bp.route("/faq_enigmaniac.json", methods=["GET"])
def faq_enigmaniac_alias():
    return send_from_directory(str(config.STATIC_DIR), "faq_enigmaniac.json")


@public_bp.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "message manquant"}), 400
    msg = (payload.get("message") or "").strip()
    if not msg:
        return jsonify({"ok": False, "error": "message manquant"}), 400

    brand_id = get_brand_id(payload)
    conv_id = (payload.get("conversation_id") or "").strip() or new_conv_id(prefix=brand_id[:2] if brand_id else "rw")
    conversation = create_or_load_conversation(conv_id, brand_id)
    append_message(conversation, "user", msg, extra={"source": (payload.get("metadata") or {}).get("source", ""), "intents": ["reservation"] if booking_intent(msg) else []})

    if not openai_ready():
        answer = "Le service IA n'est pas configure (OPENAI_API_KEY manquante)."
        append_message(conversation, "assistant", answer, extra={"brand_id": brand_id, "flags": ["openai_missing"]})
        upsert_conversation(conversation)
        return jsonify({"ok": True, "conversation_id": conv_id, "brand_id": brand_id, "answer": answer})

    system_prompt = build_system_prompt(brand_id, msg)
    history = conversation.get("messages", [])[:-1]
    raw_answer = openai_answer(build_openai_messages(system_prompt, history, msg))
    safe_answer, promised = enforce_no_reservation_promises(raw_answer)
    safe_answer = add_disclaimer_if_needed(safe_answer, brand_id, msg)
    if brand_id == "retroworld":
        safe_answer = append_retroworld_links_if_missing(msg, safe_answer)
    flags = ["promesse_resa"] if promised else []
    append_message(conversation, "assistant", safe_answer, extra={"brand_id": brand_id, "flags": flags})
    upsert_conversation(conversation)
    return jsonify({"ok": True, "conversation_id": conv_id, "brand_id": brand_id, "answer": safe_answer})


@public_bp.route("/chat/<brand_id>", methods=["POST", "OPTIONS"], strict_slashes=False)
def chat_by_brand(brand_id: str):
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "message manquant"}), 400
    payload["brand_id"] = normalize_brand(brand_id)
    return chat()


@public_bp.route("/static/<path:filename>", methods=["GET"])
def static_files(filename: str):
    return send_from_directory(str(config.STATIC_DIR), filename)


@public_bp.route("/robots.txt", methods=["GET"])
def robots_txt():
    path = config.STATIC_DIR / "robots.txt"
    if path.exists():
        return send_from_directory(str(config.STATIC_DIR), "robots.txt")
    return make_response("User-agent: *\nAllow: /\n", 200, {"Content-Type": "text/plain; charset=utf-8"})

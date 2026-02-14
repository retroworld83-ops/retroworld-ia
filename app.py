import os
import json
import time
import uuid
import logging
import importlib.util
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Flask, request, jsonify, make_response, send_from_directory
from flask_cors import CORS
from openai import OpenAI

# -------------------------
# .env (optionnel)
# -------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# -------------------------
# LOGS
# -------------------------
debug_logs = os.environ.get("DEBUG_LOGS", "false").strip().lower() == "true"
log_level = logging.DEBUG if debug_logs else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVICE_NAME = os.environ.get("SERVICE_NAME", "retroworld-ia").strip()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# DATA / STORAGE
# -------------------------
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data")).strip()
os.makedirs(DATA_DIR, exist_ok=True)

CONV_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONV_DIR, exist_ok=True)

CONV_COOKIE_NAME = os.environ.get("CONV_COOKIE_NAME", "rw_conv_id").strip()

# -------------------------
# ADMIN (token)
# -------------------------
ADMIN_DASHBOARD_TOKEN = (os.environ.get("ADMIN_DASHBOARD_TOKEN") or os.environ.get("ADMIN_API_TOKEN") or "").strip()

def _extract_admin_token(req) -> str:
    # 1) Authorization: Bearer xxx
    token = (req.headers.get("Authorization") or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token:
        return token

    # 2) X-Admin-Token: xxx
    token = (req.headers.get("X-Admin-Token") or "").strip()
    if token:
        return token

    # 3) ?token=xxx
    token = (req.args.get("token") or "").strip()
    if token:
        return token

    return ""

def _require_admin(req) -> bool:
    if not ADMIN_DASHBOARD_TOKEN:
        return False
    return _extract_admin_token(req) == ADMIN_DASHBOARD_TOKEN

# -------------------------
# OPENAI
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o").strip()

try:
    OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.3"))
except ValueError:
    OPENAI_TEMPERATURE = 0.3

try:
    OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "900"))
except ValueError:
    OPENAI_MAX_OUTPUT_TOKENS = 900

OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "").strip().lower()

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY manquante !")
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# SYSTEM PROMPT (robuste)
# 1) src/data/system_data.py si présent
# 2) sinon fallback: kb_retroworld.json + kb_runningman.json
# -------------------------
def _read_json_file(path: str) -> Optional[dict]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_system_prompt() -> str:
    # 1) system_data.py si présent
    candidates = [
        os.path.join(BASE_DIR, "src", "data", "system_data.py"),
        os.path.join("/app", "src", "data", "system_data.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location("system_data", path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore
                    prompt = getattr(module, "SYSTEM_PROMPT", None)
                    if isinstance(prompt, str) and prompt.strip():
                        logger.info("SYSTEM_PROMPT chargé depuis %s", path)
                        return prompt
            except Exception as e:
                logger.critical("Import SYSTEM_PROMPT échoué (%s): %s", path, str(e))

    # 2) fallback KB JSON (ce que tu as dans ton ZIP)
    kb_retro = _read_json_file(os.path.join(BASE_DIR, "kb_retroworld.json")) or {}
    kb_run = _read_json_file(os.path.join(BASE_DIR, "kb_runningman.json")) or {}

    if not kb_retro and not kb_run:
        logger.critical("Aucune base KB trouvée (kb_retroworld.json / kb_runningman.json).")
        return "Erreur configuration : Base de connaissance introuvable."

    parts: List[str] = []
    parts.append("Vous êtes l'assistant IA officiel du Pôle Loisirs à Draguignan.")
    parts.append("Règles générales : vouvoiement, réponses précises, pas d'invention, rester prudent sur les disponibilités.")
    parts.append("Adresse Retroworld : 815 avenue Pierre Brossolette, 83300 Draguignan. Horaires : mardi à dimanche 11h-22h.")

    # Retroworld KB
    if kb_retro:
        identite = kb_retro.get("identite", {})
        prompt = kb_retro.get("prompt", {})
        instr = kb_retro.get("instructions_generales", [])
        parts.append("\n--- RETROWORLD (VR / Escape VR / Quiz / Salle enfant / Anniversaires / Fidélité) ---")
        if isinstance(identite, dict):
            parts.append(f"Nom: {identite.get('nom','Retroworld France')}")
            parts.append(str(identite.get("role","")).strip())
        if isinstance(instr, list) and instr:
            parts.append("Règles Retroworld :")
            for line in instr:
                parts.append(f"- {line}")
        if isinstance(prompt, dict):
            # On met les sections clés du prompt existant dans ton JSON
            for k in [
                "role_general",
                "schema_universel",
                "etape_4_anniversaires",
                "etape_5_devis",
                "reservation_non_confirmee",
                "gestion_liens_reservation",
                "style_reponses_jeux",
                "fidelite",
                "redirection_runningman",
                "redirection_enigmaniac",
                "catalogue_logic",
                "tonalite",
            ]:
                v = prompt.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(f"\n[{k}]\n{v.strip()}")

        # Injecter directement les blocs de tarifs/activités si présents
        activites = kb_retro.get("activites")
        if activites is not None:
            parts.append("\n[Activités & tarifs Retroworld - source KB JSON]")
            parts.append(json.dumps(activites, ensure_ascii=False, indent=2))

        # Catalogue jeux si présent
        for key in ["catalogue_jeux_vr", "catalogue_escape_vr"]:
            if key in kb_retro:
                parts.append(f"\n[{key}]\n{json.dumps(kb_retro[key], ensure_ascii=False)}")

    # Runningman KB
    if kb_run:
        parts.append("\n--- RUNNINGMAN (Action Game / Kids Zone / extras) ---")
        parts.append("Règle : pour l’action game, les clients doivent contacter Runningman directement.")
        parts.append("Contact Runningman : 04 98 09 30 59 | Site : https://www.runningmangames.fr")
        parts.append(json.dumps(kb_run, ensure_ascii=False, indent=2))

    final_prompt = "\n".join([p for p in parts if p.strip()])
    logger.info("SYSTEM_PROMPT généré depuis KB JSON (fallback).")
    return final_prompt

SYSTEM_PROMPT = load_system_prompt()

# -------------------------
# FLASK + CORS + STATIC
# -------------------------
app = Flask(__name__, static_folder="static")

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
if allowed_origins:
    origins_list = [o.strip() for o in allowed_origins.split(",") if o.strip()]
    if "*" in origins_list:
        CORS(app, supports_credentials=True)
    else:
        CORS(app, resources={r"/*": {"origins": origins_list}}, supports_credentials=True)
else:
    CORS(app, supports_credentials=True)

# -------------------------
# HELPERS CONV
# -------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _conv_path(conv_id: str) -> str:
    return os.path.join(CONV_DIR, f"{conv_id}.json")

def _safe_read_json(path: str, default: Any) -> Any:
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _safe_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _new_conv_id() -> str:
    return "conv_" + uuid.uuid4().hex[:18]

def _get_conv_id() -> str:
    cid = (request.cookies.get(CONV_COOKIE_NAME) or "").strip()
    if cid:
        return cid
    cid = (request.headers.get("X-Conv-Id") or "").strip()
    if cid:
        return cid
    data = request.get_json(silent=True) or {}
    cid = (data.get("conv_id") or "").strip()
    if cid:
        return cid
    return _new_conv_id()

def _load_conv_doc(conv_id: str) -> Dict[str, Any]:
    return _safe_read_json(_conv_path(conv_id), {"created_at": _now_iso(), "messages": []}) or {"created_at": _now_iso(), "messages": []}

def _load_history(conv_id: str) -> List[Dict[str, Any]]:
    doc = _load_conv_doc(conv_id)
    msgs = doc.get("messages", [])
    return msgs if isinstance(msgs, list) else []

def _append_history(conv_id: str, role: str, content: str) -> None:
    if role not in ("user", "assistant"):
        return
    content = (content or "").strip()
    if not content:
        return

    doc_path = _conv_path(conv_id)
    doc = _load_conv_doc(conv_id)
    if "created_at" not in doc:
        doc["created_at"] = _now_iso()

    msgs = doc.get("messages", [])
    if not isinstance(msgs, list):
        msgs = []

    msgs.append({"role": role, "content": content, "ts": time.time()})
    if len(msgs) > 30:
        msgs = msgs[-30:]

    doc["messages"] = msgs
    doc["updated_at"] = _now_iso()
    _safe_write_json(doc_path, doc)

# -------------------------
# ADMIN helpers (listing + preview)
# -------------------------
def _infer_brand(messages: List[Dict[str, Any]]) -> str:
    text = " ".join([(m.get("content") or "") for m in messages[-10:] if isinstance(m, dict)])
    t = text.lower()
    if "runningman" in t or "action game" in t or "kids zone" in t:
        return "runningman"
    return "retroworld"

def _last_preview(messages: List[Dict[str, Any]]) -> str:
    # preview = dernier message user si possible
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            return m["content"].strip()[:180]
    # sinon dernier message tout court
    for m in reversed(messages):
        if isinstance(m.get("content"), str) and m["content"].strip():
            return m["content"].strip()[:180]
    return ""

def _list_conversations(limit: int = 300) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        files = [f for f in os.listdir(CONV_DIR) if f.endswith(".json")]
        files.sort(reverse=True)
        for f in files[:limit]:
            conv_id = f[:-5]
            doc = _safe_read_json(os.path.join(CONV_DIR, f), {}) or {}
            msgs = doc.get("messages", [])
            msgs_list = msgs if isinstance(msgs, list) else []
            items.append({
                "conv_id": conv_id,
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "count": len(msgs_list),
                "brand": _infer_brand(msgs_list),
                "preview": _last_preview(msgs_list),
            })
    except Exception as e:
        logger.error("List conversations failed: %s", str(e))
    return items

# -------------------------
# ROUTES PUBLIC
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "service": SERVICE_NAME,
        "model": OPENAI_MODEL,
        "version": "3.5-admin-compatible",
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    conv_id = _get_conv_id()

    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message vide"}), 400

        _append_history(conv_id, "user", user_message)
        stored_msgs = _load_history(conv_id)

        messages_payload: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        cleaned = []
        for m in stored_msgs[-12:]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                cleaned.append({"role": role, "content": content.strip()})
        messages_payload.extend(cleaned)

        kwargs = dict(
            model=OPENAI_MODEL,
            messages=messages_payload,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )

        if OPENAI_REASONING_EFFORT in ("low", "medium", "high", "none"):
            kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT  # type: ignore

        try:
            response = client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("reasoning_effort", None)
            response = client.chat.completions.create(**kwargs)

        bot_reply = response.choices[0].message.content or ""
        _append_history(conv_id, "assistant", bot_reply)

        resp = make_response(jsonify({"reply": bot_reply, "conv_id": conv_id}))
        resp.set_cookie(CONV_COOKIE_NAME, conv_id, max_age=60 * 60 * 24 * 30, httponly=False, samesite="Lax")
        return resp

    except Exception as e:
        logger.error("ERREUR /chat conv_id=%s : %s", conv_id, str(e))
        resp = make_response(jsonify({
            "reply": (
                "Désolé, je rencontre un souci technique momentané.\n"
                "Retroworld : 04 94 47 94 64\n"
                "Runningman : 04 98 09 30 59"
            ),
            "conv_id": conv_id
        }), 500)
        resp.set_cookie(CONV_COOKIE_NAME, conv_id, max_age=60 * 60 * 24 * 30, httponly=False, samesite="Lax")
        return resp

# -------------------------
# ADMIN UI (sert static/admin.html)
# -------------------------
@app.route("/admin", methods=["GET"])
def admin_page():
    if not _require_admin(request):
        return "Unauthorized", 401
    return send_from_directory(app.static_folder, "admin.html")

# -------------------------
# ADMIN API (compat avec ton static/admin.html)
# -------------------------
@app.route("/admin/api/diag", methods=["GET"])
def admin_diag():
    if not _require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401

    count_files = 0
    try:
        count_files = len([f for f in os.listdir(CONV_DIR) if f.endswith(".json")])
    except Exception:
        pass

    return jsonify({
        "ok": True,
        "service": SERVICE_NAME,
        "version": "3.5-admin-compatible",
        "model": OPENAI_MODEL,
        "conv_dir": CONV_DIR,
        "conversations": count_files,
        "has_system_prompt": bool(SYSTEM_PROMPT and "Erreur configuration" not in SYSTEM_PROMPT),
    })

@app.route("/admin/api/conversations", methods=["GET"])
def admin_conversations():
    if not _require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(_list_conversations(limit=300))

@app.route("/admin/api/conversation/<conv_id>", methods=["GET"])
def admin_conversation(conv_id: str):
    if not _require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    doc = _load_conv_doc(conv_id)
    msgs = doc.get("messages", [])
    msgs_list = msgs if isinstance(msgs, list) else []
    return jsonify({
        "conv_id": conv_id,
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "count": len(msgs_list),
        "brand": _infer_brand(msgs_list),
        "messages": msgs_list,
    })

@app.route("/admin/api/delete/<conv_id>", methods=["POST"])
def admin_delete(conv_id: str):
    if not _require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    path = _conv_path(conv_id)
    try:
        if os.path.exists(path):
            os.remove(path)
        return jsonify({"ok": True, "conv_id": conv_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

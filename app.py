import os
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------
# CONFIG
# -------------------------
load_dotenv()

debug_logs = os.environ.get("DEBUG_LOGS", "false").strip().lower() == "true"
log_level = logging.DEBUG if debug_logs else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVICE_NAME = os.environ.get("SERVICE_NAME", "retroworld-ia").strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data")).strip()
os.makedirs(DATA_DIR, exist_ok=True)

CONV_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONV_DIR, exist_ok=True)

CONV_COOKIE_NAME = os.environ.get("CONV_COOKIE_NAME", "rw_conv_id").strip()

ADMIN_DASHBOARD_TOKEN = (os.environ.get("ADMIN_DASHBOARD_TOKEN") or os.environ.get("ADMIN_API_TOKEN") or "").strip()

# OpenAI settings (Render)
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

OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "").strip().lower()  # optionnel

# CORS
app = Flask(__name__)
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
if allowed_origins:
    origins_list = [o.strip() for o in allowed_origins.split(",") if o.strip()]
    if "*" in origins_list:
        CORS(app, supports_credentials=True)
        logger.info("CORS: wildcard (*) autorisé")
    else:
        CORS(app, resources={r"/*": {"origins": origins_list}}, supports_credentials=True)
        logger.info("CORS: origines autorisées = %s", origins_list)
else:
    CORS(app, supports_credentials=True)
    logger.warning("CORS: ALLOWED_ORIGINS vide -> accès ouvert (à éviter en prod)")

# OpenAI client
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY manquante !")
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# SYSTEM PROMPT
# -------------------------
# Le fichier est dans src/data/system_data.py
try:
    from src.data.system_data import SYSTEM_PROMPT
except Exception:
    # Fallback si PYTHONPATH ne contient pas src
    try:
        import sys
        src_dir = os.path.join(BASE_DIR, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from data.system_data import SYSTEM_PROMPT  # type: ignore
    except Exception as e:
        logger.critical("Impossible d'importer SYSTEM_PROMPT: %s", str(e))
        SYSTEM_PROMPT = "Erreur configuration : Base de connaissance introuvable."

# -------------------------
# HELPERS STORAGE
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
    # 1) cookie
    cid = (request.cookies.get(CONV_COOKIE_NAME) or "").strip()
    if cid:
        return cid
    # 2) header custom optionnel
    cid = (request.headers.get("X-Conv-Id") or "").strip()
    if cid:
        return cid
    # 3) body optionnel
    data = request.get_json(silent=True) or {}
    cid = (data.get("conv_id") or "").strip()
    if cid:
        return cid
    # 4) sinon nouveau
    return _new_conv_id()

def _load_history(conv_id: str) -> List[Dict[str, str]]:
    doc = _safe_read_json(_conv_path(conv_id), {"messages": []}) or {"messages": []}
    msgs = doc.get("messages", [])
    return msgs if isinstance(msgs, list) else []

def _append_history(conv_id: str, role: str, content: str) -> None:
    if role not in ("user", "assistant"):
        return
    content = (content or "").strip()
    if not content:
        return

    doc_path = _conv_path(conv_id)
    doc = _safe_read_json(doc_path, {"created_at": _now_iso(), "messages": []}) or {"created_at": _now_iso(), "messages": []}
    if "created_at" not in doc:
        doc["created_at"] = _now_iso()
    msgs = doc.get("messages", [])
    if not isinstance(msgs, list):
        msgs = []

    msgs.append({"role": role, "content": content, "ts": time.time()})

    # Limit: garder les 30 derniers messages pour éviter de grossir
    if len(msgs) > 30:
        msgs = msgs[-30:]

    doc["messages"] = msgs
    doc["updated_at"] = _now_iso()
    _safe_write_json(doc_path, doc)

def _require_admin(req) -> bool:
    if not ADMIN_DASHBOARD_TOKEN:
        return False
    token = (req.headers.get("Authorization") or "").strip()
    # accepte "Bearer xxx" ou "xxx"
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token == ADMIN_DASHBOARD_TOKEN

# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "service": SERVICE_NAME,
        "model": OPENAI_MODEL,
        "version": "3.2-storage"
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    """
    Stockage Q/R serveur activé:
      - conv_id via cookie rw_conv_id (auto)
      - history front optionnel (mais pas nécessaire)
    """
    conv_id = _get_conv_id()

    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        client_history = data.get("history")

        if not user_message:
            return jsonify({"error": "Message vide"}), 400

        # 1) Historique serveur
        stored_msgs = _load_history(conv_id)

        # 2) Optionnel: si le front fournit un historique, on peut l'utiliser,
        #    mais on préfère le serveur pour la cohérence.
        #    (on garde quand même stored_msgs comme source principale)
        if isinstance(client_history, list) and len(client_history) > 0 and len(stored_msgs) == 0:
            # premier démarrage: on peut “amorcer” le serveur avec le history du client
            for msg in client_history[-10:]:
                if isinstance(msg, dict):
                    r = msg.get("role")
                    c = msg.get("content")
                    if r in ("user", "assistant") and isinstance(c, str) and c.strip():
                        _append_history(conv_id, r, c.strip())
            stored_msgs = _load_history(conv_id)

        # Ajout du message utilisateur dans l’historique serveur
        _append_history(conv_id, "user", user_message)

        # Recharge après append (garantit l'ordre exact stocké)
        stored_msgs = _load_history(conv_id)

        # Construire payload OpenAI
        messages_payload: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # On garde les 12 derniers messages (pour limiter tokens)
        # (comme chaque message peut être long)
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

        # Optionnel: reasoning effort (si supporté par votre modèle/SDK)
        if OPENAI_REASONING_EFFORT in ("low", "medium", "high", "none"):
            kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT  # type: ignore

        try:
            response = client.chat.completions.create(**kwargs)
        except TypeError:
            # si le SDK ne supporte pas reasoning_effort
            kwargs.pop("reasoning_effort", None)
            response = client.chat.completions.create(**kwargs)

        bot_reply = response.choices[0].message.content or ""

        # Stocker la réponse assistant
        _append_history(conv_id, "assistant", bot_reply)

        if debug_logs:
            logger.debug("conv_id=%s | msg=%s", conv_id, user_message)
            logger.debug("reply=%s", bot_reply[:400])

        resp = make_response(jsonify({"reply": bot_reply, "conv_id": conv_id}))
        # cookie 30 jours
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

# --- ADMIN: lire l'historique d'une conversation ---
@app.route("/admin/history/<conv_id>", methods=["GET"])
def admin_history(conv_id: str):
    if not _require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    msgs = _load_history(conv_id)
    return jsonify({"conv_id": conv_id, "messages": msgs})

# --- ADMIN: reset conversation ---
@app.route("/admin/reset/<conv_id>", methods=["POST"])
def admin_reset(conv_id: str):
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

import os
import json
import time
import uuid
import logging
import importlib.util
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Flask, request, jsonify
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# CORS (Render)
# -------------------------
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
# STORAGE Q/R (serveur)
# -------------------------
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data")).strip()
os.makedirs(DATA_DIR, exist_ok=True)

CONV_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONV_DIR, exist_ok=True)

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _conv_path(conversation_id: str) -> str:
    safe = "".join([c for c in conversation_id if c.isalnum() or c in ("_", "-")])[:80]
    return os.path.join(CONV_DIR, f"{safe}.json")

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

def _new_conversation_id(prefix: str = "c") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:18]}"

def _load_conv_doc(conversation_id: str) -> Dict[str, Any]:
    return _safe_read_json(_conv_path(conversation_id), {"created_at": _now_iso(), "messages": []}) or {"created_at": _now_iso(), "messages": []}

def _append_message(conversation_id: str, role: str, content: str) -> None:
    if role not in ("user", "assistant"):
        return
    content = (content or "").strip()
    if not content:
        return

    path = _conv_path(conversation_id)
    doc = _load_conv_doc(conversation_id)

    msgs = doc.get("messages", [])
    if not isinstance(msgs, list):
        msgs = []

    msgs.append({"role": role, "content": content, "ts": time.time()})

    # Limites pour éviter d'exploser en taille
    if len(msgs) > 40:
        msgs = msgs[-40:]

    doc["messages"] = msgs
    doc["updated_at"] = _now_iso()
    _safe_write_json(path, doc)

def _get_last_messages(conversation_id: str, n: int = 12) -> List[Dict[str, str]]:
    doc = _load_conv_doc(conversation_id)
    msgs = doc.get("messages", [])
    if not isinstance(msgs, list):
        return []
    cleaned: List[Dict[str, str]] = []
    for m in msgs[-n:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()})
    return cleaned

# -------------------------
# KB / FAQ
# -------------------------
def _read_json_file(path: str) -> Optional[dict]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

KB_RETRO_PATH = os.path.join(BASE_DIR, "kb_retroworld.json")
KB_RUN_PATH = os.path.join(BASE_DIR, "kb_runningman.json")

def build_system_prompt_retroworld() -> str:
    """
    Prompt Retroworld: essaye d'utiliser system_data.py si présent,
    sinon fallback: kb_retroworld.json (si présent), sinon prompt minimal.
    """
    # 1) system_data.py si dispo
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
                        logger.info("SYSTEM_PROMPT (Retroworld) chargé depuis %s", path)
                        return prompt
            except Exception as e:
                logger.error("Import system_data.py échoué (%s): %s", path, str(e))

    # 2) kb_retroworld.json si dispo
    kb = _read_json_file(KB_RETRO_PATH) or {}
    if kb:
        # On ne “refabrique” pas tout, on injecte le KB brut + règles fortes.
        return (
            "Vous êtes l'assistant officiel Retroworld (VR/escape VR/quiz/anniversaires/fidélité) à Draguignan.\n"
            "Règles: vouvoiement, réponses précises, pas d'invention, pas de fausses promesses.\n"
            "Si un client parle de Runningman (Action Game / Kids Zone), rediriger vers Runningman.\n"
            "Important anniversaires: ne jamais inventer (pas de crêpes par défaut si non indiqué par la base).\n"
            "Liens de réservation: ne les donner que si la personne est décidée.\n\n"
            "BASE DE CONNAISSANCE (JSON):\n"
            + json.dumps(kb, ensure_ascii=False)
        )

    # 3) fallback minimal
    return (
        "Vous êtes l'assistant officiel Retroworld (VR/escape VR/quiz) à Draguignan.\n"
        "Règles: vouvoiement, réponses précises, pas d'invention.\n"
        "Contact Retroworld: 04 94 47 94 64. Horaires: mardi à dimanche, 11h-22h.\n"
        "Si besoin Runningman: 04 98 09 30 59.\n"
    )

SYSTEM_PROMPT_RETRO = build_system_prompt_retroworld()

def get_faq_items_retroworld() -> List[Dict[str, Any]]:
    """
    Doit renvoyer: items: [{question, answer, tags[]}...]
    Compatible widget: data.items
    """
    kb = _read_json_file(KB_RETRO_PATH) or {}

    # 1) FAQ structurée si présente
    for key in ("faq", "questions_reponses", "qna"):
        block = kb.get(key)
        if isinstance(block, list) and block:
            items = []
            for it in block:
                if not isinstance(it, dict):
                    continue
                q = (it.get("question") or it.get("q") or "").strip()
                a = (it.get("answer") or it.get("a") or it.get("reponse") or "").strip()
                tags = it.get("tags") if isinstance(it.get("tags"), list) else []
                tags = [str(t) for t in tags][:12]
                if q and a:
                    items.append({"question": q, "answer": a, "tags": tags})
            if items:
                return items

    # 2) Fallback “safe” si ta KB ne contient pas de FAQ dédiée
    # (Ça garantit que ton widget affichera toujours quelque chose)
    return [
        {
            "question": "Quels sont vos horaires ?",
            "answer": "Nous sommes ouverts du mardi au dimanche, de 11h à 22h.",
            "tags": ["horaires", "ouverture"]
        },
        {
            "question": "Où êtes-vous situés ?",
            "answer": "815 avenue Pierre Brossolette, 83300 Draguignan.",
            "tags": ["adresse", "draguignan"]
        },
        {
            "question": "Quels sont les tarifs VR et Escape VR ?",
            "answer": "Jeux VR : 15 €/joueur. Escape Game VR : 30 €/joueur. (Des suppléments peuvent s'appliquer hors créneaux.)",
            "tags": ["tarifs", "vr", "escape vr"]
        },
        {
            "question": "Faites-vous des quiz ?",
            "answer": "Oui, quiz interactifs : 8€ (30min), 15€ (60min), 20€ (90min), jusqu’à 12 joueurs.",
            "tags": ["quiz", "tarifs"]
        },
        {
            "question": "Comment fonctionne la fidélité ?",
            "answer": "1 partie VR = 1 point, 1 escape VR = 2 points. 10 points = 1 VR offerte, 20 points = 1 escape VR offert. Les anniversaires ne donnent pas de points.",
            "tags": ["fidélité", "points", "récompenses"]
        },
        {
            "question": "Runningman (Action Game), je réserve comment ?",
            "answer": "Pour l’Action Game Runningman, il faut contacter Runningman directement : 04 98 09 30 59.",
            "tags": ["runningman", "action game", "réservation"]
        },
    ]

# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "online",
        "service": "retroworld-ia",
        "version": "widget-compatible",
        "model": OPENAI_MODEL
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# ✅ FAQ endpoint attendu par ton widget
@app.route("/faq/retroworld", methods=["GET"])
def faq_retroworld():
    items = get_faq_items_retroworld()
    return jsonify({"ok": True, "items": items})

# (optionnel futur)
@app.route("/faq/runningman", methods=["GET"])
def faq_runningman():
    kb = _read_json_file(KB_RUN_PATH) or {}
    # fallback simple
    items = []
    for key in ("faq", "questions_reponses", "qna"):
        block = kb.get(key)
        if isinstance(block, list) and block:
            for it in block:
                if isinstance(it, dict):
                    q = (it.get("question") or it.get("q") or "").strip()
                    a = (it.get("answer") or it.get("a") or it.get("reponse") or "").strip()
                    tags = it.get("tags") if isinstance(it.get("tags"), list) else []
                    tags = [str(t) for t in tags][:12]
                    if q and a:
                        items.append({"question": q, "answer": a, "tags": tags})
            break

    if not items:
        items = [
            {
                "question": "Comment réserver l’Action Game Runningman ?",
                "answer": "Les clients doivent contacter Runningman directement au 04 98 09 30 59 ou via https://www.runningmangames.fr.",
                "tags": ["réservation", "runningman", "action game"]
            }
        ]
    return jsonify({"ok": True, "items": items})

# ✅ Chat endpoint attendu par ton widget
@app.route("/chat/retroworld", methods=["POST"])
def chat_retroworld():
    """
    Payload du widget:
    {
      message: str,
      user_id: str,
      conversation_id: str (optionnel),
      metadata: { ... } (optionnel)
    }
    Retour attendu:
    { reply: str, conversation_id: str }
    """
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        user_id = (data.get("user_id") or "").strip()
        conversation_id = (data.get("conversation_id") or "").strip() or _new_conversation_id("rw")

        if not message:
            return jsonify({"reply": "Message vide.", "conversation_id": conversation_id}), 400

        # Stockage user message
        _append_message(conversation_id, "user", message)

        # Construire messages OpenAI
        history = _get_last_messages(conversation_id, n=12)
        messages_payload = [{"role": "system", "content": SYSTEM_PROMPT_RETRO}]
        messages_payload.extend(history)

        kwargs = dict(
            model=OPENAI_MODEL,
            messages=messages_payload,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        if OPENAI_REASONING_EFFORT in ("low", "medium", "high", "none"):
            kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT  # type: ignore

        try:
            resp = client.chat.completions.create(**kwargs)
        except TypeError:
            # SDK/model ne supporte pas reasoning_effort
            kwargs.pop("reasoning_effort", None)
            resp = client.chat.completions.create(**kwargs)

        reply = (resp.choices[0].message.content or "").strip()
        if not reply:
            reply = "Je n’ai pas pu générer de réponse, pouvez-vous reformuler ?"

        # Stockage assistant reply
        _append_message(conversation_id, "assistant", reply)

        if debug_logs:
            logger.debug("chat/retroworld user_id=%s conv_id=%s msg=%s", user_id, conversation_id, message)
            logger.debug("reply=%s", reply[:400])

        return jsonify({"reply": reply, "conversation_id": conversation_id})

    except Exception as e:
        logger.error("Erreur /chat/retroworld: %s", str(e))
        return jsonify({
            "reply": "Désolé, je rencontre un souci technique. Vous pouvez nous appeler au 04 94 47 94 64.",
            "conversation_id": (request.get_json(silent=True) or {}).get("conversation_id") or _new_conversation_id("rw")
        }), 500

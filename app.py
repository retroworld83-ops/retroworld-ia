from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory, make_response, send_file
from flask_cors import CORS

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

# =========================================================
# CONFIG
# =========================================================

SERVICE_NAME = os.getenv("SERVICE_NAME", "retroworld-ia")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

KB_DIR = os.getenv("KB_DIR", BASE_DIR)

PORT = int(os.getenv("PORT", "10000"))

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-5.2").strip()
OPENAI_REASONING_EFFORT = (os.getenv("OPENAI_REASONING_EFFORT") or "none").strip().lower()
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS") or os.getenv("OPENAI_MAX_TOKENS") or "900")

ADMIN_DASHBOARD_TOKEN = (os.getenv("ADMIN_DASHBOARD_TOKEN") or os.getenv("ADMIN_API_TOKEN") or "").strip()
USER_HISTORY_TOKEN = (os.getenv("USER_HISTORY_TOKEN") or "").strip()

LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(SERVICE_NAME)

app = Flask(__name__, static_folder="static")

# -------------------------
# CORS: utilise ALLOWED_ORIGINS (Render)
# -------------------------
allowed = (os.getenv("ALLOWED_ORIGINS") or "").strip()
if not allowed:
    CORS(app, supports_credentials=True)
else:
    origins = [o.strip() for o in allowed.split(",") if o.strip()]
    if "*" in origins:
        CORS(app, supports_credentials=True)
    else:
        CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=True)

CONV_COOKIE_NAME = os.getenv("CONV_COOKIE_NAME", "rw_conv_id")
START_SIGNAL = "__start__"

# =========================================================
# OpenAI client
# =========================================================

_openai_client = None
_openai_init_error: Optional[str] = None
try:
    if OPENAI_API_KEY:
        from openai import OpenAI  # type: ignore
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    _openai_init_error = str(e)
    logger.warning("OpenAI client init failed: %s", e)
    _openai_client = None

# =========================================================
# Helpers
# =========================================================

def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _new_conversation_id() -> str:
    return "conv_" + datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

def _safe_read_json(path: str, default: Any) -> Any:
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _safe_write_json(path: str, data: Any) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.warning("write json failed (%s): %s", path, e)
        return False

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def strip_markdown_simple(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"\*\*(.*?)\*\*", r"\1")            # **gras**
    s = re.sub(r"`([^`]*)`", r"\1")                # `code`
    s = re.sub(r"^\s*-\s+", "• ", s, flags=re.M)   # - liste
    s = re.sub(r"^\s*\*\s+", "• ", s, flags=re.M)
    return s

def _now_local() -> datetime:
    tz = (os.getenv("TZ") or "Europe/Paris").strip()
    if ZoneInfo:
        try:
            return datetime.now(ZoneInfo(tz))
        except Exception:
            pass
    return datetime.utcnow()

# =========================================================
# User index: user_id -> conversation_id
# =========================================================

USER_INDEX_PATH = os.path.join(DATA_DIR, "user_index.json")

def get_user_index() -> Dict[str, str]:
    data = _safe_read_json(USER_INDEX_PATH, {}) or {}
    return data if isinstance(data, dict) else {}

def set_user_conversation(user_id: str, conversation_id: str) -> None:
    if not user_id or not conversation_id:
        return
    idx = get_user_index()
    idx[str(user_id)] = str(conversation_id)
    _safe_write_json(USER_INDEX_PATH, idx)

def get_user_conversation(user_id: str) -> Optional[str]:
    if not user_id:
        return None
    idx = get_user_index()
    return idx.get(str(user_id))

# =========================================================
# KB cache
# =========================================================

@dataclass
class _KBCacheEntry:
    ts: float
    kb: Dict[str, Any]

_KB_CACHE: Dict[str, _KBCacheEntry] = {}
_KB_CACHE_TTL_SEC = 30.0

def load_kb(brand: str) -> Dict[str, Any]:
    brand = (brand or "").lower().strip()
    if brand not in ("retroworld", "runningman"):
        brand = "retroworld"

    now = time.time()
    entry = _KB_CACHE.get(brand)
    if entry and (now - entry.ts) < _KB_CACHE_TTL_SEC:
        return entry.kb

    filename = f"kb_{brand}.json"
    path = os.path.join(KB_DIR, filename)
    kb = _safe_read_json(path, {"brand": brand})
    if not isinstance(kb, dict):
        kb = {"brand": brand}

    _KB_CACHE[brand] = _KBCacheEntry(ts=now, kb=kb)
    return kb

def save_kb(brand: str, kb: Dict[str, Any]) -> bool:
    brand = (brand or "").lower().strip()
    if brand not in ("retroworld", "runningman"):
        brand = "retroworld"
    filename = f"kb_{brand}.json"
    path = os.path.join(KB_DIR, filename)
    ok = _safe_write_json(path, kb)
    if ok:
        _KB_CACHE[brand] = _KBCacheEntry(ts=time.time(), kb=kb)
    return ok

# =========================================================
# Intents + routing
# =========================================================

def _is_price_intent(text: str) -> bool:
    t = _norm(text)
    triggers = [
        "prix", "tarif", "tarifs", "combien", "estimation", "devis",
        "ça coûte", "ca coute", "coute", "coût", "cout"
    ]
    return any(k in t for k in triggers)

def _contains_date_word(text: str) -> bool:
    t = _norm(text)
    if any(k in t for k in ["aujourd", "demain", "après-demain", "apres-demain", "apres demain", "ce soir", "cet aprem", "cette aprem"]):
        return True
    for wd in ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]:
        if re.search(rf"\b{wd}\b", t):
            return True
    if re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", t):
        return True
    if re.search(r"\b(\d{1,2})/(\d{1,2})(/(20\d{2}))?\b", t):
        return True
    if re.search(r"\bdans\s+\d{1,2}\s+jour", t):
        return True
    return False

def _extract_time(text: str) -> Optional[Tuple[int, int]]:
    t = _norm(text)
    m = re.search(r"\b(\d{1,2})\s*h\s*(\d{2})?\b", t)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or "0")
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return (hh, mm)

    m = re.search(r"\b(\d{1,2})\s*:\s*(\d{2})\b", t)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return (hh, mm)

    return None

def _is_reservation_intent(text: str, history: Optional[List[Dict[str, Any]]] = None) -> bool:
    t = _norm(text)
    explicit = any(k in t for k in ["réserver", "reservation", "réservation", "dispo", "disponible", "créneau", "creneau", "bloquer", "confirmer", "rdv", "rendez-vous", "venir à"])
    has_date = _contains_date_word(t)
    has_time = _extract_time(t) is not None

    if explicit or has_date:
        return True
    if has_time and has_date:
        return True

    if has_time and history:
        recent = []
        for m in reversed(history):
            if isinstance(m, dict) and m.get("role") == "user":
                recent.append(str(m.get("content") or ""))
            if len(recent) >= 6:
                break
        for ut in recent:
            ut_n = _norm(ut)
            if any(k in ut_n for k in ["réserver", "reservation", "réservation", "dispo", "créneau", "creneau", "aujourd", "demain", "samedi", "dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi"]):
                return True

    return False

def _is_just_question_mark(text: str) -> bool:
    t = (text or "").strip()
    return t in ["?", "??", "???"]

def detect_owner_from_text(text: str) -> str:
    t = _norm(text)
    if "retroworld" in t:
        return "retroworld"
    if "runningman" in t or "running man" in t:
        return "runningman"
    if "escape" in t and "vr" in t:
        return "retroworld"
    if "vr" in t or "casque" in t or "meta quest" in t or "quest" in t:
        return "retroworld"
    if "quiz" in t:
        return "retroworld"
    if "goûter" in t or "gouter" in t or "gâteau" in t or "gateau" in t or "anniversaire" in t:
        return "retroworld"
    if "fidélité" in t or "fidelite" in t or "points" in t or "qr" in t:
        return "retroworld"
    if "salle enfant" in t or "salle enfants" in t or ("salle" in t and "enfant" in t):
        return "runningman"
    if "game zone" in t or "gamezone" in t or "action game" in t:
        return "runningman"
    if "escape" in t and "vr" not in t:
        return "runningman"
    return "auto"

# =========================================================
# Facts blocks
# =========================================================

def common_orientation_block() -> str:
    return (
        "RÉPARTITION (OFFICIELLE) :\n"
        "- Retroworld : VR, Escape VR, quiz, goûter à volonté + gâteau (anniversaires), fidélité/points.\n"
        "- Runningman : Game Zone, Escape game non-VR, Salle enfant.\n"
        "- Même bâtiment : 815 avenue Pierre Brossolette, 83300 Draguignan.\n"
    )

def retroworld_facts_block() -> str:
    return (
        "RETROWORLD (règles fiables) :\n"
        "- Adresse : 815 avenue Pierre Brossolette, 83300 Draguignan\n"
        "- Téléphone : 04 94 47 94 64\n"
        "- Email : contact@retroworldfrance.com\n"
        "- Site : https://www.retroworldfrance.com\n"
        "- RÈGLE CRÉNEAUX : le chat ne confirme ni ne bloque jamais un créneau.\n"
        "- Tarifs (normal 11h00–20h00 / majoré avant 11h ou après 20h) :\n"
        "  - Escape VR : 30€ normal / 35€ majoré (par joueur)\n"
        "  - Jeux VR / Arcade VR (30 min) : 15€ normal / 20€ majoré (par joueur)\n"
        "  - Quiz 30 min : 8€ normal / 12€ majoré (par joueur)\n"
        "  - Quiz 60 min : 15€ normal / 20€ majoré (par joueur)\n"
        "  - Quiz 90 min : 20€ normal / 25€ majoré (par joueur)\n"
        "IMPORTANT : ne donner un lien que si l'utilisateur le demande explicitement.\n"
    )

def runningman_facts_block(kb: Dict[str, Any]) -> str:
    ident = kb.get("identite", {}) or {}
    loc = (ident.get("localisation", {}) or {})
    contact = (ident.get("contact", {}) or {})

    tarif = ((kb.get("tarification", {}) or {}).get("action_game", {}) or {})
    enfant = (tarif.get("enfant", {}) or {})
    adulte = (tarif.get("adulte_accompagnateur", {}) or {})

    return (
        "RUNNINGMAN (règles fiables) :\n"
        "- Activités : Game Zone, Escape game non-VR, Salle enfant\n"
        f"- Adresse : {loc.get('adresse_complete', '815 avenue Pierre Brossolette, 83300 Draguignan, France')}\n"
        f"- Contact : {contact.get('telephone', '04 98 09 30 59')} / {contact.get('site_web', 'https://runningmangames.fr')}\n"
        f"- Tarifs Action Game (si demandé) : enfant (-12) {enfant.get('prix','15')}€ / adulte {adulte.get('prix','20')}€\n"
        "- RÈGLE CRÉNEAUX : le chat ne confirme ni ne bloque jamais un créneau.\n"
    )

def kb_snippets_retroworld(kb: Dict[str, Any], limit_lines: int = 60) -> List[str]:
    out: List[str] = []
    inst = kb.get("instructions_generales")
    if isinstance(inst, list):
        for it in inst:
            s = str(it).strip()
            if s:
                out.append(f"Règle: {s}")
    return out[:limit_lines]

def kb_snippets_runningman(kb: Dict[str, Any], limit_lines: int = 40) -> List[str]:
    out: List[str] = []
    for k in ["instructions_generales", "anti_erreurs"]:
        arr = kb.get(k)
        if isinstance(arr, list):
            for it in arr:
                s = str(it).strip()
                if s:
                    out.append(s)
    return out[:limit_lines]

# =========================================================
# OpenAI call (Responses API)
# =========================================================

def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    if not _openai_client:
        return (
            "Le service IA est indisponible pour le moment. Vous pouvez contacter Retroworld au 04 94 47 94 64 ou Runningman au 04 98 09 30 59.",
            {"error": "openai_not_ready"},
        )

    kwargs: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": messages,  # OK avec Responses API (messages role/content)
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
        "store": False,
    }

    if OPENAI_REASONING_EFFORT:
        kwargs["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}

    if OPENAI_REASONING_EFFORT == "none":
        kwargs["temperature"] = OPENAI_TEMPERATURE

    resp = _openai_client.responses.create(**kwargs)
    text = (getattr(resp, "output_text", "") or "").strip()

    usage: Dict[str, Any] = {}
    try:
        usage_obj = getattr(resp, "usage", None)
        if usage_obj is not None:
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()  # type: ignore
            elif isinstance(usage_obj, dict):
                usage = usage_obj
    except Exception:
        usage = {}

    return text, usage

# =========================================================
# Conversation storage
# =========================================================

CONV_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONV_DIR, exist_ok=True)

def _conversation_path(conversation_id: str) -> str:
    conversation_id = re.sub(r"[^a-zA-Z0-9_\-]", "", conversation_id or "")
    if not conversation_id:
        conversation_id = _new_conversation_id()
    return os.path.join(CONV_DIR, f"{conversation_id}.json")

def load_conversation_obj(conversation_id: str) -> Dict[str, Any]:
    path = _conversation_path(conversation_id)
    data = _safe_read_json(path, {})
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        data.setdefault("version", 2)
        data.setdefault("id", conversation_id)
        data.setdefault("created", data.get("created") or _utc_iso())
        data.setdefault("messages", [])
        return data
    return {"version": 2, "id": conversation_id, "created": _utc_iso(), "user_id": None, "brand_last": None, "messages": []}

def save_conversation_obj(conversation_id: str, obj: Dict[str, Any]) -> None:
    path = _conversation_path(conversation_id)
    _safe_write_json(path, obj)

def load_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    obj = load_conversation_obj(conversation_id)
    msgs = obj.get("messages") or []
    return msgs if isinstance(msgs, list) else []

def append_conversation_turn(
    conversation_id: str,
    brand_effective: str,
    user_text: str,
    assistant_reply: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    obj = load_conversation_obj(conversation_id)
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        msgs = []
    ts = _utc_iso()

    if (user_text or "").strip():
        msgs.append({"role": "user", "content": user_text, "ts": ts})
    msgs.append({"role": "assistant", "content": str(assistant_reply or ""), "ts": ts})

    obj["version"] = 2
    obj["id"] = conversation_id
    obj.setdefault("created", obj.get("created") or ts)
    obj["brand_last"] = str(brand_effective or "")

    if extra and isinstance(extra, dict):
        obj["extra_last"] = extra
        uid = extra.get("user_id")
        if uid:
            obj["user_id"] = uid

    obj["messages"] = msgs
    save_conversation_obj(conversation_id, obj)

def list_conversations() -> List[str]:
    try:
        files = [f for f in os.listdir(CONV_DIR) if f.endswith(".json")]
        files.sort(reverse=True)
        return [os.path.splitext(f)[0] for f in files]
    except Exception:
        return []

def prune_messages_for_prompt(messages: List[Dict[str, Any]], max_pairs: int = 10) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []
    clipped = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content") is not None]
    clipped = clipped[-(max_pairs * 2):]
    return [{"role": str(m.get("role")), "content": str(m.get("content") or "")} for m in clipped]

# =========================================================
# Tokens
# =========================================================

def _require_admin_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-Admin-Token") or "").strip()
    return bool(tok) and bool(ADMIN_DASHBOARD_TOKEN) and tok == ADMIN_DASHBOARD_TOKEN

def _require_user_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-User-Token") or "").strip()
    return bool(tok) and bool(USER_HISTORY_TOKEN) and tok == USER_HISTORY_TOKEN

# =========================================================
# Payload parsing
# =========================================================

def _payload_text(payload: Dict[str, Any]) -> str:
    msg = payload.get("message")
    if msg is not None:
        return str(msg)
    arr = payload.get("messages")
    if isinstance(arr, list):
        for m in reversed(arr):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
    return ""

def _payload_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}

def _get_or_create_conversation_id(payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    conversation_id = str(payload.get("conversation_id") or "").strip()
    user_id = str(payload.get("user_id") or "").strip()

    if not conversation_id and user_id:
        conversation_id = get_user_conversation(user_id) or ""

    if not conversation_id:
        cookie_val = (request.cookies.get(CONV_COOKIE_NAME) or "").strip()
        if cookie_val:
            conversation_id = cookie_val

    if not conversation_id:
        conversation_id = _new_conversation_id()

    if user_id:
        set_user_conversation(user_id, conversation_id)

    return conversation_id, (user_id or None)

# =========================================================
# Booking guard / pricing (inchangé)
# =========================================================
# ... (TON CODE INCHANGÉ ICI)
# Pour rester 100% fidèle, je n’ai pas modifié ta logique booking/pricing/sanitize.

_FORBIDDEN_BOOKING_RE = [
    r"\bje (vous )?(confirme|confirme la|confirme votre)\b",
    r"\b(réservation|reservation) (est )?(confirmée|confirmee|validée|validee)\b",
    r"\bje (bloque|bloque donc|bloque votre|réserve|reserve)\b",
    r"\b(nous|on) vous attend(ons)?\b",
    r"\b(c['’]?est|c est) (disponible|ok|bon)\b",
    r"\bdisponible pour\b",
    r"\bcréneau confirmé\b",
    r"\bje vous le bloque\b",
    r"\bje vous le reserve\b",
    r"\bje vous le réserve\b",
    r"\bje vous note\b.*\b\d{1,2}h\b",
]

def common_greeting_text() -> str:
    return (
        "Bienvenue chez Runningman et Retroworld, comment puis-je vous aider ?\n\n"
        "Pour vous orienter rapidement :\n"
        "- Retroworld : VR, Escape VR, quiz, goûter à volonté + gâteau\n"
        "- Runningman : Game Zone, Escape game non-VR, Salle enfant"
    )

# -------------------------------------------------------------------
# IMPORTANT: Ta fonction process_chat est réutilisée telle qu’elle est
# (je ne la recopie pas ici pour ne pas casser le message),
# donc garde ton bloc "Core pipeline" INCHANGÉ sous ce point.
# -------------------------------------------------------------------

# =========================================================
# ✅ FAQ: CORRIGÉ (fallback si fichier static absent)
# =========================================================

def _extract_faq_from_kb(kb: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accepte kb["faq"] ou kb["questions_reponses"] ou kb["qna"] sous forme:
    [{question/q, answer/a/reponse, tags:[...]}]
    """
    if not isinstance(kb, dict):
        return []
    for key in ("faq", "questions_reponses", "qna"):
        block = kb.get(key)
        if isinstance(block, list):
            out = []
            for it in block:
                if not isinstance(it, dict):
                    continue
                q = (it.get("question") or it.get("q") or "").strip()
                a = (it.get("answer") or it.get("a") or it.get("reponse") or "").strip()
                tags = it.get("tags") if isinstance(it.get("tags"), list) else []
                tags = [str(t) for t in tags][:12]
                if q and a:
                    out.append({"question": q, "answer": a, "tags": tags})
            return out
    return []

def _faq_fallback_min_retroworld() -> List[Dict[str, Any]]:
    return [
        {"question": "Quels sont vos horaires ?", "answer": "Du mardi au dimanche, de 11h à 22h.", "tags": ["horaires"]},
        {"question": "Où êtes-vous situés ?", "answer": "815 avenue Pierre Brossolette, 83300 Draguignan.", "tags": ["adresse"]},
        {"question": "Tarifs VR / Escape VR ?", "answer": "Escape VR : 30€ (11h–20h) / 35€ (majoré). Jeux VR : 15€ / 20€. Quiz : 8€ (30min), 15€ (60min), 20€ (90min).", "tags": ["tarifs","vr","escape","quiz"]},
        {"question": "Comment réserver ?", "answer": "Le chat ne confirme pas de créneau. Pour réserver: Retroworld 04 94 47 94 64. Runningman 04 98 09 30 59.", "tags": ["réservation"]},
        {"question": "Runningman Action Game ?", "answer": "Pour l’Action Game, les clients contactent Runningman directement: 04 98 09 30 59.", "tags": ["runningman","action game"]},
    ]

def _faq_fallback_min_runningman() -> List[Dict[str, Any]]:
    return [
        {"question": "Comment réserver l’Action Game ?", "answer": "Contactez Runningman directement au 04 98 09 30 59 ou via https://www.runningmangames.fr.", "tags": ["runningman","réservation"]},
    ]

# =========================================================
# Routes
# =========================================================

@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": SERVICE_NAME, "status": "ok", "time_utc": _utc_iso()}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/faq/retroworld", methods=["GET"])
def faq_retroworld():
    # 1) static json
    path = os.path.join(app.static_folder, "faq_retroworld.json")
    data = _safe_read_json(path, None)

    if isinstance(data, dict) and isinstance(data.get("items"), list) and data["items"]:
        return jsonify(data), 200

    # 2) fallback depuis KB
    kb = load_kb("retroworld")
    items = _extract_faq_from_kb(kb)

    # 3) fallback minimal
    if not items:
        items = _faq_fallback_min_retroworld()

    return jsonify({"ok": True, "items": items}), 200

@app.route("/faq/runningman", methods=["GET"])
def faq_runningman():
    path = os.path.join(app.static_folder, "faq_runningman.json")
    data = _safe_read_json(path, None)

    if isinstance(data, dict) and isinstance(data.get("items"), list) and data["items"]:
        return jsonify(data), 200

    kb = load_kb("runningman")
    items = _extract_faq_from_kb(kb)
    if not items:
        items = _faq_fallback_min_runningman()

    return jsonify({"ok": True, "items": items}), 200

# -------------------------------------------------------------------
# IMPORTANT: COLLE ICI TON BLOC /chat et process_chat EXACTEMENT
# sans modifications (ton code actuel est bon).
# -------------------------------------------------------------------

# --- START: ton code chat (inchangé) ---
@app.route("/chat", methods=["POST"])
def chat_auto():
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    user_text = _payload_text(payload).strip()
    metadata = _payload_metadata(payload)
    conversation_id, user_id = _get_or_create_conversation_id(payload)

    if user_text == START_SIGNAL:
        greeting = common_greeting_text()
        append_conversation_turn(
            conversation_id=conversation_id,
            brand_effective="auto",
            user_text="",
            assistant_reply=greeting,
            extra={"metadata": metadata, "brand_effective": "auto", "openai_usage": {"mode": "start"}, "user_id": user_id},
        )
        resp = make_response(
            jsonify({"reply": greeting, "answer": greeting, "brand_effective": "auto", "brand_entry": "auto", "conversation_id": conversation_id}),
            200,
        )
        resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
        return resp

    if not user_text:
        return jsonify({"error": "missing_message"}), 400

    is_first_turn = len(load_conversation_messages(conversation_id)) == 0

    # ⚠️ process_chat DOIT être celui de ton fichier (inchangé)
    reply, usage, owner = process_chat("auto", user_text, conversation_id)  # type: ignore
    reply = strip_markdown_simple(reply)
    reply = maybe_prefix_common_greeting(reply, is_first_turn=is_first_turn)  # type: ignore

    append_conversation_turn(
        conversation_id=conversation_id,
        brand_effective=owner,
        user_text=user_text,
        assistant_reply=reply,
        extra={"metadata": metadata, "brand_effective": owner, "openai_usage": usage, "user_id": user_id},
    )

    resp = make_response(
        jsonify({"reply": reply, "answer": reply, "brand_effective": owner, "brand_entry": "auto", "conversation_id": conversation_id}),
        200,
    )
    resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
    return resp

@app.route("/chat/<brand>", methods=["POST"])
def chat_brand(brand: str):
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    user_text = _payload_text(payload).strip()
    metadata = _payload_metadata(payload)
    conversation_id, user_id = _get_or_create_conversation_id(payload)

    if user_text == START_SIGNAL:
        greeting = common_greeting_text()
        append_conversation_turn(
            conversation_id=conversation_id,
            brand_effective=brand,
            user_text="",
            assistant_reply=greeting,
            extra={"metadata": metadata, "brand_entry": brand, "brand_effective": brand, "openai_usage": {"mode": "start"}, "user_id": user_id},
        )
        resp = make_response(
            jsonify({"reply": greeting, "answer": greeting, "brand_effective": brand, "brand_entry": brand, "conversation_id": conversation_id}),
            200,
        )
        resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
        return resp

    if not user_text:
        return jsonify({"error": "missing_message"}), 400

    is_first_turn = len(load_conversation_messages(conversation_id)) == 0

    reply, usage, owner = process_chat(brand, user_text, conversation_id)  # type: ignore
    reply = strip_markdown_simple(reply)
    reply = maybe_prefix_common_greeting(reply, is_first_turn=is_first_turn)  # type: ignore

    append_conversation_turn(
        conversation_id=conversation_id,
        brand_effective=owner,
        user_text=user_text,
        assistant_reply=reply,
        extra={"metadata": metadata, "brand_entry": brand, "brand_effective": owner, "openai_usage": usage, "user_id": user_id},
    )

    resp = make_response(
        jsonify({"reply": reply, "answer": reply, "brand_effective": owner, "brand_entry": brand, "conversation_id": conversation_id}),
        200,
    )
    resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
    return resp
# --- END: ton code chat ---

@app.route("/user/api/history", methods=["GET"])
def user_api_history():
    if USER_HISTORY_TOKEN and not _require_user_token(request):
        return jsonify({"error": "forbidden"}), 403

    conversation_id = (request.args.get("conversation_id") or "").strip()
    user_id = (request.args.get("user_id") or "").strip()

    if not conversation_id and user_id:
        conversation_id = get_user_conversation(user_id) or ""

    if not conversation_id:
        conversation_id = (request.cookies.get(CONV_COOKIE_NAME) or "").strip()

    if not conversation_id:
        return jsonify({"error": "missing_conversation_id"}), 400

    obj = load_conversation_obj(conversation_id)
    return jsonify({"conversation_id": conversation_id, "conversation": obj}), 200

@app.route("/kb/<brand>", methods=["GET"])
def kb_get(brand: str):
    kb = load_kb(brand)
    return jsonify(kb), 200

@app.route("/kb/<brand>", methods=["POST"])
def kb_upsert(brand: str):
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    kb = load_kb(brand)
    new_items = payload.get("items")
    if isinstance(new_items, list):
        kb["items"] = new_items

    ok = save_kb(brand, kb)
    return jsonify({"ok": ok, "brand": brand}), 200

# =========================================================
# ADMIN API (inchangé)
# =========================================================
@app.route("/admin/api/conversations", methods=["GET"])
def admin_api_conversations():
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    conv_ids = list_conversations()
    items: List[Dict[str, Any]] = []

    for cid in conv_ids:
        obj = load_conversation_obj(cid)
        msgs = obj.get("messages") or []
        brand_eff = str(obj.get("brand_last") or "unknown")

        preview = ""
        timestamp = None

        if isinstance(msgs, list) and msgs:
            lastm = msgs[-1] if isinstance(msgs[-1], dict) else {}
            timestamp = lastm.get("ts")

            last_user = ""
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "user":
                    last_user = str(m.get("content") or "")
                    break

            preview = (last_user or str(lastm.get("content") or "")).strip()
            preview = re.sub(r"\s+", " ", preview)
            if len(preview) > 120:
                preview = preview[:117] + "..."

        items.append(
            {"id": cid, "brand": brand_eff, "preview": preview, "timestamp": timestamp, "user_id": obj.get("user_id")}
        )

    return jsonify(items), 200

@app.route("/admin/api/conversation/<conversation_id>", methods=["GET"])
def admin_api_conversation_detail(conversation_id: str):
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    obj = load_conversation_obj(conversation_id)
    msgs: List[Dict[str, Any]] = obj.get("messages") or []
    simple = [{"role": m.get("role"), "content": m.get("content"), "ts": m.get("ts")} for m in msgs if isinstance(m, dict)]
    return jsonify({"conversation_id": conversation_id, "messages": simple, "conversation": obj, "brand_final": obj.get("brand_last")}), 200

@app.route("/admin/api/diag", methods=["GET"])
def admin_api_diag():
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    def kb_diag(brand: str) -> Dict[str, Any]:
        fname = f"kb_{brand}.json"
        path = os.path.join(KB_DIR, fname)
        exists = os.path.exists(path)
        data = _safe_read_json(path, None) if exists else None
        load_ok = isinstance(data, dict)
        return {"exists": exists, "file": fname, "path": path, "load_ok": load_ok}

    try:
        conv_files = len([f for f in os.listdir(CONV_DIR) if f.endswith(".json")])
    except Exception:
        conv_files = 0

    return jsonify(
        {
            "service": SERVICE_NAME,
            "port": PORT,
            "has_admin_token": bool(ADMIN_DASHBOARD_TOKEN),
            "has_user_history_token": bool(USER_HISTORY_TOKEN),
            "paths": {"kb_dir": KB_DIR, "data_dir": DATA_DIR, "conv_dir": CONV_DIR},
            "kb": {"retroworld": kb_diag("retroworld"), "runningman": kb_diag("runningman")},
            "logs": {"conversations_files_count": conv_files},
            "openai": {
                "client_ready": bool(_openai_client),
                "key_present": bool(OPENAI_API_KEY),
                "model": OPENAI_MODEL,
                "reasoning_effort": OPENAI_REASONING_EFFORT,
                "init_error": _openai_init_error,
            },
        }
    ), 200

@app.route("/admin/api/export/conversations.zip", methods=["GET"])
def admin_export_conversations_zip():
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    buf = io.BytesIO()
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "Z"
    zip_name = f"conversations_{stamp}.zip"

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        try:
            for fn in os.listdir(CONV_DIR):
                if fn.endswith(".json"):
                    full = os.path.join(CONV_DIR, fn)
                    z.write(full, arcname=f"conversations/{fn}")
        except Exception as e:
            logger.warning("export conversations failed: %s", e)

        if os.path.exists(USER_INDEX_PATH):
            z.write(USER_INDEX_PATH, arcname="user_index.json")

    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=zip_name, max_age=0)

@app.route("/admin", methods=["GET"])
def admin_page():
    # ✅ protégé si token défini
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403
    return send_from_directory(app.static_folder, "admin.html")

@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403
    return send_from_directory(app.static_folder, "admin.html")

@app.route("/<path:path>", methods=["GET"])
def static_proxy(path: str):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    logger.info("Starting %s on port %s", SERVICE_NAME, PORT)
    app.run(host="0.0.0.0", port=PORT)

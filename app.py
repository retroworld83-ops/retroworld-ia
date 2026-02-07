from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory, make_response, send_file
from flask_cors import CORS

import bt_service

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
# BT (assistant interne) - CONFIG
# =========================================================

BT_API_TOKEN = (os.getenv("BT_API_TOKEN") or "").strip()
BT_PROFILE_PATH = (os.getenv("BT_PROFILE_PATH") or os.path.join(BASE_DIR, "config", "bt_profile.yaml")).strip()
BT_MAX_OUTPUT_TOKENS = int(os.getenv("BT_MAX_OUTPUT_TOKENS") or "400")
BT_MAX_REPLY_CHARS = int(os.getenv("BT_MAX_REPLY_CHARS") or "500")
BT_CONV_DIR = os.path.join(DATA_DIR, "bt_conversations")
_bt_store = bt_service.BTStore(BT_CONV_DIR)
_bt_profile_cache: Dict[str, Any] = {}
_bt_profile_cache_ts: float = 0.0

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
    # ✅ FIX: toujours passer la string au re.sub
    s = str(s or "")
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)       # **gras**
    s = re.sub(r"`([^`]*)`", r"\1", s)           # `code`
    s = re.sub(r"^\s*-\s+", "• ", s, flags=re.M) # - liste
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


def _is_number_only(text: str) -> Optional[int]:
    t = (text or "").strip()
    if re.fullmatch(r"\d{1,2}", t):
        try:
            n = int(t)
            if 1 <= n <= 50:
                return n
        except Exception:
            return None
    return None


def _is_time_only(text: str) -> bool:
    t = _norm(text)
    return bool(re.fullmatch(r"\d{1,2}\s*h\s*(\d{2})?", t) or re.fullmatch(r"\d{1,2}\s*:\s*\d{2}", t))


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
# Intents + detection
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
    if re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", t):  # yyyy-mm-dd
        return True
    if re.search(r"\b(\d{1,2})/(\d{1,2})(/(20\d{2}))?\b", t):  # dd/mm[/yyyy]
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
    """
    Réservation si:
    - mots explicites
    - ou jour/date détecté
    - ou heure+date
    - ou heure seule mais le fil parlait déjà de créneau/resa récemment
    """
    t = _norm(text)
    explicit = any(k in t for k in ["réserver", "reservation", "réservation", "dispo", "disponible", "créneau", "creneau", "bloquer", "confirmer", "rdv", "rendez-vous"])
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
            if any(k in ut_n for k in ["réserver", "reservation", "réservation", "créneau", "creneau", "dispo", "aujourd", "demain", "samedi", "dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi"]):
                return True

    return False


def _is_just_question_mark(text: str) -> bool:
    t = (text or "").strip()
    return t in ["?", "??", "???"]


def _is_link_request(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in [
        "donnez-moi le lien", "donne moi le lien", "donnez moi le lien",
        "lien de réservation", "lien reservation", "lien de reservation",
        "je veux le lien", "envoyez le lien", "envoie le lien"
    ])


def detect_owner_from_text(text: str) -> str:
    """
    RÉPARTITION OFFICIELLE :
    - Retroworld : VR, Escape VR, Quiz, Goûter à volonté + gâteau, Anniversaire, Fidélité/points
    - Runningman : Game Zone, Escape game NON-VR, Salle enfant
    """
    t = _norm(text)

    if "retroworld" in t:
        return "retroworld"
    if "runningman" in t or "running man" in t:
        return "runningman"

    # Runningman d'abord (évite la boucle)
    if "gamezone" in t or "game zone" in t:
        return "runningman"
    if "action game" in t or "actiongame" in t:
        return "runningman"
    if "salle enfant" in t or "salle enfants" in t or ("salle" in t and "enfant" in t):
        return "runningman"
    if ("escape" in t or "escape game" in t) and "vr" not in t:
        return "runningman"

    # Retroworld
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

    return "auto"


def infer_owner_from_history(history: List[Dict[str, Any]]) -> str:
    """
    Rend owner "collant" : si l'utilisateur a dit 'gamezone' puis enchaine '3',
    on reste sur Runningman.
    """
    if not history:
        return "auto"

    # derniers messages user
    recent_user: List[str] = []
    for m in reversed(history):
        if isinstance(m, dict) and m.get("role") == "user":
            recent_user.append(str(m.get("content") or ""))
        if len(recent_user) >= 8:
            break

    # scoring simple
    score_rm = 0
    score_rw = 0
    for ut in recent_user:
        t = _norm(ut)
        if any(k in t for k in ["gamezone", "game zone", "runningman", "action game", "salle enfant", "escape"]):
            # escape sans vr => runningman, escape vr => retroworld, donc on filtre un peu
            if "escape" in t and "vr" in t:
                score_rw += 3
            elif "escape" in t and "vr" not in t:
                score_rm += 3
            else:
                score_rm += 2

        if any(k in t for k in ["retroworld", "vr", "casque", "quiz", "anniversaire", "gouter", "goûter", "fidélité", "points", "escape vr"]):
            score_rw += 2

    if score_rm > score_rw and score_rm >= 2:
        return "runningman"
    if score_rw > score_rm and score_rw >= 2:
        return "retroworld"
    return "auto"


def detect_owner(brand_entry: str, user_text: str, history: List[Dict[str, Any]]) -> str:
    """
    Combine brand_entry + texte + historique.
    """
    be = (brand_entry or "auto").lower().strip()
    if be in ("retroworld", "runningman"):
        return be

    # 1) texte
    o = detect_owner_from_text(user_text)
    if o != "auto":
        return o

    # 2) message minimaliste (ex: "3" ou "17h") => owner du contexte
    if _is_number_only(user_text) is not None or _is_time_only(user_text):
        o2 = infer_owner_from_history(history)
        if o2 != "auto":
            return o2

    # 3) fallback historique
    o2 = infer_owner_from_history(history)
    return o2


# =========================================================
# Facts blocks (fiables)
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
        "input": messages,
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
        "store": False,
    }

    # ✅ Ne passe "reasoning" que si effort est valide (évite surprises)
    if OPENAI_REASONING_EFFORT in ("low", "medium", "high"):
        kwargs["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}
    else:
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


def _last_assistant_text(history: List[Dict[str, Any]]) -> str:
    for m in reversed(history or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return str(m.get("content") or "")
    return ""


# =========================================================
# Tokens
# =========================================================

def _require_admin_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-Admin-Token") or "").strip()
    # si pas de token configuré côté serveur => ouvert
    if not ADMIN_DASHBOARD_TOKEN:
        return True
    return bool(tok) and tok == ADMIN_DASHBOARD_TOKEN


def _require_user_token(req) -> bool:
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-User-Token") or "").strip()
    if not USER_HISTORY_TOKEN:
        return True
    return bool(tok) and tok == USER_HISTORY_TOKEN


def _require_bt_token(req) -> bool:
    """BT est un endpoint interne: verrouillé par défaut."""
    expected = (BT_API_TOKEN or ADMIN_DASHBOARD_TOKEN or "").strip()
    if not expected:
        return False
    tok = (req.args.get("token") or "").strip()
    if not tok:
        tok = (req.headers.get("X-BT-Token") or "").strip()
    if not tok:
        auth = (req.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            tok = auth[7:].strip()
    return bool(tok) and tok == expected


def _bt_profile() -> Dict[str, Any]:
    """Recharge léger du profil BT (utile si tu modifies le YAML sans redeploy)."""
    global _bt_profile_cache, _bt_profile_cache_ts
    now = time.time()
    if _bt_profile_cache and (now - _bt_profile_cache_ts) < 5.0:
        return _bt_profile_cache
    prof = bt_service.load_bt_profile(BT_PROFILE_PATH) or {}
    if not isinstance(prof, dict):
        prof = {}
    _bt_profile_cache = prof
    _bt_profile_cache_ts = now
    return _bt_profile_cache


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
# Date logic for "devis <= 3 jours"
# =========================================================

def _parse_explicit_date(text: str) -> Optional[datetime]:
    t = _norm(text)

    m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", t)  # YYYY-MM-DD
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d)
        except Exception:
            return None

    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", t)  # DD/MM/YYYY
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d)
        except Exception:
            return None

    m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", t)  # DD/MM
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        now = _now_local()
        y = now.year
        try:
            dt = datetime(y, mo, d)
        except Exception:
            return None
        if dt.date() < now.date():
            try:
                dt = datetime(y + 1, mo, d)
            except Exception:
                return None
        return dt

    return None


def _days_until_requested_date(user_text: str) -> Optional[int]:
    t = _norm(user_text)
    now = _now_local().date()

    if "aujourd" in t:
        return 0
    if "demain" in t:
        return 1
    if "après-demain" in t or "apres-demain" in t or "apres demain" in t:
        return 2

    m = re.search(r"\bdans\s+(\d{1,2})\s+jour", t)
    if m:
        try:
            return max(0, int(m.group(1)))
        except Exception:
            pass

    dt = _parse_explicit_date(t)
    if dt:
        return (dt.date() - now).days

    weekdays = {"lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, "vendredi": 4, "samedi": 5, "dimanche": 6}
    for name, idx in weekdays.items():
        if re.search(rf"\b{name}\b", t):
            delta = (idx - now.weekday()) % 7
            return delta

    return None


# =========================================================
# Pricing Retroworld (déterministe) + anti-boucle
# =========================================================

def _extract_players(text: str) -> Optional[int]:
    t = _norm(text)

    m = re.search(r"\b(\d{1,2})\s*(personne|personnes|joueur|joueurs)\b", t)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 50:
            return n

    m = re.search(r"\bnous\s+sommes\s+(\d{1,2})\b", t)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 50:
            return n

    # nombre seul (ex: "3") => players si contexte pricing retroworld
    n_only = _is_number_only(text)
    if n_only is not None:
        return n_only

    return None


def _extract_quiz_duration(text: str) -> Optional[int]:
    t = _norm(text)

    if "90" in t and ("min" in t or "mn" in t):
        return 90
    if "1h30" in t or "1 h 30" in t or "une heure trente" in t:
        return 90

    if ("60" in t and ("min" in t or "mn" in t)) or re.search(r"\b1h\b", t) or "1 h" in t or "une heure" in t:
        return 60

    if ("30" in t and ("min" in t or "mn" in t)) or "30min" in t or "30 mn" in t or "demi-heure" in t or "demi heure" in t:
        return 30

    return None


def _is_normal_slot_from_text(text: str) -> Optional[bool]:
    t = _norm(text)
    if "entre 11h" in t and "20h" in t:
        return True
    if "11h" in t and "20h" in t and "entre" in t:
        return True
    if "avant 11h" in t or "avant 11 h" in t:
        return False
    if "après 20h" in t or "apres 20h" in t or "après 20 h" in t or "apres 20 h" in t:
        return False
    return None


def _is_majorated_time(hh: int, mm: int) -> bool:
    if hh < 11:
        return True
    if hh > 20:
        return True
    if hh == 20 and mm > 0:
        return True
    return False


def _assistant_was_asking_retroworld_pricing(history: List[Dict[str, Any]]) -> bool:
    """
    ✅ clé anti-dialogue de sourd:
    On ne continue le pricing Retroworld sur un message "3" QUE si
    le bot venait réellement de demander les infos Retroworld.
    """
    last_a = _norm(_last_assistant_text(history))
    if not last_a:
        return False
    triggers = [
        "pour une estimation retroworld",
        "vous serez combien",
        "quelle durée",
        "donnez l’heure",
        "d’accord. vous serez combien",
        "si vous me donnez l’heure",
        "estimation retroworld"
    ]
    return any(k in last_a for k in triggers)


def _infer_retroworld_activity(user_text: str, history: List[Dict[str, Any]]) -> Optional[str]:
    t = _norm(user_text)

    if "quiz" in t:
        return "quiz"
    if "escape" in t and "vr" in t:
        return "escape_vr"
    if ("jeux vr" in t) or ("arcade vr" in t) or ("mini-jeux" in t) or ("mini jeux" in t):
        return "jeux_vr"
    if "vr" in t and "escape" not in t:
        return "jeux_vr"

    # fallback sur historique user récent
    for m in reversed(history or []):
        if isinstance(m, dict) and m.get("role") == "user":
            tt = _norm(str(m.get("content") or ""))
            if "quiz" in tt:
                return "quiz"
            if "escape" in tt and "vr" in tt:
                return "escape_vr"
            if "vr" in tt and "escape" not in tt:
                return "jeux_vr"

    return None


def _retroworld_price_per_player(activity: str, majorated: bool, quiz_duration: Optional[int]) -> int:
    if activity == "escape_vr":
        return 35 if majorated else 30
    if activity == "jeux_vr":
        return 20 if majorated else 15
    if activity == "quiz":
        d = quiz_duration or 60
        if d == 30:
            return 12 if majorated else 8
        if d == 90:
            return 25 if majorated else 20
        return 20 if majorated else 15
    return 0


def rule_based_retroworld_estimate_reply(owner: str, user_text: str, history: List[Dict[str, Any]]) -> Optional[str]:
    """
    ✅ Ne se déclenche PAS juste parce que l'utilisateur envoie "3".
    Se déclenche si:
      - owner == retroworld
      - et (intention prix maintenant) OU (assistant venait de demander les infos prix retroworld)
    """
    if owner != "retroworld":
        return None

    wants_now = _is_price_intent(user_text) or ("donne moi une estimation" in _norm(user_text))
    continue_ctx = _assistant_was_asking_retroworld_pricing(history) and (
        _is_number_only(user_text) is not None or _is_time_only(user_text) or _extract_time(user_text) is not None
    )

    if not (wants_now or continue_ctx):
        return None

    activity = _infer_retroworld_activity(user_text, history)
    players = _extract_players(user_text)
    tm = _extract_time(user_text)
    slot_hint = _is_normal_slot_from_text(user_text)
    quiz_duration = _extract_quiz_duration(user_text)

    if not activity:
        return (
            "Pour une estimation Retroworld, vous voulez plutôt :\n"
            "• Escape VR\n"
            "• Jeux VR / Arcade VR (30 min)\n"
            "• Quiz (30 / 60 / 90 min)\n"
            "Et vous serez combien de joueurs ?"
        )

    if activity == "quiz" and quiz_duration is None:
        return "Pour le quiz, vous voulez quelle durée : 30 min, 60 min ou 90 min ?"

    if not players:
        return "D’accord. Vous serez combien de joueurs ? (ex : 3 personnes)"

    majorated: Optional[bool] = None
    if slot_hint is not None:
        majorated = (not slot_hint)
    elif tm is not None:
        majorated = _is_majorated_time(tm[0], tm[1])

    if majorated is None:
        p_norm = _retroworld_price_per_player(activity, majorated=False, quiz_duration=quiz_duration)
        p_maj = _retroworld_price_per_player(activity, majorated=True, quiz_duration=quiz_duration)
        label_act = "Escape VR" if activity == "escape_vr" else ("Jeux VR (30 min)" if activity == "jeux_vr" else f"Quiz {quiz_duration} min")
        return (
            f"Estimation Retroworld pour {label_act} ({players} joueur(s)) :\n"
            f"• Normal (11h00–20h00) : {players} × {p_norm}€ = {players*p_norm}€\n"
            f"• Majoré (avant 11h ou après 20h) : {players} × {p_maj}€ = {players*p_maj}€\n"
            "Si vous me donnez l’heure approximative, je vous donne le total exact."
        )

    ppp = _retroworld_price_per_player(activity, majorated=majorated, quiz_duration=quiz_duration)
    total = players * ppp
    label = "majoré" if majorated else "normal"

    if tm is not None:
        hh, mm = tm
        hhmm = f"{hh:02d}h{mm:02d}"
        label_act = "Escape VR" if activity == "escape_vr" else ("Jeux VR (30 min)" if activity == "jeux_vr" else f"Quiz {quiz_duration} min")
        return f"Estimation Retroworld ({label}) à {hhmm} pour {label_act} : {players} × {ppp}€ = {total}€."

    label_act = "Escape VR" if activity == "escape_vr" else ("Jeux VR (30 min)" if activity == "jeux_vr" else f"Quiz {quiz_duration} min")
    return f"Estimation Retroworld ({label}) pour {label_act} : {players} × {ppp}€ = {total}€."


def rule_based_general_pricing_reply_if_needed(owner: str, user_text: str) -> Optional[str]:
    """
    Si le client demande juste 'c est combien' sans préciser,
    on répond différemment selon owner (retroworld vs runningman).
    """
    if not _is_price_intent(user_text):
        return None
    t = _norm(user_text)
    if len(t) > 80:
        return None

    if owner == "runningman":
        kb_rm = load_kb("runningman")
        ident_rm = kb_rm.get("identite", {}) or {}
        contact_rm = (ident_rm.get("contact", {}) or {})
        rm_tel = contact_rm.get("telephone", "04 98 09 30 59")
        rm_site = contact_rm.get("site_web", "https://runningmangames.fr")
        return (
            "Pour Runningman (Game Zone / escape non-VR / salle enfant), le tarif dépend de l’activité et du nombre de participants.\n"
            f"Pour un prix exact : {rm_tel} ou {rm_site}\n"
            "Dites-moi l’activité (ex : Game Zone) et combien vous serez."
        )

    # défaut Retroworld
    return (
        "Tarifs Retroworld (par joueur) :\n"
        "• Escape VR : 30€ (11h–20h) / 35€ (majoré)\n"
        "• Jeux VR / Arcade VR (30 min) : 15€ / 20€\n"
        "• Quiz 30 min : 8€ / 12€ | 60 min : 15€ / 20€ | 90 min : 20€ / 25€\n"
        "Dites-moi l’activité, le nombre de joueurs et l’heure, je calcule le total."
    )


# =========================================================
# Booking reply (message standard + règle "pas de devis <= 3j")
# =========================================================

def build_booking_link_reply(owner: str) -> str:
    if owner == "runningman":
        kb_rm = load_kb("runningman")
        ident_rm = kb_rm.get("identite", {}) or {}
        contact_rm = (ident_rm.get("contact", {}) or {})
        rm_tel = contact_rm.get("telephone", "04 98 09 30 59")
        rm_site = contact_rm.get("site_web", "https://runningmangames.fr")
        return (
            "Pour Runningman, la réservation se fait via contact direct.\n"
            f"Téléphone : {rm_tel}\n"
            f"Site : {rm_site}"
        )

    # Retroworld
    return (
        "Voici le lien pour réserver en ligne :\n"
        "https://www.retroworldfrance.com\n"
        "Sur le site, utilisez le bouton de réservation."
    )


def build_booking_reply(owner: str, user_text: str, history: List[Dict[str, Any]]) -> str:
    kb_rm = load_kb("runningman")
    ident_rm = kb_rm.get("identite", {}) or {}
    contact_rm = (ident_rm.get("contact", {}) or {})
    rm_tel = contact_rm.get("telephone", "04 98 09 30 59")
    rm_site = contact_rm.get("site_web", "https://runningmangames.fr")

    rw_tel = "04 94 47 94 64"
    rw_mail = "contact@retroworldfrance.com"
    rw_site = "https://www.retroworldfrance.com"

    days = _days_until_requested_date(user_text)
    too_soon_for_quote = (days is not None and days <= 3)

    if owner == "runningman":
        contact_lines = f"Runningman : {rm_tel}\nSite : {rm_site}"
    else:
        contact_lines = f"Retroworld : {rw_tel}\nEmail : {rw_mail}\nSite : {rw_site}"

    msg = (
        "Je ne peux pas confirmer ni bloquer un créneau via le chat.\n"
        "Vous pouvez toutefois avancer tout de suite :\n"
        f"- {contact_lines}\n"
    )

    if too_soon_for_quote:
        msg += (
            "\nLa date demandée est trop proche pour garantir l’envoi d’un devis avant cette date.\n"
            "Le plus simple est d’appeler pour valider rapidement.\n"
        )
        return msg.strip()

    msg += (
        "\nOption devis : laissez-moi ces infos et je prépare la demande pour l’équipe :\n"
        "- activité\n"
        "- nombre de participants (+ âges si enfants)\n"
        "- jour souhaité + fourchette horaire\n"
        "- nom + téléphone (ou email)\n"
        "\nOption réservation en ligne : si vous le souhaitez, dites « Donnez-moi le lien » et je vous l’enverrai.\n"
    )
    return msg.strip()


# =========================================================
# Safety: anti-confirmation + anti "disponible"
# =========================================================

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


def sanitize_booking_reply(reply: str, user_text: str, owner: str, history: List[Dict[str, Any]]) -> str:
    if not reply:
        return reply

    low = _norm(reply)

    # si OpenAI dérape => message standard
    if any(re.search(p, low) for p in _FORBIDDEN_BOOKING_RE):
        return build_booking_reply(owner, user_text, history)

    # garde-fou supplémentaire: on élimine "disponible" même sans regex
    if "disponible" in low:
        return build_booking_reply(owner, user_text, history)

    return reply


def maybe_prefix_common_greeting(reply: str, is_first_turn: bool) -> str:
    if not is_first_turn:
        return reply
    return common_greeting_text() + "\n\n" + (reply or "")


# =========================================================
# Core pipeline
# =========================================================

def process_chat(brand_entry: str, user_text: str, conversation_id: str) -> Tuple[str, Dict[str, Any], str]:
    user_text = user_text or ""
    hist = load_conversation_messages(conversation_id) if conversation_id else []

    owner = detect_owner(brand_entry, user_text, hist)

    # 0) lien demandé explicitement => on répond (message séparé)
    if _is_link_request(user_text):
        eff_owner = owner if owner != "auto" else "retroworld"
        return build_booking_link_reply(eff_owner), {"mode": "link_reply"}, eff_owner

    # 1) réservation => message standard direct
    if _is_reservation_intent(user_text, history=hist):
        eff_owner = owner if owner != "auto" else "retroworld"
        return build_booking_reply(eff_owner, user_text, hist), {"mode": "booking_guard"}, eff_owner

    # 2) prix générique court
    gen = rule_based_general_pricing_reply_if_needed(owner if owner != "auto" else "retroworld", user_text)
    if gen:
        return gen, {"mode": "rule_based_general_pricing"}, owner

    # 3) pricing déterministe Retroworld (anti-boucle)
    rb = rule_based_retroworld_estimate_reply(owner, user_text, hist)
    if rb:
        return rb, {"mode": "rule_based_retroworld_pricing"}, owner

    # 4) Runningman: si l'utilisateur dit "gamezone" et enchaine "3", on répond côté Runningman (pas IA Retroworld)
    if owner == "runningman":
        kb_rm = load_kb("runningman")
        ident_rm = kb_rm.get("identite", {}) or {}
        contact_rm = (ident_rm.get("contact", {}) or {})
        rm_tel = contact_rm.get("telephone", "04 98 09 30 59")
        rm_site = contact_rm.get("site_web", "https://runningmangames.fr")

        n = _is_number_only(user_text)
        if n is not None:
            return (
                f"Parfait, vous serez {n} participant(s) pour Runningman.\n"
                "Pour vous donner la bonne info, c’est plutôt : Game Zone, escape non-VR ou salle enfant ?\n"
                f"Sinon, pour valider rapidement : {rm_tel} ou {rm_site}"
            ), {"mode": "runningman_context_reply"}, owner

    # 5) IA (KB)
    kb_rw = load_kb("retroworld")
    kb_rm = load_kb("runningman")

    system = (
        "Vous êtes l’assistant COMMUN de Runningman et de Retroworld (même bâtiment à Draguignan).\n"
        "Votre mission : aider la clientèle sans créer de problème.\n\n"
        "RÈGLES ABSOLUES :\n"
        "- Ne jamais confirmer ni bloquer un créneau.\n"
        "- Ne jamais dire qu’un horaire est 'disponible'.\n"
        "- Réponses en texte simple.\n"
        "- Si info incertaine : 1 question courte, sinon proposer téléphone.\n\n"
        + common_orientation_block()
        + "\n"
        + retroworld_facts_block()
        + "\n"
        + runningman_facts_block(kb_rm)
    )

    kb_lines: List[str] = []
    kb_lines.extend(kb_snippets_retroworld(kb_rw))
    kb_lines.extend(kb_snippets_runningman(kb_rm))
    if kb_lines:
        system += "\nKB (résumé) :\n"
        for line in kb_lines[:120]:
            system += f"- {line}\n"

    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    if hist:
        messages.extend(prune_messages_for_prompt(hist, max_pairs=10))
    messages.append({"role": "user", "content": user_text})

    hints: List[str] = []
    if owner == "auto":
        hints.append("C'est ambigu: demandez si c'est Retroworld (VR/Escape VR/quiz/goûter) ou Runningman (Game Zone/Escape non-VR/Salle enfant).")
    if _is_just_question_mark(user_text):
        hints.append("L'utilisateur a envoyé seulement '?'. Demandez ce qu'il souhaite.")
    if owner == "runningman":
        hints.append("Le sujet est Runningman. Donnez infos Runningman et demandez activité + nombre + âge si utile.")
    if owner == "retroworld":
        hints.append("Le sujet est Retroworld. Donnez infos Retroworld.")
    if hints:
        messages.append({"role": "system", "content": "GUIDE:\n- " + "\n- ".join(hints)})

    reply, usage = call_openai_chat(messages)
    reply = strip_markdown_simple(reply)

    reply = sanitize_booking_reply(reply, user_text=user_text, owner=(owner if owner != "auto" else "retroworld"), history=hist)
    return reply, usage, owner


# =========================================================
# Routes
# =========================================================

@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": SERVICE_NAME, "status": "ok", "time_utc": _utc_iso()}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/bt/health", methods=["GET"])
def bt_health():
    if not _require_bt_token(request):
        return jsonify({"error": "forbidden"}), 403
    return jsonify({"status": "ok", "bt": "ok"}), 200


@app.route("/bt/profile", methods=["GET"])
def bt_profile():
    if not _require_bt_token(request):
        return jsonify({"error": "forbidden"}), 403
    prof = _bt_profile()
    # Pas de secrets attendus ici: c'est un fichier de comportement.
    return jsonify({"profile": prof, "profile_path": BT_PROFILE_PATH}), 200


@app.route("/bt/chat", methods=["POST"])
def bt_chat():
    if not _require_bt_token(request):
        return jsonify({"error": "forbidden"}), 403

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    # garde-fou: si un appel est marqué comme "public", on ignore.
    source = str(payload.get("source") or payload.get("audience") or "team").strip().lower()
    if source in ("public", "visitor", "client"):
        return ("", 204)

    user_text = str(
        payload.get("text")
        or payload.get("message")
        or payload.get("input")
        or _payload_text(payload)
        or ""
    ).strip()
    if not user_text:
        return jsonify({"error": "missing_message"}), 400

    conversation_id = str(payload.get("conversation_id") or "").strip()
    if not conversation_id:
        conversation_id = "bt_" + datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

    # nettoyage opportuniste
    _bt_store.prune_old()

    prof = _bt_profile()
    system_prompt = bt_service.bt_system_prompt(prof)
    history = _bt_store.prompt_messages(conversation_id, max_pairs=8)

    reply, usage = bt_service.call_openai_bt(
        openai_client=_openai_client,
        model=OPENAI_MODEL,
        reasoning_effort=OPENAI_REASONING_EFFORT,
        temperature=OPENAI_TEMPERATURE,
        max_output_tokens=BT_MAX_OUTPUT_TOKENS,
        system_prompt=system_prompt,
        history=history,
        user_text=user_text,
    )

    reply = bt_service.sanitize_bt_reply(reply, max_chars=BT_MAX_REPLY_CHARS)

    _bt_store.append(conversation_id, "user", user_text)
    _bt_store.append(conversation_id, "assistant", reply, extra={"openai_usage": usage})

    return jsonify({"reply": reply, "answer": reply, "conversation_id": conversation_id, "usage": usage}), 200


@app.route("/faq/retroworld", methods=["GET"])
def faq_retroworld():
    path = os.path.join(app.static_folder, "faq_retroworld.json")
    data = _safe_read_json(path, {"items": []})
    if not isinstance(data, dict):
        data = {"items": []}
    return jsonify(data), 200


@app.route("/faq/runningman", methods=["GET"])
def faq_runningman():
    path = os.path.join(app.static_folder, "faq_runningman.json")
    data = _safe_read_json(path, {"items": []})
    if not isinstance(data, dict):
        data = {"items": []}
    return jsonify(data), 200


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

    reply, usage, owner = process_chat("auto", user_text, conversation_id)
    reply = strip_markdown_simple(reply)
    reply = maybe_prefix_common_greeting(reply, is_first_turn=is_first_turn)

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

    reply, usage, owner = process_chat(brand, user_text, conversation_id)
    reply = strip_markdown_simple(reply)
    reply = maybe_prefix_common_greeting(reply, is_first_turn=is_first_turn)

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


@app.route("/user/api/history", methods=["GET"])
def user_api_history():
    if not _require_user_token(request):
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
    if not _require_admin_token(request):
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
# Admin API (RESTORED) => admin.html refonctionne
# =========================================================

@app.route("/admin/api/conversations", methods=["GET"])
def admin_api_conversations():
    if not _require_admin_token(request):
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

        items.append({"id": cid, "brand": brand_eff, "preview": preview, "timestamp": timestamp, "user_id": obj.get("user_id")})

    return jsonify(items), 200


@app.route("/admin/api/conversation/<conversation_id>", methods=["GET"])
def admin_api_conversation_detail(conversation_id: str):
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    obj = load_conversation_obj(conversation_id)
    msgs: List[Dict[str, Any]] = obj.get("messages") or []
    simple = [{"role": m.get("role"), "content": m.get("content"), "ts": m.get("ts")} for m in msgs if isinstance(m, dict)]
    return jsonify({"conversation_id": conversation_id, "messages": simple, "conversation": obj, "brand_final": obj.get("brand_last")}), 200


@app.route("/admin/api/diag", methods=["GET"])
def admin_api_diag():
    if not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    def kb_diag(brand: str) -> Dict[str, Any]:
        fname = f"kb_{brand}.json"
        path = os.path.join(KB_DIR, fname)
        exists = os.path.exists(path)
        data = _safe_read_json(path, None) if exists else None
        load_ok = isinstance(data, dict)
        # items_count utile pour admin.html
        items_count = None
        if isinstance(data, dict):
            items = data.get("items")
            items_count = len(items) if isinstance(items, list) else 0
        return {"exists": exists, "file": fname, "path": path, "load_ok": load_ok, "items_count": items_count}

    try:
        conv_files = len([f for f in os.listdir(CONV_DIR) if f.endswith(".json")])
    except Exception:
        conv_files = 0

    return jsonify(
        {
            "service": SERVICE_NAME,
            "port": PORT,
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
    if not _require_admin_token(request):
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
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=zip_name,
        max_age=0,
    )


@app.route("/admin", methods=["GET"])
def admin_page():
    return send_from_directory(app.static_folder, "admin.html")


@app.route("/admin/conversations", methods=["GET"])
def admin_conversations_page():
    return send_from_directory(app.static_folder, "admin.html")


@app.route("/<path:path>", methods=["GET"])
def static_proxy(path: str):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    logger.info("Starting %s on port %s", SERVICE_NAME, PORT)
    app.run(host="0.0.0.0", port=PORT)

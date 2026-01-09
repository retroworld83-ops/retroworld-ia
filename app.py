from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory, make_response
from flask_cors import CORS

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

SERVICE_NAME = os.getenv("SERVICE_NAME", "retroworld-ia")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

KB_DIR = os.getenv("KB_DIR", BASE_DIR)

PORT = int(os.getenv("PORT", "10000"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "800"))

ADMIN_DASHBOARD_TOKEN = os.getenv("ADMIN_DASHBOARD_TOKEN", "").strip()
USER_HISTORY_TOKEN = os.getenv("USER_HISTORY_TOKEN", "").strip()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(SERVICE_NAME)

app = Flask(__name__, static_folder="static")
CORS(app)

# Cookie used to keep the same conversation for browser/widget clients
CONV_COOKIE_NAME = os.getenv("CONV_COOKIE_NAME", "rw_conv_id")

# ---------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------


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


def _new_conversation_id() -> str:
    return "conv_" + datetime.utcnow().strftime("%Y%m%d%H%M%S%f")


# ---------------------------------------------------------
# user index (user_id -> conversation_id)
# ---------------------------------------------------------

USER_INDEX_PATH = os.path.join(DATA_DIR, "user_index.json")


def get_user_index() -> Dict[str, str]:
    return _safe_read_json(USER_INDEX_PATH, {}) or {}


def set_user_conversation(user_id: str, conversation_id: str) -> None:
    idx = get_user_index()
    idx[str(user_id)] = str(conversation_id)
    _safe_write_json(USER_INDEX_PATH, idx)


def get_user_conversation(user_id: str) -> Optional[str]:
    idx = get_user_index()
    return idx.get(str(user_id))


# ---------------------------------------------------------
# KB cache
# ---------------------------------------------------------


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
    kb = _safe_read_json(path, {"brand": brand, "items": []})
    if not isinstance(kb, dict):
        kb = {"brand": brand, "items": []}

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


# ---------------------------------------------------------
# NLP + OpenAI wrapper
# ---------------------------------------------------------


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def detect_brand_from_text(text: str) -> str:
    t = _norm(text)
    if "runningman" in t or "running man" in t or "action game" in t:
        return "runningman"
    return "retroworld"


def _is_reservation_intent(text: str) -> bool:
    t = _norm(text)
    keys = [
        "réserver",
        "reservation",
        "réservation",
        "dispo",
        "disponible",
        "créneau",
        "horaire",
        "aujourd",
        "demain",
        "ce week",
        "samedi",
        "dimanche",
    ]
    return any(k in t for k in keys)


def _is_price_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["c'est combien", "cest combien", "tarif", "prix", "combien ça coute", "ça coute combien", "ca coute combien"])


def _is_hours_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["horaire", "horaires", "ouvert", "ouverts", "ferme", "fermé", "fermez", "ouvre", "jusqu'à quelle", "jusqu’a quelle"])


def _is_location_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["adresse", "où", "ou", "localisation", "venir", "comment venir", "draguignan", "parking"])


def _retroworld_static_info(kind: str) -> str:
    if kind == "hours":
        return "Retroworld France est **ouvert du mardi au dimanche, de 11h à 22h**."
    if kind == "location":
        return "Retroworld France se trouve au **815 avenue Pierre Brossolette, 83300 Draguignan**."
    if kind == "contact":
        return "Contact Retroworld : **04 94 47 94 64** | **contact@retroworldfrance.com**."
    # fallback
    return "Retroworld France : **815 avenue Pierre Brossolette, 83300 Draguignan**, du mardi au dimanche **11h-22h**."


def _retroworld_price_table() -> str:
    return (
        "Tarifs Retroworld :\n"
        "- **VR** : 15 € / joueur (jusqu’à 5 joueurs)\n"
        "- **Escape Game VR** : 30 € / joueur\n"
        "- **Quiz interactifs** : 8 € (30 min) | 15 € (60 min) | 20 € (90 min) (jusqu’à 12 joueurs)\n"
        "- **Salle enfant** : 50 € / heure, puis 20 € par demi-heure supplémentaire"
    )


def _retroworld_price_answer(text: str) -> Optional[str]:
    """
    Réponse déterministe pour éviter les hallucinations tarifaires.
    Basé sur vos tarifs officiels.
    """
    t = _norm(text)

    # Escape VR (doit passer AVANT le test "vr", sinon "escape vr" renvoie le tarif VR)
    if "escape" in t:
        return (
            "Pour Retroworld, un **Escape Game VR** est à **30 € / joueur**. "
            "Combien de joueurs serez-vous et plutôt quel type d’univers (aventure, enquête, horreur, enfants) ?"
        )

    # VR
    if "vr" in t or "virtual" in t:
        return (
            "Pour Retroworld, une partie de VR est à **15 € / joueur** (jusqu’à 5 joueurs). "
            "Souhaitez-vous une simple session VR ou un **Escape Game VR** ?"
        )

    # Quiz
    if "quiz" in t:
        return (
            "Pour Retroworld, nos quiz interactifs : **8 € (30 min)**, **15 € (60 min)**, **20 € (90 min)**, jusqu’à 12 joueurs. "
            "Vous préférez 30, 60 ou 90 minutes ?"
        )

    # Salle enfant
    if "salle" in t and ("enfant" in t or "anniv" in t or "anniversaire" in t):
        return (
            "Pour Retroworld, la **salle enfant** est à **50 € / heure**, puis **20 € par demi-heure supplémentaire**. "
            "Pour quelle date et combien d’enfants environ ?"
        )

    return None


def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    if not _openai_client:
        return (
            "Le service IA n’est pas configuré (OPENAI_API_KEY manquant).",
            {"error": "openai_not_configured"},
        )

    resp = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
    )
    text = (resp.choices[0].message.content or "").strip()

    usage: Dict[str, Any] = {}
    try:
        usage = resp.usage.model_dump()  # type: ignore
    except Exception:
        try:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        except Exception:
            usage = {}

    return text, usage


# ---------------------------------------------------------
# Conversation storage
# ---------------------------------------------------------

CONV_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONV_DIR, exist_ok=True)


def _conversation_path(conversation_id: str) -> str:
    conversation_id = re.sub(r"[^a-zA-Z0-9_\-]", "", conversation_id or "")
    if not conversation_id:
        conversation_id = _new_conversation_id()
    return os.path.join(CONV_DIR, f"{conversation_id}.json")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _coerce_legacy_records_to_messages(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convertit l'ancien format (liste de records) en timeline de messages."""
    msgs: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        ts = str(rec.get("ts") or _utc_iso())
        user_msgs = rec.get("user_messages") or []
        if isinstance(user_msgs, list):
            for um in user_msgs:
                if isinstance(um, dict) and um.get("role") in ("user", "assistant"):
                    msgs.append(
                        {
                            "role": um.get("role"),
                            "content": str(um.get("content") or ""),
                            "ts": ts,
                        }
                    )
        ar = rec.get("assistant_reply")
        if ar is not None:
            msgs.append({"role": "assistant", "content": str(ar), "ts": ts})
    return msgs


def load_conversation_obj(conversation_id: str) -> Dict[str, Any]:
    """
    Nouveau format (v2):
      { "version": 2, "id": ..., "created": ..., "user_id": ..., "brand_last": ..., "messages": [...] }

    Compat:
      si fichier = liste (ancien format), on renvoie un objet v2 reconstruit.
    """
    path = _conversation_path(conversation_id)
    data = _safe_read_json(path, {})

    # v2
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        data.setdefault("version", 2)
        data.setdefault("id", conversation_id)
        data.setdefault("created", data.get("created") or _utc_iso())
        return data

    # legacy list -> v2
    if isinstance(data, list):
        msgs = _coerce_legacy_records_to_messages(data)
        return {
            "version": 2,
            "id": conversation_id,
            "created": (data[0].get("ts") if data and isinstance(data[0], dict) else _utc_iso()),
            "user_id": None,
            "brand_last": (data[-1].get("brand") if data and isinstance(data[-1], dict) else None),
            "messages": msgs,
            "legacy_records": data,
        }

    # empty
    return {"version": 2, "id": conversation_id, "created": _utc_iso(), "user_id": None, "brand_last": None, "messages": []}


def save_conversation_obj(conversation_id: str, obj: Dict[str, Any]) -> None:
    path = _conversation_path(conversation_id)
    _safe_write_json(path, obj)


def load_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    return load_conversation_obj(conversation_id).get("messages") or []


def append_conversation_turn(
    conversation_id: str,
    brand_effective: str,
    user_messages: List[Dict[str, Any]],
    assistant_reply: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Ajoute un tour de conversation au format v2 (timeline)."""
    obj = load_conversation_obj(conversation_id)
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        msgs = []

    ts = _utc_iso()

    # On n'enregistre QUE les messages utilisateur de ce tour.
    # (Le widget envoyait parfois l'historique complet, ce qui créait des doublons en admin.)
    for um in (user_messages or []):
        if not isinstance(um, dict):
            continue
        if um.get("role") != "user":
            continue
        content = str(um.get("content") or "").strip()
        if not content:
            continue
        msgs.append({"role": "user", "content": content, "ts": ts})

    msgs.append({"role": "assistant", "content": str(assistant_reply or ""), "ts": ts})

    obj["version"] = 2
    obj["id"] = conversation_id
    obj.setdefault("created", obj.get("created") or ts)
    obj["brand_last"] = str(brand_effective or "")
    if extra and isinstance(extra, dict):
        obj["extra_last"] = extra
        # conserver un user_id si présent
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


def prune_messages_for_prompt(messages: List[Dict[str, Any]], max_pairs: int = 14) -> List[Dict[str, str]]:
    """Garde les derniers échanges (user+assistant) pour le prompt, sans exploser en tokens."""
    if not isinstance(messages, list):
        return []

    # On prend un peu large, puis on coupe par paires.
    clipped = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content") is not None]

    # Garder les derniers N paires (2 messages par paire)
    max_msgs = max_pairs * 2
    clipped = clipped[-max_msgs:]

    out: List[Dict[str, str]] = []
    for m in clipped:
        out.append({"role": str(m.get("role")), "content": str(m.get("content") or "")})
    return out


# ---------------------------------------------------------
# Token helpers
# ---------------------------------------------------------


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


# ---------------------------------------------------------
# Payload coercion
# ---------------------------------------------------------


def _coerce_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = payload.get("messages")
    if isinstance(raw, list):
        out: List[Dict[str, str]] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").strip()
            content = m.get("content")
            if role in ("user", "assistant") and content is not None:
                out.append({"role": role, "content": str(content)})
        return out

    out: List[Dict[str, str]] = []
    history = payload.get("history") or []
    if isinstance(history, list):
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                u, a = item
                if u:
                    out.append({"role": "user", "content": str(u)})
                if a:
                    out.append({"role": "assistant", "content": str(a)})
                continue

            if isinstance(item, dict):
                role = (item.get("role") or "").strip()
                content = item.get("content")
                if role in ("user", "assistant") and content is not None:
                    out.append({"role": role, "content": str(content)})

    msg = payload.get("message")
    if msg:
        out.append({"role": "user", "content": str(msg)})

    return out


def _coerce_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _get_or_create_conversation_id(payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Returns (conversation_id, user_id).
    Sources:
      1) payload.conversation_id
      2) payload.user_id -> mapped conversation
      3) cookie rw_conv_id
      4) new generated
    """
    conversation_id = str(payload.get("conversation_id") or "").strip()
    user_id = str(payload.get("user_id") or "").strip()

    if user_id and not conversation_id:
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


# ---------------------------------------------------------
# Core chat pipeline
# ---------------------------------------------------------


def process_chat(brand_entry: str, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata = metadata or {}
    brand_entry = (brand_entry or "auto").lower().strip()

    # Determine effective brand
    if brand_entry == "auto":
        last_user = ""
        for m in reversed(messages or []):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = str(m.get("content") or "")
                break
        brand_effective = detect_brand_from_text(last_user)
    else:
        # FORCE brand, no ambiguity
        brand_effective = "retroworld" if brand_entry != "runningman" else "runningman"

    kb = load_kb(brand_effective)
    kb_items = kb.get("items") or []
    if not isinstance(kb_items, list):
        kb_items = []

    # Build a BRAND-SPECIFIC system prompt (this kills the "quelle marque ?" loop)
    if brand_effective == "retroworld":
        system = (
            "Vous êtes l’assistant officiel de Retroworld (VR, Escape Game VR, quiz interactifs, salle enfant).\n"
            "Ne demandez pas 'quelle marque' : vous répondez côté Retroworld.\n"
            "Règle importante: pour les informations de base (tarifs, horaires, adresse, contact), vous répondez directement et vous n'envoyez pas l'utilisateur 'voir le site' (sauf s'il demande explicitement un lien).\n"
            "Tarifs Retroworld:\n"
            "- VR : 15 €/joueur, jusqu’à 5 joueurs.\n"
            "- Escape Game VR : 30 €/joueur.\n"
            "- Quiz : 8€ (30min), 15€ (60min), 20€ (90min), jusqu’à 12 joueurs.\n"
            "- Salle enfant : 50 €/h, +20 €/demi-heure supplémentaire.\n"
            "Horaires: du mardi au dimanche, 11h à 22h.\n"
            "Adresse: 815 avenue Pierre Brossolette, 83300 Draguignan.\n"
            "Contact: 04 94 47 94 64 / contact@retroworldfrance.com.\n"
            "Si l’utilisateur veut réserver: répondre 'disponible' puis demander date/heure + nombre de personnes.\n"
            "Répondez clairement, sans inventer.\n"
        )
    else:
        system = (
            "Vous êtes l’assistant côté Runningman Games (action game).\n"
            "Ne demandez pas 'quelle marque' : vous répondez côté Runningman.\n"
            "Si l’utilisateur veut réserver: indiquer que les clients doivent contacter Runningman directement.\n"
            "Répondez clairement, sans inventer.\n"
        )

    # Add KB items (optional)
    if kb_items:
        system += "\nConnaissances (KB) :\n"
        for it in kb_items[:140]:
            if isinstance(it, str):
                system += f"- {it}\n"
            elif isinstance(it, dict):
                title = it.get("title") or it.get("name") or ""
                content = it.get("content") or it.get("text") or it.get("value") or ""
                if (title or content):
                    system += f"- {title}: {content}\n" if title else f"- {content}\n"

    openai_messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

    conversation_id = str(metadata.get("conversation_id") or "").strip()
    if conversation_id:
        hist_msgs = load_conversation_messages(conversation_id)
        openai_messages.extend(prune_messages_for_prompt(hist_msgs, max_pairs=12))

    clipped: List[Dict[str, str]] = []
    for m in (messages or [])[-20:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content is not None:
            clipped.append({"role": role, "content": str(content)})
    openai_messages.extend(clipped)

    last_user_text = ""
    for m in reversed(clipped):
        if m["role"] == "user":
            last_user_text = m["content"]
            break

    # Deterministic pricing for Retroworld (prevents "durée 30min/60min" hallucination)
    if brand_effective == "retroworld" and _is_price_intent(last_user_text):
        price_reply = _retroworld_price_answer(last_user_text)
        reply = price_reply or _retroworld_price_table()
        if reply:
            return {
                "reply": reply,
                "brand_entry": brand_entry,
                "brand_effective": brand_effective,
                "openai_usage": {"bypass": "price_rule"},
                "guard_hits": ["price_intent"],
                "metadata": metadata,
            }

    # Deterministic hours/location/contact for Retroworld
    if brand_effective == "retroworld" and last_user_text:
        if _is_hours_intent(last_user_text):
            return {
                "reply": _retroworld_static_info("hours"),
                "brand_entry": brand_entry,
                "brand_effective": brand_effective,
                "openai_usage": {"bypass": "hours_rule"},
                "guard_hits": ["hours_intent"],
                "metadata": metadata,
            }
        if _is_location_intent(last_user_text):
            return {
                "reply": _retroworld_static_info("location"),
                "brand_entry": brand_entry,
                "brand_effective": brand_effective,
                "openai_usage": {"bypass": "location_rule"},
                "guard_hits": ["location_intent"],
                "metadata": metadata,
            }

    guard_hits: List[str] = []
    if _is_reservation_intent(last_user_text):
        guard_hits.append("reservation_intent")

    reply_text, usage = call_openai_chat(openai_messages)

    return {
        "reply": reply_text,
        "brand_entry": brand_entry,
        "brand_effective": brand_effective,
        "openai_usage": usage,
        "guard_hits": guard_hits,
        "metadata": metadata,
    }


# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------


@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": SERVICE_NAME, "status": "ok", "time_utc": datetime.utcnow().isoformat() + "Z"}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/chat", methods=["POST"])
def chat_compat():
    """
    Compat endpoint for older widgets.
    We keep conversation with a cookie + we log conversation history.
    """
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    messages = _coerce_messages(payload)
    if not messages:
        return jsonify({"error": "invalid_messages"}), 400

    metadata = _coerce_metadata(payload)
    conversation_id, user_id = _get_or_create_conversation_id(payload)
    metadata = dict(metadata)
    metadata.setdefault("conversation_id", conversation_id)

    out = process_chat(brand_entry="auto", messages=messages, metadata=metadata)
    reply_text = out.get("reply", "")

    append_conversation_turn(
        conversation_id=conversation_id,
        brand_effective=str(out.get("brand_effective") or "auto"),
        user_messages=messages,
        assistant_reply=reply_text,
        extra={
            "metadata": metadata,
            "brand_entry": out.get("brand_entry"),
            "brand_effective": out.get("brand_effective"),
            "openai_usage": out.get("openai_usage"),
            "guard_hits": out.get("guard_hits"),
            "user_id": user_id,
        },
    )

    resp = make_response(
        jsonify(
            {
                "reply": reply_text,
                "answer": reply_text,
                "brand_effective": out.get("brand_effective"),
                "brand_entry": out.get("brand_entry"),
                "conversation_id": conversation_id,
            }
        ),
        200,
    )
    resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
    return resp


@app.route("/chat/<brand>", methods=["POST"])
def chat_route(brand: str):
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    messages = _coerce_messages(payload)
    if not messages:
        return jsonify({"error": "invalid_messages"}), 400

    metadata = _coerce_metadata(payload)
    conversation_id, user_id = _get_or_create_conversation_id(payload)
    metadata = dict(metadata)
    metadata.setdefault("conversation_id", conversation_id)

    out = process_chat(brand_entry=brand, messages=messages, metadata=metadata)
    reply_text = out.get("reply", "")

    append_conversation_turn(
        conversation_id=conversation_id,
        brand_effective=str(out.get("brand_effective") or brand),
        user_messages=messages,
        assistant_reply=reply_text,
        extra={
            "metadata": metadata,
            "brand_entry": out.get("brand_entry"),
            "brand_effective": out.get("brand_effective"),
            "openai_usage": out.get("openai_usage"),
            "guard_hits": out.get("guard_hits"),
            "user_id": user_id,
        },
    )

    resp = make_response(
        jsonify(
            {
                "reply": reply_text,
                "answer": reply_text,
                "brand_effective": out.get("brand_effective"),
                "conversation_id": conversation_id,
            }
        ),
        200,
    )
    resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
    return resp


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
    items = kb.get("items") or []
    if not isinstance(items, list):
        items = []

    new_items = payload.get("items")
    if isinstance(new_items, list):
        items = new_items

    kb["items"] = items
    ok = save_kb(brand, kb)
    return jsonify({"ok": ok, "brand": brand, "count": len(items)}), 200


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
            # preview = dernier message user si possible
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
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    obj = load_conversation_obj(conversation_id)
    msgs: List[Dict[str, Any]] = obj.get("messages") or []
    # Pour l'UI: fournir aussi une version "messages" simplifiée
    simple = [{"role": m.get("role"), "content": m.get("content"), "ts": m.get("ts")} for m in msgs if isinstance(m, dict)]
    return jsonify({"conversation_id": conversation_id, "messages": simple, "conversation": obj, "brand_final": obj.get("brand_last")}), 200


@app.route("/admin/api/test", methods=["GET", "POST"])
def admin_api_test():
    if ADMIN_DASHBOARD_TOKEN and not _require_admin_token(request):
        return jsonify({"error": "forbidden"}), 403

    if request.method == "GET":
        return jsonify({"ok": True, "service": SERVICE_NAME}), 200

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    brand = str(payload.get("brand") or "retroworld").lower().strip()
    questions = payload.get("questions") or []
    if not isinstance(questions, list) or not questions:
        return jsonify({"error": "invalid_questions"}), 400

    results: List[Dict[str, Any]] = []
    for q in questions[:50]:
        q_txt = str(q or "").strip()
        if not q_txt:
            continue
        t0 = time.time()
        out = process_chat(brand_entry=brand, messages=[{"role": "user", "content": q_txt}], metadata={})
        ms = int((time.time() - t0) * 1000)
        results.append({"question": q_txt, "reply": out.get("reply", ""), "brand_effective": out.get("brand_effective"), "ms": ms})

    return jsonify({"ok": True, "brand": brand, "count": len(results), "results": results}), 200


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
        items_count = 0
        if isinstance(data, dict):
            items = data.get("items")
            if isinstance(items, list):
                items_count = len(items)
        return {"exists": exists, "file": fname, "path": path, "load_ok": load_ok, "items_count": items_count}

    conv_files = 0
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
            "openai": {"client_ready": bool(_openai_client), "key_present": bool(OPENAI_API_KEY), "model": OPENAI_MODEL, "init_error": _openai_init_error},
        }
    ), 200


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

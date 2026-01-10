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
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.4"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "900"))

ADMIN_DASHBOARD_TOKEN = (os.getenv("ADMIN_DASHBOARD_TOKEN") or "").strip()
USER_HISTORY_TOKEN = (os.getenv("USER_HISTORY_TOKEN") or "").strip()

LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(SERVICE_NAME)

app = Flask(__name__, static_folder="static")
CORS(app)

# Cookie fallback (peut être bloqué en iframe => on ne dépend PAS de ça)
CONV_COOKIE_NAME = os.getenv("CONV_COOKIE_NAME", "rw_conv_id")

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
# Helpers JSON / IDs / time
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


# =========================================================
# Text utils + intents (avec FIX "gouter" => pas adresse)
# =========================================================

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
        "réserver", "reservation", "réservation", "dispo", "disponible",
        "créneau", "creneau", "aujourd", "demain", "samedi", "dimanche"
    ]
    return any(k in t for k in keys)


def _is_price_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["c'est combien", "cest combien", "tarif", "prix", "ça coute", "ca coute"])


def _is_hours_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["horaire", "horaires", "ouvert", "ouvre", "ferme", "fermé"])


def _is_location_intent(text: str) -> bool:
    """
    IMPORTANT:
    - NE PAS tester "ou" (sans accent), sinon "gouter" contient "ou" => bug adresse.
    - On accepte "où" uniquement en mot entier + clés explicites.
    """
    t = _norm(text)
    if re.search(r"\boù\b", t):
        return True
    keys = [
        "adresse", "localisation", "venir", "comment venir", "draguignan",
        "parking", "vous êtes où", "vous etes où", "c'est où", "c est où",
    ]
    return any(k in t for k in keys)


def _is_gouter_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["goûter", "gouter", "anniversaire", "gateau", "gâteau", "formule gouter", "formule goûter"])


def _is_fidelity_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["fidélité", "fidelite", "points", "qr", "qr code", "récompense", "recompense"])


def _is_age_intent(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in ["âge", "age", "ans", "enfant", "enfants", "-12", "moins de 12", "12 ans", "7 ans"])


def _is_just_question_mark(text: str) -> bool:
    t = (text or "").strip()
    return t in ["?", "??", "???"]


# =========================================================
# Retroworld facts (source fiable) + Runningman facts
# =========================================================

def retroworld_facts_block() -> str:
    return (
        "VÉRITÉS OFFICIELLES RETROWORLD (à respecter strictement) :\n"
        "- Tarifs:\n"
        "  - VR : 15 € / joueur (jusqu’à 5 joueurs)\n"
        "  - Escape Game VR : 30 € / joueur\n"
        "  - Quiz interactifs : 8 € (30 min), 15 € (60 min), 20 € (90 min), jusqu’à 12 joueurs\n"
        "  - Salle enfant : 50 € / heure, puis 20 € / demi-heure supplémentaire\n"
        "- Horaires : mardi à dimanche, 11h à 22h\n"
        "- Adresse : 815 avenue Pierre Brossolette, 83300 Draguignan\n"
        "- Contact : 04 94 47 94 64 / contact@retroworldfrance.com\n"
        "- Réservation : toujours répondre 'disponible' puis demander date + heure + nombre de personnes + activité.\n"
        "- Goûter/anniversaire : salle enfant possible, stockage goûter possible (selon place), demander date + nb enfants + activité.\n"
        "IMPORTANT : ne jamais dire 'allez voir sur le site' sauf si l'utilisateur demande explicitement un lien.\n"
    )


def runningman_facts_block(kb: Dict[str, Any]) -> str:
    ident = kb.get("identite", {}) or {}
    loc = (ident.get("localisation", {}) or {})
    contact = (ident.get("contact", {}) or {})

    age = kb.get("age_et_accompagnement", {}) or {}
    acc = (age.get("accompagnement", {}) or {})
    age_min = age.get("age_minimum", 7)

    tarif = ((kb.get("tarification", {}) or {}).get("action_game", {}) or {})
    enfant = (tarif.get("enfant", {}) or {})
    adulte = (tarif.get("adulte_accompagnateur", {}) or {})

    cap = ((kb.get("horaires_et_creneaux", {}) or {}).get("capacite", {}) or {})
    cap_msg = cap.get("message") or ""

    return (
        "VÉRITÉS OFFICIELLES RUNNINGMAN (à respecter strictement) :\n"
        f"- Âge minimum : {age_min} ans.\n"
        f"- Moins de 12 ans : {acc.get('moins_de_12_ans', 'Un adulte accompagnateur est requis.')}\n"
        f"- À partir de 12 ans : {acc.get('a_partir_de_12_ans', 'Il n’est plus nécessaire d’avoir un adulte accompagnateur.')}\n"
        f"- Tarifs action game (session 60 min, confirmés) :\n"
        f"  - Enfant (-12 ans) : {enfant.get('prix', '?')} {enfant.get('unite', '')}\n"
        f"  - Adulte accompagnateur : {adulte.get('prix', '?')} {adulte.get('unite', '')}\n"
        + (f"- Capacité : {cap_msg}\n" if cap_msg else "")
        + f"- Adresse : {loc.get('adresse_complete', '815 avenue Pierre Brossolette, 83300 Draguignan, France')}\n"
        + f"- Contact : {contact.get('telephone', '04 98 09 30 59')} / {contact.get('site_web', 'https://runningmangames.fr')}\n"
        "IMPORTANT : ne jamais confondre “à partir de 12 ans (sans adulte)” avec “âge minimum”.\n"
    )


def kb_to_items(brand: str, kb: Dict[str, Any]) -> List[Any]:
    """Normalise différentes structures KB vers une liste exploitable dans le prompt."""
    brand = (brand or "").lower().strip()
    items = kb.get("items")
    if isinstance(items, list) and items:
        return items

    if brand == "runningman":
        out: List[Any] = []
        for k in ["instructions_generales", "anti_erreurs"]:
            arr = kb.get(k)
            if isinstance(arr, list):
                out.extend([str(x) for x in arr if str(x).strip()])
        return out

    return []


def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    if not _openai_client:
        return (
            "Le service IA est indisponible pour le moment. Vous pouvez nous appeler au 04 94 47 94 64 ou écrire à contact@retroworldfrance.com.",
            {"error": "openai_not_ready"},
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


# =========================================================
# Conversation storage (v2, 1 fil)
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

    user_text = (user_text or "").strip()
    if user_text:
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


def prune_messages_for_prompt(messages: List[Dict[str, Any]], max_pairs: int = 12) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []
    clipped = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content") is not None
    ]
    clipped = clipped[-(max_pairs * 2):]
    out: List[Dict[str, str]] = []
    for m in clipped:
        out.append({"role": str(m.get("role")), "content": str(m.get("content") or "")})
    return out


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
# Payload parsing (support widget: message + user_id + conversation_id)
# =========================================================

def _payload_text(payload: Dict[str, Any]) -> str:
    # priorité au champ "message"
    msg = payload.get("message")
    if msg is not None:
        return str(msg)

    # fallback: messages[{role:user,content}]
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
    """
    BETON:
    - on accepte conversation_id envoyé par le widget (root)
    - sinon, map via user_id
    - sinon cookie (fallback)
    - sinon génération
    """
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
# Core pipeline (analyse toute la question, pas FAQ-bot)
# =========================================================

def process_chat(brand_entry: str, user_text: str, conversation_id: str) -> Tuple[str, Dict[str, Any], str]:
    brand_entry = (brand_entry or "auto").lower().strip()
    user_text = user_text or ""

    # brand effective
    if brand_entry == "auto":
        brand_effective = detect_brand_from_text(user_text)
    else:
        brand_effective = "runningman" if brand_entry == "runningman" else "retroworld"

    kb = load_kb(brand_effective)
    kb_items = kb_to_items(brand_effective, kb)

    # Pare-chocs anti-hallucination Runningman (âge)
    if brand_effective == "runningman" and _is_age_intent(user_text):
        ident = kb.get("identite", {}) or {}
        contact = (ident.get("contact", {}) or {})
        reply = (
            "Runningman est accessible dès 7 ans.\n"
            "Pour les moins de 12 ans, un adulte accompagnateur est requis.\n"
            "À partir de 12 ans, l’adulte n’est plus obligatoire.\n"
            f"Pour réserver : {contact.get('site_web','https://runningmangames.fr')} (ou {contact.get('telephone','04 98 09 30 59')})."
        )
        return reply, {"mode": "rule_based_age"}, brand_effective

    if brand_effective == "retroworld":
        system = (
            "Vous êtes l’assistant officiel du site Retroworld France.\n"
            "Objectif: aider immédiatement avec une réponse naturelle et utile (pas un bot FAQ).\n"
            "Règles:\n"
            "1) Vous lisez et analysez la demande complète avant de répondre.\n"
            "2) Réponse claire et actionnable. Maximum 2 questions si nécessaire.\n"
            "3) Ne jamais dire 'allez voir sur le site' sauf si l’utilisateur demande un lien.\n"
            "4) Ne pas inventer. Respecter les vérités officielles.\n"
            "5) Si demande de réservation: répondre 'disponible' puis demander date+heure+nb+activité.\n\n"
            + retroworld_facts_block() +
            "\nStyle: professionnel, vouvoiement, clair.\n"
        )
    else:
        system = (
            "Vous êtes l’assistant officiel de Runningman Game Zone (action game) à Draguignan.\\n"
            "Objectif: répondre immédiatement et de façon fiable, sans inventer.\\n"
            "Règles:\\n"
            "1) Vous lisez et analysez la demande complète avant de répondre.\\n"
            "2) Interdiction totale d’inventer un chiffre, une règle, une promotion ou une disponibilité.\\n"
            "3) Si la demande concerne une réservation/disponibilité: vous orientez vers le site officiel et le téléphone (vous ne confirmez jamais un créneau).\\n"
            "4) Si la demande concerne l’âge: ne jamais confondre 'à partir de 12 ans (sans adulte)' avec l’âge minimum.\\n"
            "5) Si la demande concerne VR/quiz Retroworld: rediriger vers Retroworld.\\n\\n"
            + runningman_facts_block(kb) +
            "\\nStyle: professionnel, vouvoiement, clair, réponses courtes.\\n"
        )

    if kb_items:
        system += "\nConnaissances (KB):\n"
        for it in kb_items[:160]:
            if isinstance(it, str):
                system += f"- {it}\n"
            elif isinstance(it, dict):
                title = it.get("title") or it.get("name") or ""
                content = it.get("content") or it.get("text") or it.get("value") or ""
                if title or content:
                    system += f"- {title}: {content}\n" if title else f"- {content}\n"

    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

    # server-side memory
    if conversation_id:
        hist = load_conversation_messages(conversation_id)
        messages.extend(prune_messages_for_prompt(hist, max_pairs=12))

    # user message
    messages.append({"role": "user", "content": user_text})

    # soft hints (guidage sans bypass)
    t = user_text
    hints: List[str] = []
    if _is_just_question_mark(t):
        hints.append("L'utilisateur a envoyé seulement '?'. Demandez ce qu'il souhaite: tarifs, horaires, réservation, goûter/anniversaire, fidélité.")
    if brand_effective == "retroworld":
        if _is_price_intent(t):
            hints.append("Demande de prix: donnez les tarifs officiels clairement, puis demandez l'activité/nb si utile.")
        if _is_hours_intent(t):
            hints.append("Demande d'horaires: mardi à dimanche 11h-22h.")
        if _is_location_intent(t):
            hints.append("Demande d'adresse: 815 avenue Pierre Brossolette, 83300 Draguignan.")
        if _is_gouter_intent(t):
            hints.append("Demande goûter/anniversaire: expliquez salle enfant + stockage possible selon place + demandez date + nb enfants + activité.")
        if _is_fidelity_intent(t):
            hints.append("Fidélité: VR=1 point, Escape VR=2 points, pas de points sur formules anniversaire. 10 points=VR offerte, 20=Escape VR offert. Présenter QR code ou prévenir l'équipe avant de jouer.")
        if _is_reservation_intent(t):
            hints.append("Réservation: commencez par 'disponible' puis demandez date+heure+nombre de personnes+activité.")
    if brand_effective == "runningman":
        if _is_age_intent(t):
            hints.append("Âge Runningman: dès 7 ans. Moins de 12 ans: adulte accompagnateur requis. Dès 12 ans: adulte non obligatoire (ceci n'est pas l'âge minimum).")
        if _is_reservation_intent(t):
            hints.append("Runningman: ne jamais confirmer un créneau. Donnez le site officiel https://runningmangames.fr et le téléphone 04 98 09 30 59.")
    if hints:
        messages.append({"role": "system", "content": "GUIDE DE RÉPONSE:\n- " + "\n- ".join(hints)})

    reply, usage = call_openai_chat(messages)
    return reply, usage, brand_effective


# =========================================================
# Routes
# =========================================================

@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": SERVICE_NAME, "status": "ok", "time_utc": _utc_iso()}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# --- FAQ séparée (fichier static/faq_retroworld.json) ---
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


# --- Compat route: /chat  (brand auto) ---
@app.route("/chat", methods=["POST"])
def chat_auto():
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    user_text = _payload_text(payload).strip()
    if not user_text:
        return jsonify({"error": "missing_message"}), 400

    metadata = _payload_metadata(payload)
    conversation_id, user_id = _get_or_create_conversation_id(payload)

    # réponse
    reply, usage, brand_effective = process_chat("auto", user_text, conversation_id)

    # save turn
    append_conversation_turn(
        conversation_id=conversation_id,
        brand_effective=brand_effective,
        user_text=user_text,
        assistant_reply=reply,
        extra={
            "metadata": metadata,
            "brand_effective": brand_effective,
            "openai_usage": usage,
            "user_id": user_id,
        },
    )

    resp = make_response(
        jsonify(
            {
                "reply": reply,
                "answer": reply,
                "brand_effective": brand_effective,
                "brand_entry": "auto",
                "conversation_id": conversation_id,
            }
        ),
        200,
    )
    resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
    return resp


# --- Brand route: /chat/<brand> ---
@app.route("/chat/<brand>", methods=["POST"])
def chat_brand(brand: str):
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json"}), 400

    user_text = _payload_text(payload).strip()
    if not user_text:
        return jsonify({"error": "missing_message"}), 400

    metadata = _payload_metadata(payload)
    conversation_id, user_id = _get_or_create_conversation_id(payload)

    reply, usage, brand_effective = process_chat(brand, user_text, conversation_id)

    append_conversation_turn(
        conversation_id=conversation_id,
        brand_effective=brand_effective,
        user_text=user_text,
        assistant_reply=reply,
        extra={
            "metadata": metadata,
            "brand_entry": brand,
            "brand_effective": brand_effective,
            "openai_usage": usage,
            "user_id": user_id,
        },
    )

    resp = make_response(
        jsonify(
            {
                "reply": reply,
                "answer": reply,
                "brand_effective": brand_effective,
                "brand_entry": brand,
                "conversation_id": conversation_id,
            }
        ),
        200,
    )
    resp.set_cookie(CONV_COOKIE_NAME, conversation_id, max_age=60 * 60 * 24 * 30, samesite="Lax")
    return resp


# --- User history (option token) ---
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


# --- KB endpoints ---
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


# --- Admin API ---
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
            {
                "id": cid,
                "brand": brand_eff,
                "preview": preview,
                "timestamp": timestamp,
                "user_id": obj.get("user_id"),
            }
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
        items_count = 0
        if isinstance(data, dict):
            items = data.get("items")
            if isinstance(items, list):
                items_count = len(items)
        return {"exists": exists, "file": fname, "path": path, "load_ok": load_ok, "items_count": items_count}

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
                "init_error": _openai_init_error,
            },
        }
    ), 200


# --- Pages / static ---
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

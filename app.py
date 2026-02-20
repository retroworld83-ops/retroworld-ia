from __future__ import annotations

import csv
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, Response, abort, jsonify, redirect, request, send_from_directory
from flask_cors import CORS

# Optional OpenAI import (degrade gracefully if missing)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# Optional YAML import
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


# -----------------------------
# App
# -----------------------------

app = Flask(__name__, static_folder="static", static_url_path="/static")


# -----------------------------
# Utils
# -----------------------------


def utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def get_env_list(key: str, default: str = "") -> List[str]:
    raw = (os.getenv(key, default) or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    seen = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def norm_words(text: str) -> List[str]:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9àâäçéèêëîïôöùûüÿñæœ\s-]", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    words = [w for w in t.split(" ") if len(w) >= 3]
    return words[:80]


def json_load(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return default


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def clip_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars and max_chars > 20 and len(s) > max_chars:
        return s[: max_chars - 1].rstrip() + "…"
    return s



def _booking_intent(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"\bréserv|\breservation|\bdispo|\bcréneau|\banniversaire\b|\bvenir\s+le\b|\bbooking\b", t, flags=re.I))


def _contact_for_brand(brand: str) -> str:
    b = brand_or_default(brand)
    if b == "runningman":
        return "Runningman Games : 04 98 09 30 59 • https://www.runningmangames.fr"
    if b == "retroworld":
        return "Retroworld : 04 94 47 94 64 • contact@retroworldfrance.com • https://www.retroworldfrance.com"
    # unknown / Enigmaniac
    return "Contactez l'équipe de l'établissement concerné pour confirmer."


def postprocess_reply(brand: str, user_text: str, reply: str) -> str:
    """Final guardrails: avoid promising bookings, and add a confirmation disclaimer when needed."""
    r = (reply or "").strip()

    # If the model accidentally promises a booking, neutralize it.
    risky = re.search(r"\bje\s+vous\s+bloque\b|\bcr[eé]neau\s+(confirm|bloqu|r[eé]serv)|\bc['’]?est\s+r[eé]serv|\bconfirm[eé]\b", r, flags=re.I)
    if risky:
        r = re.sub(r"\bje\s+vous\s+bloque\b", "je transmets votre demande", r, flags=re.I)
        r = re.sub(r"\bc['’]?est\s+r[eé]serv(e|é)?\b", "c'est à confirmer par l'équipe", r, flags=re.I)
        r = re.sub(r"\bconfirm(e|é)\b", "à confirmer", r, flags=re.I)

    # When user is asking about booking/availability, always include the "no planning access" disclaimer (once).
    if _booking_intent(user_text) and not re.search(r"pas\s+acc[eè]s\s+au\s+(planning|logiciel)|\b[aà]\s+confirmer\b|\bconfirmer\s+la\s+disponibilit", r, flags=re.I):
        r += "\n\n⚠️ Je n’ai pas accès au planning en temps réel, donc la disponibilité est à confirmer par l’équipe. " + _contact_for_brand(brand)

    return r
def host_from_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    # very small parser to avoid urllib overhead
    url = re.sub(r"^https?://", "", url, flags=re.I)
    url = url.split("/")[0]
    url = url.split(":")[0]
    return url.lower().strip()


def debug_log(msg: str) -> None:
    if DEBUG_LOGS:
        print(f"[dbg] {msg}", flush=True)


# -----------------------------
# Env / config
# -----------------------------

DEBUG_LOGS = (os.getenv("DEBUG_LOGS") or "").strip().lower() in ("1", "true", "yes", "on")

PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").strip()
DEFAULT_BRAND_ID = (os.getenv("BRAND_ID") or "retroworld").strip().lower() or "retroworld"

ALLOWED_ORIGINS = get_env_list(
    "ALLOWED_ORIGINS",
    "https://www.retroworldfrance.com,https://retroworldfrance.com,https://www.runningmangames.fr,https://runningmangames.fr",
)

# FAQ / public brand exposure
FAQ_ENABLED_BRANDS = get_env_list("FAQ_ENABLED_BRANDS", "retroworld,runningman")
PUBLIC_BRANDS = get_env_list("PUBLIC_BRANDS", ",".join(FAQ_ENABLED_BRANDS)) or FAQ_ENABLED_BRANDS

# CORS: if list is defined -> restrict; else allow all (dev)
if ALLOWED_ORIGINS:
    CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})
else:
    CORS(app)


# OpenAI
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-5.2").strip()
OPENAI_FALLBACK_MODELS = get_env_list("OPENAI_FALLBACK_MODELS", "gpt-4.1-mini,gpt-4.1,gpt-4o")
OPENAI_REASONING_EFFORT = (os.getenv("OPENAI_REASONING_EFFORT") or "none").strip().lower()
OPENAI_TEMPERATURE = safe_float(os.getenv("OPENAI_TEMPERATURE"), 0.3)
OPENAI_MAX_OUTPUT_TOKENS = safe_int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS"), 900)

# Conversation memory for public chat
OPENAI_HISTORY_MODE = (os.getenv("OPENAI_HISTORY_MODE") or "full").strip().lower()  # full | recent
OPENAI_MAX_HISTORY_PAIRS = safe_int(os.getenv("OPENAI_MAX_HISTORY_PAIRS"), 120)
OPENAI_PROMPT_CHAR_BUDGET = safe_int(os.getenv("OPENAI_PROMPT_CHAR_BUDGET"), 32000)

# Optional: cap final reply length (useful for website chat widgets)
CHAT_MAX_REPLY_CHARS = safe_int(os.getenv("CHAT_MAX_REPLY_CHARS") or os.getenv("BT_MAX_REPLY_CHARS"), 0)


# Admin tokens
ADMIN_API_TOKEN = (os.getenv("ADMIN_API_TOKEN") or "").strip()
ADMIN_DASHBOARD_TOKEN = (os.getenv("ADMIN_DASHBOARD_TOKEN") or "").strip()
ADMIN_EMAILS = get_env_list("ADMIN_EMAILS")

USER_HISTORY_TOKEN = (os.getenv("USER_HISTORY_TOKEN") or "").strip()


# Internal assistant (BT)
BT_API_TOKEN = (os.getenv("BT_API_TOKEN") or "").strip()
BT_PROFILE_PATH = (os.getenv("BT_PROFILE_PATH") or "config/bt_profile.yaml").strip()
BT_MAX_OUTPUT_TOKENS = safe_int(os.getenv("BT_MAX_OUTPUT_TOKENS"), 400)
BT_MAX_REPLY_CHARS = safe_int(os.getenv("BT_MAX_REPLY_CHARS"), 500)


# Storage
# Important: on Render (and in many containers), writing under /app can be restricted.
# /tmp is writable in standard Linux containers.
CONV_DIR = Path(os.getenv("CONV_DIR") or "/tmp/retroworld_conversations")
CONV_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Brand config (3 établissements)
# -----------------------------


def _default_brand_system_prompts() -> Dict[str, str]:
    # Guardrails shared by all brands (multi-site, no booking promises, no hallucinations)
    common = """Règles globales (très important) :
- Répondez en français, ton professionnel, vouvoiement.
- Ne pas inventer : si une information manque ou n'est pas certaine, dites-le et proposez le bon contact.
- Vous n'avez PAS accès à un logiciel de réservation ni au planning en temps réel : ne promettez jamais qu'un créneau est "bloqué", "confirmé" ou "réservé".
  Vous pouvez aider à préparer la demande (date, heure, nombre de personnes, activité), puis inviter le client à contacter l'équipe pour confirmation.
- Si une demande implique un devis, une privatisation, une facture, un partenariat, une réclamation, une exception ou une décision commerciale : indiquez que cela nécessite validation par l'équipe et proposez un contact.
- Multi-établissements : Retroworld, Runningman Games et Enigmaniac sont des entités distinctes au sein du même bâtiment. Si la question croise plusieurs établissements, répondez sans confusion, en séparant clairement les informations par établissement."""

    retroworld = f"""Vous êtes l'assistant officiel de Retroworld France (Draguignan).

{common}

Infos Retroworld (fiables) :
- Adresse : 815 avenue Pierre Brossolette, 83300 Draguignan, France
- Horaires : mardi à dimanche, 11h à 22h (fermé lundi)
- Contact : 04 94 47 94 64, contact@retroworldfrance.com, https://www.retroworldfrance.com
- Activités : Jeux VR (15 €/joueur, jusqu'à 5 joueurs), Escape Game VR (30 €/joueur), Quiz interactifs (8€ 30min, 15€ 60min, 20€ 90min jusqu'à 12 joueurs), salle enfant (50 €/h + 20 €/30 min), salle d'attente équipée.
- Hygiène : casques VR professionnels (Vive Pro 2, Meta Quest 3) nettoyés entre chaque session.

Réservations (important) :
- Vous pouvez demander : date, heure souhaitée, nombre de personnes, activité, et si c'est un anniversaire.
- Ne confirmez pas un créneau. Dites que l'équipe confirmera la disponibilité par téléphone ou email.

Runningman Games (partenaire) :
- Action game, même bâtiment, mais réservé directement chez eux : https://www.runningmangames.fr, 04 98 09 30 59.
- Retroworld ne gère pas leurs réservations.

Enigmaniac :
- Si la demande concerne Enigmaniac et que l'information n'est pas dans la FAQ disponible, dites-le et proposez de contacter l'équipe concernée."""

    runningman = f"""Vous êtes l'assistant d'orientation pour Runningman Games (action game) et, si besoin, vous pouvez orienter vers Retroworld et Enigmaniac.

{common}

Runningman Games :
- Réservations et infos Runningman : https://www.runningmangames.fr, 04 98 09 30 59.
- Runningman Games est une entité distincte : ne promettez pas de réservation, invitez à les contacter.

Retroworld (même bâtiment) :
- Retroworld France : https://www.retroworldfrance.com, 04 94 47 94 64.
- Activités VR / Escape VR / Quiz et informations pratiques selon la FAQ Retroworld.

Enigmaniac :
- Si la demande concerne Enigmaniac, répondez uniquement si l'info est disponible dans la FAQ Enigmaniac ; sinon, proposez le contact Enigmaniac."""

    enigmaniac = f"""Vous êtes l'assistant d'orientation pour Enigmaniac.

{common}

Règle Enigmaniac :
- Répondez uniquement avec les informations présentes dans la FAQ Enigmaniac.
- Si l'information n'est pas disponible (tarifs, horaires, adresse exacte, réservation), dites-le clairement et invitez à contacter l'équipe Enigmaniac."""

    return {"retroworld": retroworld, "runningman": runningman, "enigmaniac": enigmaniac}


def load_brands_config() -> Dict[str, Any]:
    """Load config/brands.yaml if present; otherwise return safe defaults."""
    cfg_path = Path(os.getenv("BRANDS_CONFIG_PATH") or "config/brands.yaml")

    # Defaults
    prompts = _default_brand_system_prompts()
    base: Dict[str, Any] = {
        "default_brand": DEFAULT_BRAND_ID,
        "brands": {
            "retroworld": {
                "display_name": "Retroworld",
                "faq_enabled": True,
                "domains": ["retroworldfrance.com", "www.retroworldfrance.com"],
                "kb_file": "kb_retroworld.json",
                "system_prompt": prompts["retroworld"],
                "keywords": ["retroworld", "vr", "escape", "escape game vr", "quiz", "anniversaire"],
            },
            "runningman": {
                "display_name": "Runningman Games",
                "faq_enabled": True,
                "domains": ["runningmangames.fr", "www.runningmangames.fr"],
                "kb_file": "kb_runningman.json",
                "system_prompt": prompts["runningman"],
                "keywords": ["runningman", "running man", "action game"],
            },
            "enigmaniac": {
                "display_name": "Enigmaniac",
                "faq_enabled": False,
                "domains": [],
                "kb_file": "kb_enigmaniac.json",
                "system_prompt": prompts["enigmaniac"],
                "keywords": ["enigmaniac", "énigmaniac", "enig"],
            },
        },
    }

    if yaml is None or not cfg_path.exists():
        return base

    try:
        data = yaml.safe_load(cfg_path.read_text("utf-8"))
        if not isinstance(data, dict):
            return base
        # merge shallow
        merged = base
        if isinstance(data.get("default_brand"), str):
            merged["default_brand"] = str(data["default_brand"]).strip().lower() or merged["default_brand"]
        if isinstance(data.get("brands"), dict):
            for bid, bcfg in data["brands"].items():
                if not isinstance(bid, str) or not isinstance(bcfg, dict):
                    continue
                bid2 = bid.strip().lower()
                merged["brands"].setdefault(bid2, {})
                merged["brands"][bid2].update(bcfg)
        return merged
    except Exception:
        return base


BRANDS_CFG = load_brands_config()
BRANDS: Dict[str, Dict[str, Any]] = {
    str(k).lower(): (v if isinstance(v, dict) else {})
    for k, v in (BRANDS_CFG.get("brands") or {}).items()
    if isinstance(k, str)
}
DEFAULT_BRAND_ID = str(BRANDS_CFG.get("default_brand") or DEFAULT_BRAND_ID).strip().lower() or "retroworld"

# Restrict FAQ/public brands to known brand ids
FAQ_ENABLED_BRANDS = [b for b in FAQ_ENABLED_BRANDS if b in BRANDS]
if not FAQ_ENABLED_BRANDS:
    FAQ_ENABLED_BRANDS = [DEFAULT_BRAND_ID] if DEFAULT_BRAND_ID in BRANDS else brand_ids()
PUBLIC_BRANDS = [b for b in PUBLIC_BRANDS if b in BRANDS] or FAQ_ENABLED_BRANDS



def brand_ids() -> List[str]:
    ids = sorted([k for k in BRANDS.keys() if k])
    return ids or ["retroworld"]


def brand_or_default(bid: str) -> str:
    b = (bid or "").strip().lower()
    return b if b in BRANDS else DEFAULT_BRAND_ID


def domain_to_brand_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for bid, cfg in BRANDS.items():
        doms = cfg.get("domains")
        if isinstance(doms, list):
            for d in doms:
                h = host_from_url(str(d))
                if h:
                    m[h] = bid
    return m


DOMAIN_TO_BRAND = domain_to_brand_map()


# -----------------------------
# Knowledge base (FAQ) per brand
# -----------------------------


def load_kb_file(path: str, brand: str) -> Dict[str, Any]:
    p = Path(path)
    obj = json_load(p, {})
    if not isinstance(obj, dict):
        obj = {}

    items = obj.get("items")
    if not isinstance(items, list):
        items = []

    cleaned = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = str(it.get("q") or "").strip()
        a = str(it.get("a") or "").strip()
        tags = it.get("tags")
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).strip().lower() for t in tags if str(t).strip()]
        if q and a:
            cleaned.append({"q": q, "a": a, "tags": tags})

    return {
        "brand": str(obj.get("brand") or brand),
        "updated": str(obj.get("updated") or ""),
        "items": cleaned,
    }


def save_kb_file(path: str, brand: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    cleaned: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = str(it.get("q") or "").strip()
        a = str(it.get("a") or "").strip()
        tags = it.get("tags")
        if not isinstance(tags, list):
            tags = []
        tags2 = [str(t).strip().lower() for t in tags if str(t).strip()]
        if q and a:
            cleaned.append({"q": q, "a": a, "tags": tags2})

    obj = {
        "brand": brand,
        "updated": utc_iso(),
        "items": cleaned,
    }
    json_dump(Path(path), obj)
    return obj


KB_BY_BRAND: Dict[str, Dict[str, Any]] = {}
for bid in brand_ids():
    kb_file = str(BRANDS.get(bid, {}).get("kb_file") or f"kb_{bid}.json")
    KB_BY_BRAND[bid] = load_kb_file(kb_file, bid)


def refresh_kb(brand: str) -> Dict[str, Any]:
    b = brand_or_default(brand)
    kb_file = str(BRANDS.get(b, {}).get("kb_file") or f"kb_{b}.json")
    KB_BY_BRAND[b] = load_kb_file(kb_file, b)
    return KB_BY_BRAND[b]


def retrieve_kb_snippets(kb: Dict[str, Any], query: str, top_k: int = 6) -> List[Dict[str, str]]:
    items = kb.get("items")
    if not isinstance(items, list) or not query:
        return []

    qw = norm_words(query)
    if not qw:
        return []
    qset = set(qw)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = f"{it.get('q','')} {it.get('a','')}"
        words = set(norm_words(text))
        tags = set((it.get("tags") or []) if isinstance(it.get("tags"), list) else [])
        overlap = len(qset.intersection(words))
        tag_overlap = len(qset.intersection(tags))
        score = overlap + 1.5 * tag_overlap
        if score > 0:
            scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, str]] = []
    for _, it in scored[:top_k]:
        out.append({"q": str(it.get("q") or ""), "a": str(it.get("a") or "")})
    return out


# -----------------------------
# Conversation store
# -----------------------------


@dataclass
class ConversationStore:
    root: Path

    def _safe_id(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"[^a-zA-Z0-9_-]", "", s)
        return s[:80]

    def path_for(self, conv_id: str) -> Path:
        cid = self._safe_id(conv_id)
        if not cid:
            cid = "conv_" + uuid.uuid4().hex[:12]
        return self.root / f"{cid}.json"

    def load(self, conv_id: str) -> Dict[str, Any]:
        p = self.path_for(conv_id)
        obj = json_load(p, None)
        if not isinstance(obj, dict):
            obj = {
                "id": self._safe_id(conv_id) or ("conv_" + uuid.uuid4().hex[:12]),
                "brand": DEFAULT_BRAND_ID,
                "user_id": "",
                "created": utc_iso(),
                "updated": utc_iso(),
                "messages": [],
            }
        if not isinstance(obj.get("messages"), list):
            obj["messages"] = []
        obj.setdefault("id", self._safe_id(conv_id) or obj.get("id"))
        obj.setdefault("created", utc_iso())
        obj.setdefault("updated", utc_iso())
        obj.setdefault("brand", DEFAULT_BRAND_ID)
        obj.setdefault("user_id", "")
        return obj

    def save(self, obj: Dict[str, Any]) -> None:
        cid = str(obj.get("id") or "")
        p = self.path_for(cid)
        json_dump(p, obj)

    def append(
        self,
        conv_id: str,
        role: str,
        content: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        obj = self.load(conv_id)
        msg: Dict[str, Any] = {"role": role, "content": str(content or ""), "ts": utc_iso()}
        if extra and isinstance(extra, dict):
            msg["extra"] = extra
        obj["messages"].append(msg)
        obj["updated"] = utc_iso()
        self.save(obj)
        return obj

    def list_summaries(self, limit: int = 400) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        files = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[:limit]:
            obj = json_load(p, {})
            if not isinstance(obj, dict):
                continue
            msgs = obj.get("messages")
            if not isinstance(msgs, list):
                msgs = []

            preview = ""
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "user":
                    preview = str(m.get("content") or "")
                    break
            if not preview and msgs:
                last = msgs[-1]
                if isinstance(last, dict):
                    preview = str(last.get("content") or "")
            preview = re.sub(r"\s+", " ", preview).strip()
            if len(preview) > 140:
                preview = preview[:139].rstrip() + "…"

            out.append(
                {
                    "id": str(obj.get("id") or p.stem),
                    "brand": str(obj.get("brand") or "unknown"),
                    "user_id": str(obj.get("user_id") or ""),
                    "created": str(obj.get("created") or ""),
                    "updated": str(obj.get("updated") or ""),
                    "message_count": len(msgs),
                    "flags": conversation_flags(obj),
                    "preview": preview,
                }
            )
        return out


# -----------------------------
# Conversation analysis (admin flags)
# -----------------------------


def _last_texts(messages: List[Dict[str, Any]], role: str, limit: int = 8) -> str:
    parts: List[str] = []
    for m in reversed(messages or []):
        if not isinstance(m, dict):
            continue
        if str(m.get("role") or "") != role:
            continue
        parts.append(str(m.get("content") or ""))
        if len(parts) >= limit:
            break
    return " ".join(reversed(parts)).lower()


def conversation_flags(obj: Dict[str, Any]) -> List[str]:
    """Heuristic flags to help the admin find conversations that may need attention."""
    flags: set[str] = set()
    brand = str(obj.get("brand") or "").strip().lower()
    msgs = obj.get("messages") if isinstance(obj.get("messages"), list) else []
    user_text = _last_texts(msgs, "user", limit=10)
    assistant_text = _last_texts(msgs, "assistant", limit=10)

    # User intent flags
    if re.search(r"\bdevis\b|\bprivati|\bfactur|\bentrepris|\bassociation\b|\bce\b|\bteam\s*building\b|\bgroupe\b", user_text, flags=re.I):
        flags.add("devis")
        flags.add("a_valider")

    if re.search(r"\bréserv|\breservation|\bdispo|\bcréneau|\banniversaire\b|\bvenir\s+le\b", user_text, flags=re.I):
        flags.add("reservation")

    if re.search(r"\brembours|\bannul|\bréclam|\bplainte|\bprobl[eè]me\b|\binsatisf|\blitige\b", user_text, flags=re.I):
        flags.add("reclamation")
        flags.add("a_valider")

    if re.search(r"\bpartenariat\b|\bpresse\b|\bsponsor\b|\binfluence|\bpub\b|\bcommercial\b", user_text, flags=re.I):
        flags.add("a_valider")

    # Cross-site mentions
    mentions = set(_mentioned_brands(user_text))
    if len(mentions) >= 2:
        flags.add("croise")
    else:
        if "runningman" in mentions and brand != "runningman":
            flags.add("croise")
        if "retroworld" in mentions and brand != "retroworld":
            flags.add("croise")
        if "enigmaniac" in mentions and brand != "enigmaniac":
            flags.add("croise")

    # Risk: assistant accidentally promises booking/confirmation
    if re.search(r"\bje\s+vous\s+bloque\b|\bcr[eé]neau\s+(confirm|bloqu|r[eé]serv)|\bc['’]?est\s+r[eé]serv|\bconfirm[eé]\b", assistant_text, flags=re.I):
        flags.add("promesse_resa")

    # Uncertainty language that may warrant review (light signal)
    if re.search(r"\bje\s+pense\b|\bpeut[- ]?être\b|\bprobablement\b", assistant_text, flags=re.I):
        flags.add("a_relire")

    return sorted(flags)




STORE = ConversationStore(CONV_DIR)


# -----------------------------
# OpenAI call (public chat)
# -----------------------------


OPENAI_CLIENT = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:  # pragma: no cover
        debug_log(f"openai init failed: {e}")
        OPENAI_CLIENT = None


def call_openai_chat(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any], str]:
    """Returns: (assistant_text, usage, model_used)"""
    if OPENAI_CLIENT is None:
        return "Le service IA n'est pas configuré (OPENAI_API_KEY manquante).", {"error": "openai_not_configured"}, ""

    models_to_try = [OPENAI_MODEL] + [m for m in OPENAI_FALLBACK_MODELS if m and m != OPENAI_MODEL]
    last_err = None

    for model in models_to_try:
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "input": messages,
                "max_output_tokens": int(OPENAI_MAX_OUTPUT_TOKENS),
                "store": False,
            }

            eff = (OPENAI_REASONING_EFFORT or "none").strip().lower()
            if eff in ("low", "medium", "high"):
                kwargs["reasoning"] = {"effort": eff}
            else:
                kwargs["temperature"] = float(OPENAI_TEMPERATURE)

            resp = OPENAI_CLIENT.responses.create(**kwargs)
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

            return text or "(Réponse vide)", usage, model
        except Exception as e:  # pragma: no cover
            last_err = e
            debug_log(f"openai call failed on {model}: {e}")
            continue

    return "Le service IA est temporairement indisponible.", {"error": "openai_call_failed", "detail": str(last_err) if last_err else ""}, ""


# -----------------------------
# Prompt builder
# -----------------------------


def _mentioned_brands(user_text: str) -> List[str]:
    """Detect explicit brand mentions in the text (not generic activity keywords)."""
    t = (user_text or "").lower()
    found: List[str] = []
    patterns = {
        "retroworld": [r"\bretroworld\b"],
        "runningman": [r"\brunningman\b", r"\brunning\s*man\b"],
        "enigmaniac": [r"\benigmaniac\b", r"\bénigmaniac\b"],
    }
    for bid, pats in patterns.items():
        for pat in pats:
            if re.search(pat, t, flags=re.I):
                found.append(bid)
                break
    # de-dup while preserving order
    out: List[str] = []
    seen = set()
    for b in found:
        if b in BRANDS and b not in seen:
            out.append(b)
            seen.add(b)
    return out


def build_system_prompt(brand: str, user_text: str) -> str:
    """Build a per-request system prompt with relevant FAQ snippets."""
    b = brand_or_default(brand)
    cfg = BRANDS.get(b, {})
    base = str(cfg.get("system_prompt") or "").strip()

    def _append_snippets(lines: List[str], bid: str, title: str, top_k: int = 6) -> None:
        kb = KB_BY_BRAND.get(bid) or {}
        snippets = retrieve_kb_snippets(kb, user_text, top_k=top_k)
        if not snippets:
            return
        lines.append(f"\n\n{title} (extraits) :")
        for s in snippets:
            q = (s.get("q") or "").strip()
            a = (s.get("a") or "").strip()
            if q and a:
                lines.append(f"- Q: {q}\n  A: {a}")

    lines: List[str] = [base] if base else []

    brand_name = str(cfg.get("display_name") or b)
    _append_snippets(lines, b, f"FAQ {brand_name}", top_k=6)

    mentioned = [x for x in _mentioned_brands(user_text) if x != b]
    for other in mentioned:
        ocfg = BRANDS.get(other, {})
        other_name = str(ocfg.get("display_name") or other)
        _append_snippets(lines, other, f"FAQ {other_name}", top_k=3)

    return "\n".join([ln for ln in lines if ln]).strip()


# -----------------------------
# Auth helpers

# -----------------------------


def _token_from_request() -> str:
    return (
        (request.headers.get("X-Admin-Token") or "").strip()
        or (request.args.get("token") or "").strip()
        or (request.headers.get("Authorization") or "").replace("Bearer", "").strip()
    )


def require_admin() -> None:
    # If no token configured -> open (dev)
    if not ADMIN_API_TOKEN and not ADMIN_DASHBOARD_TOKEN:
        return
    token = _token_from_request()
    ok = False
    if ADMIN_API_TOKEN and token == ADMIN_API_TOKEN:
        ok = True
    if ADMIN_DASHBOARD_TOKEN and token == ADMIN_DASHBOARD_TOKEN:
        ok = True
    if not ok:
        abort(401, description="admin_token_required")


def require_user_history() -> None:
    if not USER_HISTORY_TOKEN:
        return
    token = ((request.headers.get("X-User-History-Token") or "").strip() or (request.args.get("token") or "").strip())
    if token != USER_HISTORY_TOKEN:
        abort(401, description="user_history_token_required")


def require_bt() -> None:
    if not BT_API_TOKEN:
        abort(403, description="bt_not_configured")
    token = (request.headers.get("X-BT-Token") or "").strip() or (request.args.get("token") or "").strip()
    if token != BT_API_TOKEN:
        abort(401, description="bt_token_required")


# -----------------------------
# Brand detection
# -----------------------------


def detect_brand(payload: Dict[str, Any], user_text: str) -> str:
    """Choose a primary brand context for the request.

    Priority:
    1) explicit brand_id in payload / headers / query
    2) origin / referrer domain (if the request is sent directly from a site)
    3) explicit brand name mention in the message (retroworld / runningman / enigmaniac)
       - if multiple brands are mentioned, keep DEFAULT_BRAND_ID to avoid confusion (we'll answer cross-site in the prompt)
    4) specific activity keywords (fallback)
    """
    # 1) explicit
    forced = str(payload.get("brand_id") or payload.get("brandId") or payload.get("brand") or "").strip().lower()
    if forced in BRANDS:
        return forced

    forced2 = (request.headers.get("X-Brand-Id") or request.args.get("brand_id") or "").strip().lower()
    if forced2 in BRANDS:
        return forced2

    # 2) by origin/referrer domain (only works if you call /chat directly from your domain, not from an iframe)
    origin = request.headers.get("Origin") or ""
    ref = request.headers.get("Referer") or ""
    host = host_from_url(origin) or host_from_url(ref)
    if host and host in DOMAIN_TO_BRAND:
        return DOMAIN_TO_BRAND[host]

    # 3) explicit brand mentions
    mentioned = _mentioned_brands(user_text)
    if len(mentioned) == 1:
        return mentioned[0]
    if len(mentioned) > 1:
        return DEFAULT_BRAND_ID

    # 4) fallback keywords (keep them specific to avoid misrouting)
    txt = (user_text or "").lower()
    # Runningman intent
    if re.search(r"\baction\s*game\b", txt, flags=re.I):
        return "runningman" if "runningman" in BRANDS else DEFAULT_BRAND_ID

    return DEFAULT_BRAND_ID


# -----------------------------
# History clipping

# -----------------------------


def _clip_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if OPENAI_HISTORY_MODE == "recent":
        return messages[-(OPENAI_MAX_HISTORY_PAIRS * 2) :]

    if len(messages) > OPENAI_MAX_HISTORY_PAIRS * 2:
        messages = messages[-(OPENAI_MAX_HISTORY_PAIRS * 2) :]

    def total_chars(msgs: List[Dict[str, str]]) -> int:
        return sum(len((m.get("content") or "")) for m in msgs)

    while messages and total_chars(messages) > OPENAI_PROMPT_CHAR_BUDGET:
        messages = messages[2:] if len(messages) >= 2 else messages[1:]
    return messages


def _make_conv_id(prefix: str) -> str:
    p = re.sub(r"[^a-z0-9_-]", "", (prefix or "conv").lower())
    if not p:
        p = "conv"
    return f"{p}_{uuid.uuid4().hex[:12]}"


# -----------------------------
# Routes (public)
# -----------------------------


@app.get("/")
def home():
    return redirect("/static/chat-widget.html", code=302)


@app.get("/health")
def health():
    return jsonify({"ok": True, "ts": utc_iso(), "brands": brand_ids()})


@app.get("/brands.json")
def brands_json():
    out = []
    for bid in PUBLIC_BRANDS:
        cfg = BRANDS.get(bid, {})
        out.append(
            {
                "id": bid,
                "name": str(cfg.get("display_name") or bid),
                "display_name": str(cfg.get("display_name") or bid),
                "domains": cfg.get("domains") if isinstance(cfg.get("domains"), list) else [],
                "faq_enabled": bool(cfg.get("faq_enabled")) if isinstance(cfg, dict) else False,
            }
        )
    return jsonify({"default_brand": DEFAULT_BRAND_ID, "public_base_url": PUBLIC_BASE_URL, "brands": out})


@app.get("/faq")
def faq_page():
    # Default to the first FAQ-enabled brand
    req_brand = (request.args.get("brand_id") or request.args.get("brand") or "").strip().lower()
    brand = brand_or_default(req_brand) if req_brand else (FAQ_ENABLED_BRANDS[0] if FAQ_ENABLED_BRANDS else DEFAULT_BRAND_ID)

    if brand not in FAQ_ENABLED_BRANDS:
        brand_label = str(BRANDS.get(brand, {}).get("display_name") or brand)
        return f"""<!doctype html>
<html lang='fr'><head>
  <meta charset='utf-8'/><meta name='viewport' content='width=device-width,initial-scale=1'/>
  <title>FAQ indisponible</title>
  <style>body{{background:#0b0f14;color:#e9eef5;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;padding:18px;}}
  .card{{max-width:860px;margin:0 auto;background:#121824;border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px;}}
  a{{color:#93c5fd;}}</style>
</head><body>
  <div class='card'>
    <h1 style='margin:0 0 10px 0'>FAQ {brand_label}</h1>
    <p style='margin:0;color:#9fb1ca'>La FAQ de cet établissement n'est pas encore publiée.</p>
  </div>
</body></html>""", 404

    kb = refresh_kb(brand)
    items = kb.get("items", []) if isinstance(kb, dict) else []

    html_items = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = str(it.get("q") or "").strip()
        a = str(it.get("a") or "").strip()
        if q and a:
            html_items.append(
                f"<details><summary>{q}</summary><div style='margin:8px 0 0 0; color:#cbd5e1'>{a}</div></details>"
            )

    brand_label = str(BRANDS.get(brand, {}).get("display_name") or brand)
    page = f"""<!doctype html>
<html lang='fr'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width,initial-scale=1'/>
  <title>FAQ {brand_label}</title>
  <style>
    body{{background:#0b0f14;color:#e9eef5;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;padding:18px;}}
    .card{{max-width:860px;margin:0 auto;background:#121824;border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px;}}
    h1{{font-size:18px;margin:0 0 10px 0;}}
    details{{border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:10px 12px;margin:10px 0;background:rgba(255,255,255,.03);}}
    summary{{cursor:pointer;font-weight:800;}}
    a{{color:#93c5fd;}}
  </style>
</head>
<body>
  <div class='card'>
    <h1>FAQ {brand_label}</h1>
    <p style='margin:0 0 10px 0;color:#9fb1ca'>Version JSON: <a href='/faq.json?brand_id={brand}'>/faq.json?brand_id={brand}</a></p>
    {''.join(html_items)}
  </div>
</body>
</html>"""
    return page


@app.get("/faq.json")
def faq_json():
    bid = (request.args.get("brand_id") or "").strip().lower()
    if bid:
        b = brand_or_default(bid)
        if b not in FAQ_ENABLED_BRANDS:
            return jsonify({"error": "faq_unavailable_for_brand", "brand": b, "items": []}), 404
        return jsonify({b: refresh_kb(b)})

    # all FAQ-enabled brands only
    data = {b: refresh_kb(b) for b in FAQ_ENABLED_BRANDS}
    return jsonify(data)


def _chat_impl(brand: str):
    payload = request.get_json(silent=True) or {}
    user_text = str(payload.get("message") or payload.get("text") or "").strip()
    if not user_text:
        return jsonify({"error": "missing_message"}), 400

    brand = brand_or_default(brand)

    conv_id = str(payload.get("conversation_id") or payload.get("conversationId") or "").strip()
    if not conv_id:
        conv_id = _make_conv_id(brand[:2])

    user_id = str(payload.get("user_id") or payload.get("userId") or "").strip()

    conv = STORE.load(conv_id)
    conv["brand"] = brand
    if user_id:
        conv["user_id"] = user_id
    STORE.save(conv)

    STORE.append(conv_id, "user", user_text)

    conv = STORE.load(conv_id)
    history_msgs: List[Dict[str, str]] = []
    for m in conv.get("messages", []):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        content = str(m.get("content") or "")
        if role in ("user", "assistant") and content:
            history_msgs.append({"role": role, "content": content})

    history_msgs = _clip_history(history_msgs)
    system_prompt = build_system_prompt(brand, user_text)
    messages = [{"role": "system", "content": system_prompt}] + history_msgs

    reply, usage, model_used = call_openai_chat(messages)
    reply = postprocess_reply(brand, user_text, reply)
    reply = clip_text(reply, CHAT_MAX_REPLY_CHARS)

    STORE.append(conv_id, "assistant", reply, extra={"model": model_used, "usage": usage} if (model_used or usage) else None)

    return jsonify({"conversation_id": conv_id, "brand": brand, "reply": reply, "model": model_used, "usage": usage})


@app.post("/chat")
def chat_auto():
    payload = request.get_json(silent=True) or {}
    user_text = str(payload.get("message") or payload.get("text") or "").strip()
    brand = detect_brand(payload, user_text)
    debug_log(f"chat_auto brand={brand}")
    return _chat_impl(brand)


@app.post("/chat/<brand_id>")
def chat_brand(brand_id: str):
    return _chat_impl(brand_or_default(brand_id))


# Backward-compatible endpoints
@app.post("/chat/retroworld")
def chat_retroworld():
    return _chat_impl("retroworld")


@app.post("/chat/runningman")
def chat_runningman():
    return _chat_impl("runningman")


@app.post("/chat/enigmaniac")
def chat_enigmaniac():
    return _chat_impl("enigmaniac")


# -----------------------------
# Admin
# -----------------------------


@app.get("/admin")
def admin_page():
    return send_from_directory(app.static_folder, "admin.html")


@app.get("/admin/faq")
def admin_faq_page():
    return send_from_directory(app.static_folder, "admin-faq.html")


@app.get("/admin/api/diag")
def admin_diag():
    require_admin()
    return jsonify(
        {
            "openai": {
                "client_ready": OPENAI_CLIENT is not None,
                "model": OPENAI_MODEL,
                "fallback_models": OPENAI_FALLBACK_MODELS,
                "reasoning_effort": OPENAI_REASONING_EFFORT,
                "temperature": OPENAI_TEMPERATURE,
                "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
            },
            "chat": {"max_reply_chars": CHAT_MAX_REPLY_CHARS},
            "brands": brand_ids(),
            "admin": {"emails": ADMIN_EMAILS},
            "storage": {"conv_dir": str(CONV_DIR), "count": len(list(CONV_DIR.glob("*.json")))},
            "ts": utc_iso(),
        }
    )


@app.get("/admin/api/brands")
def admin_brands():
    require_admin()
    faq_only = str(request.args.get("faq_only") or "").strip() in ("1", "true", "yes")

    out = []
    for bid in brand_ids():
        cfg = BRANDS.get(bid, {})
        faq_enabled = bool(cfg.get("faq_enabled")) if isinstance(cfg, dict) else False
        if faq_only and not faq_enabled:
            continue
        out.append(
            {
                "id": bid,
                "name": str(cfg.get("display_name") or bid),
                "display_name": str(cfg.get("display_name") or bid),
                "kb_file": str(cfg.get("kb_file") or ""),
                "faq_enabled": faq_enabled,
            }
        )
    return jsonify({"default_brand": DEFAULT_BRAND_ID, "faq_enabled_brands": FAQ_ENABLED_BRANDS, "brands": out})


@app.get("/admin/api/conversations")
def admin_conversations():
    require_admin()
    return jsonify(STORE.list_summaries())


@app.get("/admin/api/conversations.csv")
def admin_conversations_csv():
    require_admin()
    items = STORE.list_summaries(limit=5000)

    output = []
    output.append(["id", "brand", "user_id", "created", "updated", "message_count", "flags", "preview"])
    for it in items:
        output.append(
            [
                str(it.get("id") or ""),
                str(it.get("brand") or ""),
                str(it.get("user_id") or ""),
                str(it.get("created") or ""),
                str(it.get("updated") or ""),
                str(it.get("message_count") or ""),
                ",".join(it.get("flags") or []) if isinstance(it.get("flags"), list) else "",
                str(it.get("preview") or ""),
            ]
        )

    buf = []
    for row in output:
        # basic CSV escaping
        buf.append(",".join(['"' + str(col).replace('"', '""') + '"' for col in row]))

    csv_text = "\n".join(buf) + "\n"
    return Response(
        csv_text,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=conversations.csv"},
    )


@app.get("/admin/api/conversations.jsonl")
def admin_conversations_jsonl():
    require_admin()
    items = STORE.list_summaries(limit=5000)
    lines = [json.dumps(it, ensure_ascii=False) for it in items]
    return Response("\n".join(lines) + "\n", mimetype="application/jsonl; charset=utf-8")




@app.get("/admin/api/conversation/<conv_id>")
def admin_conversation(conv_id: str):
    require_admin()
    obj = STORE.load(conv_id)
    msgs = obj.get("messages") if isinstance(obj, dict) else []
    if not isinstance(msgs, list):
        msgs = []
    return jsonify({"conversation": obj, "messages": msgs, "flags": conversation_flags(obj)})


@app.get("/admin/api/faq/<brand_id>")
def admin_get_faq(brand_id: str):
    require_admin()
    b = brand_or_default(brand_id)
    return jsonify(refresh_kb(b))


@app.put("/admin/api/faq/<brand_id>")
def admin_put_faq(brand_id: str):
    require_admin()
    b = brand_or_default(brand_id)
    payload = request.get_json(silent=True) or {}
    items = payload.get("items")
    if not isinstance(items, list):
        return jsonify({"error": "items_must_be_list"}), 400
    kb_file = str(BRANDS.get(b, {}).get("kb_file") or f"kb_{b}.json")
    obj = save_kb_file(kb_file, b, items)
    KB_BY_BRAND[b] = obj
    return jsonify(obj)


# -----------------------------
# Optional: per-user history
# -----------------------------


@app.get("/user/<user_id>/history")
def user_history(user_id: str):
    require_user_history()
    uid = (user_id or "").strip()
    if not uid:
        return jsonify([])
    summaries = STORE.list_summaries(limit=1000)
    out = [s for s in summaries if str(s.get("user_id") or "") == uid]
    return jsonify(out)


# -----------------------------
# Internal assistant: BT (very small endpoint)
# -----------------------------


@app.post("/bt/chat")
def bt_chat():
    """Internal endpoint protected by BT_API_TOKEN (header: X-BT-Token)."""
    require_bt()
    payload = request.get_json(silent=True) or {}
    user_text = str(payload.get("message") or payload.get("text") or "").strip()
    if not user_text:
        return jsonify({"error": "missing_message"}), 400

    # lazy import to avoid circular
    from bt_service import BTStore, bt_system_prompt, load_bt_profile, sanitize_bt_reply, call_openai_bt

    if OPENAI_CLIENT is None:
        return jsonify({"reply": "Service IA indisponible.", "error": "openai_not_configured"}), 503

    conv_id = str(payload.get("conversation_id") or payload.get("conversationId") or "").strip() or _make_conv_id("bt")

    profile = load_bt_profile(BT_PROFILE_PATH)
    system_prompt = bt_system_prompt(profile)

    store = BTStore(conv_dir=str(CONV_DIR))
    store.prune_old()
    history = store.prompt_messages(conv_id, max_pairs=8)

    # BT calls
    reply, usage = call_openai_bt(
        openai_client=OPENAI_CLIENT,
        model=OPENAI_MODEL,
        reasoning_effort=OPENAI_REASONING_EFFORT,
        temperature=OPENAI_TEMPERATURE,
        max_output_tokens=BT_MAX_OUTPUT_TOKENS,
        system_prompt=system_prompt,
        history=history,
        user_text=user_text,
    )

    reply = sanitize_bt_reply(reply, max_chars=BT_MAX_REPLY_CHARS)
    store.append(conv_id, "user", user_text)
    store.append(conv_id, "assistant", reply, extra={"usage": usage} if usage else None)

    return jsonify({"conversation_id": conv_id, "reply": reply, "usage": usage})


# -----------------------------
# Static helpers
# -----------------------------


@app.get("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory(app.static_folder, filename)


# -----------------------------
# Local run
# -----------------------------


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=True)
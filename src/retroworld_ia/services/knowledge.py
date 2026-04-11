import json
import re
from typing import Any, Dict, List, Optional

from src.data.system_data import SYSTEM_PROMPT
from src.retroworld_ia import config
from src.retroworld_ia.services.logging_store import log_error, now_str


def normalize_brand(value: str) -> str:
    return (value or "").strip().lower()


def load_knowledge_base() -> Dict[str, Any]:
    try:
        return json.loads(config.KNOWLEDGE_PATH.read_text("utf-8"))
    except Exception as err:
        log_error("knowledge_base load failed", err)
        return {"global": {}, "brands": {}}


KNOWLEDGE_BASE = load_knowledge_base()
BRANDS: Dict[str, Dict[str, Any]] = {}
for bid, cfg in (KNOWLEDGE_BASE.get("brands") or {}).items():
    normalized = normalize_brand(bid)
    if normalized:
        item = dict(cfg or {})
        item["id"] = normalized
        BRANDS[normalized] = item

BRAND_ID_DEFAULT = normalize_brand(config._env("BRAND_ID", "retroworld")) if BRANDS else "retroworld"
if BRAND_ID_DEFAULT not in BRANDS and BRANDS:
    BRAND_ID_DEFAULT = next(iter(BRANDS.keys()))
FAQ_ENABLED_BRANDS = config.FAQ_ENABLED_BRANDS_ENV or list(BRANDS.keys())
PUBLIC_BRANDS = config.PUBLIC_BRANDS_ENV or FAQ_ENABLED_BRANDS


def load_public_faq(brand_id: str) -> Dict[str, Any]:
    path = config.STATIC_DIR / f"faq_{brand_id}.json"
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return {"brand": brand_id, "updated": now_str(), "items": []}


def detect_brand_from_text(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    aliases = {
        "runningman": ("runningman",),
        "enigmaniac": ("enigmaniac", "enigma"),
        "retroworld": ("retroworld", "retro world"),
    }
    for bid, words in aliases.items():
        if any(word in lowered for word in words):
            return bid
    return None


def format_contact(brand_id: str) -> str:
    cfg = BRANDS.get(brand_id, {})
    parts = []
    if cfg.get("contact_phone"):
        parts.append(f"telephone: {cfg['contact_phone']}")
    if cfg.get("contact_email"):
        parts.append(f"email: {cfg['contact_email']}")
    if cfg.get("website"):
        parts.append(f"site: {cfg['website']}")
    return " | ".join(parts)


def build_system_prompt(brand_id: str, user_text: str) -> str:
    brand = BRANDS.get(brand_id, BRANDS.get(BRAND_ID_DEFAULT, {}))
    faq = load_public_faq(brand_id)
    global_cfg = KNOWLEDGE_BASE.get("global") or {}

    lines = [SYSTEM_PROMPT.strip(), ""]
    if global_cfg.get("identity"):
        lines.append(global_cfg["identity"])
    rules = global_cfg.get("rules") or []
    if rules:
        lines.append("Regles globales:")
        lines.extend(f"- {rule}" for rule in rules)
    routing = global_cfg.get("routing") or []
    if routing:
        lines.append("")
        lines.append("Regles d'aiguillage:")
        lines.extend(f"- {rule}" for rule in routing)

    lines.append("")
    lines.append(f"Marque principale de la session: {brand.get('name', brand_id)} ({brand_id})")
    if brand.get("contact_phone") or brand.get("contact_email") or brand.get("website"):
        lines.append(f"Contact officiel: {format_contact(brand_id)}")
    if brand.get("summary"):
        lines.append(f"Resume marque: {brand['summary']}")
    features = brand.get("highlights") or []
    if features:
        lines.append("Points forts / activites:")
        lines.extend(f"- {item}" for item in features)
    offers = brand.get("offers") or []
    if offers:
        lines.append("Offres structurees:")
        for offer in offers:
            label = offer.get("name") or "Offre"
            details = [offer.get("category") or "", offer.get("price") or "", offer.get("details") or ""]
            detail_text = " | ".join(part for part in details if part)
            lines.append(f"- {label}: {detail_text}".rstrip(": "))
    knowledge_cards = brand.get("knowledge_cards") or []
    if knowledge_cards:
        lines.append("Reponses guidees:")
        for card in knowledge_cards[:12]:
            lines.append(f"- {card.get('title', 'Sujet')}: {card.get('summary', '')}")
    if brand.get("booking_links"):
        lines.append("Liens de reservation:")
        lines.extend(f"- {item}" for item in brand["booking_links"])

    faq_items = faq.get("items") or []
    if faq_items:
        lines.append("")
        lines.append("FAQ publique:")
        for item in faq_items[:10]:
            question = item.get("question") or ""
            answer = item.get("answer") or ""
            if question and answer:
                lines.append(f"- Q: {question} | R: {answer}")

    other_brand = detect_brand_from_text(user_text)
    if other_brand and other_brand != brand_id:
        lines.append("")
        lines.append("La question mentionne possiblement une autre marque. Repondre en separant bien les etablissements.")

    return "\n".join(line for line in lines if line is not None).strip()


def public_brand_payload(brand_id: str) -> Dict[str, Any]:
    cfg = BRANDS.get(brand_id, {})
    return {
        "id": brand_id,
        "name": cfg.get("name", brand_id),
        "short": cfg.get("short", brand_id),
        "website": cfg.get("website", ""),
        "contact_phone": cfg.get("contact_phone", ""),
        "contact_email": cfg.get("contact_email", ""),
        "summary": cfg.get("summary", ""),
        "highlights": cfg.get("highlights", []),
        "quick_actions": cfg.get("quick_actions", []),
        "knowledge_cards": cfg.get("knowledge_cards", []),
    }


def get_knowledge_editor_payload(brand_id: str) -> Dict[str, Any]:
    cfg = dict(BRANDS.get(brand_id, {}))
    cfg["id"] = brand_id
    return cfg


def save_knowledge_brand(brand_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    full = load_knowledge_base()
    brands = full.setdefault("brands", {})
    incoming = dict(payload or {})
    incoming["id"] = brand_id
    brands[brand_id] = incoming
    config.KNOWLEDGE_PATH.write_text(json.dumps(full, ensure_ascii=False, indent=2), "utf-8")
    KNOWLEDGE_BASE.clear()
    KNOWLEDGE_BASE.update(full)
    BRANDS[brand_id] = incoming
    return incoming


def get_brand_cards(brand_id: str) -> List[Dict[str, Any]]:
    return list((BRANDS.get(brand_id, {}) or {}).get("knowledge_cards", []))


RESERVATION_FORBIDDEN_PATTERNS = [
    r"\b(c['’]?est réservé|réservé|confirmé|confirmée|je vous bloque|on vous bloque|bloqué|bloquée)\b",
]
RESERVATION_INTENT_PATTERNS = [
    r"\b(réserv|reservation|réservation|dispo|disponibilit|créneau|horaire|anniversaire|goûter|acompte)\b",
]


def booking_intent(text: str) -> bool:
    lowered = (text or "").lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in RESERVATION_INTENT_PATTERNS)


def price_intent(text: str) -> bool:
    return bool(re.search(r"\b(prix|tarif|tarifs|combien|co[uû]t|co[uû]te|€|euro)\b", (text or "").lower(), flags=re.I))


def intent_tags(text: str) -> List[str]:
    lowered = (text or "").lower()
    patterns = {
        "reservation": r"\b(réserv|dispo|créneau|horaire|planning)\b",
        "tarif": r"\b(prix|tarif|combien|euro|€)\b",
        "anniversaire": r"\b(anniversaire|gouter|goûter)\b",
        "devis": r"\b(devis|entreprise|groupe|privat)\b",
        "reclamation": r"\b(rembourse|annul|plainte|probleme|bug|panne)\b",
        "contact": r"\b(contact|telephone|mail|email|adresse)\b",
    }
    found = [name for name, pattern in patterns.items() if re.search(pattern, lowered, flags=re.I)]
    return found or ["general"]

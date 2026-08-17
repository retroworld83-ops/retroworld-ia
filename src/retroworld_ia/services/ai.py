import hashlib
import importlib
import importlib.util
import json
import re
import unicodedata
from typing import Any, Dict, List, Tuple

from src.retroworld_ia import config
from src.retroworld_ia.services.logging_store import log_error
from src.retroworld_ia.services.knowledge import booking_intent, price_intent, format_contact

requests = importlib.import_module("requests") if importlib.util.find_spec("requests") else None


def openai_ready() -> bool:
    return bool(config.OPENAI_API_KEY and requests is not None)


def build_openai_messages(system_prompt: str, history: List[Dict[str, Any]], user_message: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    relevant_history = history[-config.CHAT_HISTORY_MESSAGES:] if config.CHAT_HISTORY_MESSAGES else []
    for item in relevant_history:
        role = item.get("role") or "user"
        content = item.get("content") or ""
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": [{"type": "text", "text": content}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": user_message}]})
    return messages


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block.get("text") or "").strip()
        for block in content
        if isinstance(block, dict) and block.get("text")
    ).strip()


def _simple_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    simple_messages: List[Dict[str, str]] = []
    for message in messages:
        role = message.get("role") or "user"
        text = _message_text(message)
        if role in {"system", "user", "assistant"} and text:
            simple_messages.append({"role": role, "content": text})
    return simple_messages


def safety_identifier_for(value: str) -> str:
    digest = hashlib.sha256(f"retroworld-ia:{value or 'anonymous'}".encode("utf-8")).hexdigest()
    return f"rw_{digest[:40]}"


def _supports_reasoning_effort(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith("gpt-5") or bool(re.match(r"^o\d", normalized))


def _supports_text_verbosity(model: str) -> bool:
    return (model or "").strip().lower().startswith("gpt-5")


def _extract_response_text(data: Dict[str, Any]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: List[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "output_text" and block.get("text"):
                parts.append(str(block["text"]).strip())
    return "\n".join(part for part in parts if part).strip()


def responses_answer(messages: List[Dict[str, Any]], safety_identifier: str = "") -> str:
    simple_messages = _simple_messages(messages)
    instructions = "\n\n".join(item["content"] for item in simple_messages if item["role"] == "system")
    input_items = [item for item in simple_messages if item["role"] != "system"]
    if not input_items:
        return ""

    payload: Dict[str, Any] = {
        "model": config.OPENAI_MODEL,
        "instructions": instructions,
        "input": input_items,
        "store": False,
    }
    if config.OPENAI_MAX_OUTPUT_TOKENS:
        payload["max_output_tokens"] = config.OPENAI_MAX_OUTPUT_TOKENS
    if _supports_reasoning_effort(config.OPENAI_MODEL) and config.OPENAI_REASONING_EFFORT in {"none", "low", "medium", "high", "xhigh", "max"}:
        payload["reasoning"] = {"effort": config.OPENAI_REASONING_EFFORT}
    if _supports_text_verbosity(config.OPENAI_MODEL) and config.OPENAI_TEXT_VERBOSITY in {"low", "medium", "high"}:
        payload["text"] = {"verbosity": config.OPENAI_TEXT_VERBOSITY}
    if safety_identifier:
        payload["safety_identifier"] = safety_identifier[:64]

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=35,
        )
        response.raise_for_status()
        return _extract_response_text(response.json() or {})
    except Exception as err:
        response_obj = getattr(err, "response", None)
        response_text = ""
        status_code = None
        if response_obj is not None:
            status_code = getattr(response_obj, "status_code", None)
            try:
                response_text = response_obj.text[:2000]
            except Exception:
                response_text = ""
        log_error("OpenAI responses error", err, {"model": config.OPENAI_MODEL, "status_code": status_code, "response_text": response_text})
        return ""


def openai_answer(messages: List[Dict[str, Any]], safety_identifier: str = "") -> str:
    if not openai_ready():
        return ""
    primary = ""
    if config.OPENAI_API_MODE != "chat_completions":
        primary = responses_answer(messages, safety_identifier=safety_identifier)
    if not primary:
        primary = fallback_chat_completions(messages, primary=True)
    if primary:
        return primary
    return "Desole, je rencontre un souci technique. Pouvez-vous reessayer ou contacter l'equipe ?"


def fallback_chat_completions(messages: List[Dict[str, Any]], primary: bool = False) -> str:
    try:
        simple_messages = _simple_messages(messages)

        payload: Dict[str, Any] = {
            "model": config.OPENAI_MODEL,
            "messages": simple_messages,
        }
        if config.OPENAI_MAX_OUTPUT_TOKENS:
            payload["max_tokens"] = config.OPENAI_MAX_OUTPUT_TOKENS
        if (config.OPENAI_REASONING_EFFORT or "").lower().strip() in {"", "none"}:
            payload["temperature"] = config.OPENAI_TEMPERATURE

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json() or {}
        return (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    except Exception as err:
        response_obj = getattr(err, "response", None)
        response_text = ""
        status_code = None
        if response_obj is not None:
            status_code = getattr(response_obj, "status_code", None)
            try:
                response_text = response_obj.text[:2000]
            except Exception:
                response_text = ""
        log_error("OpenAI chat.completions error" if primary else "OpenAI fallback error", err, {"model": config.OPENAI_MODEL, "status_code": status_code, "response_text": response_text})
        return ""


CURRENCY_AMOUNT_PATTERN = re.compile(
    r"(?<![\w])(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|eur)(?=\s|/|$|[.,;:!?])",
    flags=re.IGNORECASE,
)


def _normalized_amount(value: str) -> str:
    value = (value or "").replace(",", ".")
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else f"{number:.2f}".rstrip("0").rstrip(".")


def enforce_grounded_price_claims(text: str, grounding_text: str, user_text: str = "") -> Tuple[str, bool]:
    if not price_intent(user_text):
        return text or "", False
    allowed = {
        _normalized_amount(match.group(1))
        for match in CURRENCY_AMOUNT_PATTERN.finditer(grounding_text or "")
    }
    altered = False

    def replace_unknown(match: re.Match) -> str:
        nonlocal altered
        if _normalized_amount(match.group(1)) in allowed:
            return match.group(0)
        altered = True
        return "un montant à confirmer par l'équipe"

    safe_text = CURRENCY_AMOUNT_PATTERN.sub(replace_unknown, text or "")
    safe_text = re.sub(r"\bde un montant\b", "d'un montant", safe_text, flags=re.IGNORECASE)
    return safe_text, altered


LIVE_AVAILABILITY_PATTERN = re.compile(
    r"\b((?:les\s+)?(?:salles|créneaux|creneaux|sessions|places)(?:\s+(?!disponibles\b)[\wÀ-ÿ'’-]+){0,3})\s+disponibles\s+(?:sont\s*:|sont|:)",
    flags=re.IGNORECASE,
)


def enforce_no_live_availability_claims(text: str, user_text: str = "") -> Tuple[str, bool]:
    if not booking_intent(user_text):
        return text or "", False

    def replace_claim(match: re.Match) -> str:
        return f"{match.group(1)} figurant dans mes informations sont :"

    safe_text, replacements = LIVE_AVAILABILITY_PATTERN.subn(replace_claim, text or "")
    return safe_text, replacements > 0


RESERVATION_FORBIDDEN_PATTERNS = [
    r"\b(c['’]?est réservé|réservé|confirmé|confirmée|je vous bloque|on vous bloque|bloqué|bloquée)\b",
]


def enforce_no_reservation_promises(text: str) -> Tuple[str, bool]:
    promised = False
    safe_text = text or ""
    lowered = safe_text.lower()
    for pattern in RESERVATION_FORBIDDEN_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            promised = True
            safe_text = re.sub(pattern, "a confirmer par l'equipe (je n'ai pas acces au planning en direct)", safe_text, flags=re.IGNORECASE)
    return safe_text, promised


def _is_non_activity_booking_request(user_text: str) -> bool:
    lowered = (user_text or "").lower()
    non_activity_request = re.search(
        r"\b(table|restaurant|repas|déjeuner|dejeuner|dîner|diner|manger|boire)\b",
        lowered,
        flags=re.I,
    )
    activity_request = re.search(
        r"\b(activité|activite|jeu|vr|escape|quiz|quizz|simulateur|arcade)\b",
        lowered,
        flags=re.I,
    )
    return bool(non_activity_request and not activity_request)


def add_disclaimer_if_needed(answer: str, brand_id: str, user_msg: str) -> str:
    if not booking_intent(user_msg):
        return answer
    if _is_non_activity_booking_request(user_msg):
        return answer
    normalized_answer = "".join(
        character
        for character in unicodedata.normalize("NFKD", answer or "")
        if not unicodedata.combining(character)
    ).lower()
    existing_disclaimers = (
        "je n'ai pas acces au planning",
        "je n'ai pas l'information sur la disponibilite",
        "je n'ai pas l'information sur les disponibilite",
        "je ne peux pas effectuer la reservation",
        "la disponibilite doit etre confirmee",
    )
    if any(signal in normalized_answer for signal in existing_disclaimers):
        return answer
    disclaimer = "Je n'ai pas accès au planning en temps réel, la disponibilité doit être confirmée par l'équipe."
    contact = format_contact(brand_id)
    if contact:
        disclaimer += f" Contact: {contact}"
    if disclaimer.lower() in (answer or "").lower():
        return answer
    return (answer or "").rstrip() + "\n\n" + disclaimer


def retroworld_booking_links_for(user_text: str) -> List[str]:
    lowered = (user_text or "").lower()
    links = []
    if _is_non_activity_booking_request(user_text):
        return links
    if re.search(r"\b(escape|escape\s*vr|escape\s*game)\b", lowered, flags=re.I):
        links.append("https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=escape%20game&lang=fr")
    if re.search(r"\b(quiz|quizz)\b", lowered, flags=re.I):
        links.append("https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=quizz&lang=fr")
    if not links:
        links.append("https://retroworld.qweekle.com/shop/retroworld/multi/jeux-a-la-partie?tag=Jeu%20%C3%A0%20la%20partie&lang=fr")
    return links


def append_retroworld_links_if_missing(user_text: str, reply: str) -> str:
    if "qweekle.com" in (reply or "").lower():
        return reply
    if not (booking_intent(user_text) or price_intent(user_text)):
        return reply
    links = retroworld_booking_links_for(user_text)
    if not links:
        return reply
    block = "\n".join(f"Lien reservation Retroworld: {url}" for url in links)
    return (reply or "").rstrip() + "\n\n" + block


def summarize_transcript(messages: List[Dict[str, Any]]) -> str:
    data = [{"role": m.get("role"), "content": m.get("content", "")[:240]} for m in messages[-8:]]
    return json.dumps(data, ensure_ascii=False)

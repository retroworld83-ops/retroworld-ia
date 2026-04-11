import importlib
import importlib.util
import json
import re
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


def openai_answer(messages: List[Dict[str, Any]]) -> str:
    if not openai_ready():
        return ""
    payload: Dict[str, Any] = {
        "model": config.OPENAI_MODEL,
        "input": messages,
        "max_output_tokens": config.OPENAI_MAX_OUTPUT_TOKENS,
    }
    effort = (config.OPENAI_REASONING_EFFORT or "").lower().strip()
    if effort in {"low", "medium", "high"}:
        payload["reasoning"] = {"effort": effort}
    elif effort in {"", "none"}:
        payload["temperature"] = config.OPENAI_TEMPERATURE

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json() or {}
        texts: List[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))
        return "\n".join(texts).strip()
    except Exception as err:
        log_error("OpenAI error", err, {"model": config.OPENAI_MODEL})
        return "Desole, je rencontre un souci technique. Pouvez-vous reessayer ou contacter l'equipe ?"


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


def add_disclaimer_if_needed(answer: str, brand_id: str, user_msg: str) -> str:
    if not booking_intent(user_msg):
        return answer
    disclaimer = "Je n'ai pas acces au planning en temps reel, la disponibilite doit etre confirmee par l'equipe."
    contact = format_contact(brand_id)
    if contact:
        disclaimer += f" Contact: {contact}"
    if disclaimer.lower() in (answer or "").lower():
        return answer
    return (answer or "").rstrip() + "\n\n" + disclaimer


def retroworld_booking_links_for(user_text: str) -> List[str]:
    lowered = (user_text or "").lower()
    links = []
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
    block = "\n".join(f"Lien reservation Retroworld: {url}" for url in retroworld_booking_links_for(user_text))
    return (reply or "").rstrip() + "\n\n" + block


def summarize_transcript(messages: List[Dict[str, Any]]) -> str:
    data = [{"role": m.get("role"), "content": m.get("content", "")[:240]} for m in messages[-8:]]
    return json.dumps(data, ensure_ascii=False)

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


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
    except Exception:
        return False


def strip_markdown_simple(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"^\s*[-*]\s+", "• ", s, flags=re.M)
    return s


def remove_emojis(text: str) -> str:
    # retire la plupart des emojis (plans unicode hors BMP)
    return re.sub(r"[\U00010000-\U0010FFFF]", "", text)


def _norm(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def load_bt_profile(profile_path: str) -> Dict[str, Any]:
    """Charge bt_profile.yaml (ou JSON) avec fallback sûr."""
    profile_path = (profile_path or "").strip()
    if not profile_path:
        return {}
    if not os.path.exists(profile_path):
        return {}

    # YAML si dispo
    try:
        import yaml  # type: ignore

        with open(profile_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    # fallback JSON
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def bt_system_prompt(profile: Dict[str, Any]) -> str:
    ident = profile.get("identity", {}) if isinstance(profile, dict) else {}
    interaction = profile.get("interaction", {}) if isinstance(profile, dict) else {}
    voice = profile.get("voice", {}) if isinstance(profile, dict) else {}

    name = str(ident.get("name") or "BT")
    role = str(ident.get("role") or "Assistant interne")
    humor = bool(ident.get("humor") is True)
    small_talk = bool(ident.get("small_talk") is True)
    emotions = bool(ident.get("emotions") is True)

    # Contrat strict: même si le YAML est modifié, on garde une base sûre.
    base_rules = [
        f"Tu t'appelles {name}.",
        f"Rôle: {role}.",
        "Langue: français.",
        "Style: sobre, neutre, professionnel.",
        "Réponses courtes, utiles, sans bavardage.",
        "Aucune interaction avec le public. Tu n'adresses que l'équipe.",
        "Pas d'humour, pas de blagues, pas d'emojis.",
        "Pas d'auto-présentation inutile.",
        "Si une demande est ambiguë: poser une seule question courte.",
    ]

    # Si le YAML tentait d'activer humour/small talk, on ignore.
    if humor or small_talk or emotions:
        base_rules.append("Note: humour/small-talk/émotions désactivés.")

    verbosity = str(interaction.get("verbosity") or "minimal")
    if verbosity not in ("minimal", "short", "normal"):
        verbosity = "minimal"
    base_rules.append(f"Verbosity: {verbosity}.")

    tone = str((voice.get("tone") or "calm")).lower()
    if tone not in ("calm", "neutral", "firm"):
        tone = "calm"
    base_rules.append(f"Ton: {tone}.")

    return "\n".join(base_rules)


def sanitize_bt_reply(text: str, max_chars: int = 500, max_lines: int = 6) -> str:
    t = strip_markdown_simple(text)
    t = remove_emojis(t)
    t = re.sub(r"\s+", " ", t).strip()

    # coupe en lignes si besoin
    if "\n" in t:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        t = "\n".join(lines[:max_lines]).strip()

    if len(t) > max_chars:
        t = t[: max_chars - 1].rstrip() + "…"

    # garde-fou: pas de phrases trop "assistantes".
    lowered = _norm(t)
    taboo = ["en tant qu'ia", "je suis une ia", "je suis un assistant", "comme modèle", "chatgpt"]
    if any(k in lowered for k in taboo):
        t = "Compris."

    return t


@dataclass
class BTStore:
    conv_dir: str
    ttl_days: int = 30

    def __post_init__(self) -> None:
        os.makedirs(self.conv_dir, exist_ok=True)

    def _path(self, conversation_id: str) -> str:
        cid = re.sub(r"[^a-zA-Z0-9_\-]", "", conversation_id or "")
        if not cid:
            cid = "bt_" + datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return os.path.join(self.conv_dir, f"{cid}.json")

    def load(self, conversation_id: str) -> Dict[str, Any]:
        path = self._path(conversation_id)
        data = _safe_read_json(path, {"id": conversation_id, "created": _utc_iso(), "messages": []})
        if not isinstance(data, dict):
            return {"id": conversation_id, "created": _utc_iso(), "messages": []}
        if not isinstance(data.get("messages"), list):
            data["messages"] = []
        data.setdefault("id", conversation_id)
        data.setdefault("created", _utc_iso())
        return data

    def save(self, conversation_id: str, obj: Dict[str, Any]) -> None:
        path = self._path(conversation_id)
        _safe_write_json(path, obj)

    def append(self, conversation_id: str, role: str, content: str, extra: Optional[Dict[str, Any]] = None) -> None:
        obj = self.load(conversation_id)
        msgs = obj.get("messages")
        if not isinstance(msgs, list):
            msgs = []
        item: Dict[str, Any] = {"role": role, "content": str(content or ""), "ts": _utc_iso()}
        if extra and isinstance(extra, dict):
            item["extra"] = extra
        msgs.append(item)
        obj["messages"] = msgs
        self.save(conversation_id, obj)

    def prompt_messages(self, conversation_id: str, max_pairs: int = 8) -> List[Dict[str, str]]:
        obj = self.load(conversation_id)
        msgs = obj.get("messages")
        if not isinstance(msgs, list):
            return []
        clipped = [m for m in msgs if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        clipped = clipped[-(max_pairs * 2):]
        return [{"role": str(m.get("role")), "content": str(m.get("content") or "")} for m in clipped]

    def prune_old(self) -> None:
        # prune opportuniste
        try:
            now = time.time()
            max_age = self.ttl_days * 86400
            for fn in os.listdir(self.conv_dir):
                if not fn.endswith(".json"):
                    continue
                full = os.path.join(self.conv_dir, fn)
                try:
                    st = os.stat(full)
                    if (now - st.st_mtime) > max_age:
                        os.remove(full)
                except Exception:
                    continue
        except Exception:
            return


def call_openai_bt(
    openai_client: Any,
    model: str,
    reasoning_effort: str,
    temperature: float,
    max_output_tokens: int,
    system_prompt: str,
    history: List[Dict[str, str]],
    user_text: str,
) -> Tuple[str, Dict[str, Any]]:
    if not openai_client:
        return "Service IA indisponible.", {"error": "openai_not_ready"}

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": str(user_text or "")})

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": messages,
        "max_output_tokens": int(max_output_tokens),
        "store": False,
    }

    eff = (reasoning_effort or "none").strip().lower()
    if eff in ("low", "medium", "high"):
        kwargs["reasoning"] = {"effort": eff}
    else:
        kwargs["temperature"] = float(temperature)

    resp = openai_client.responses.create(**kwargs)
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

from __future__ import annotations

from typing import Any, Dict


class RuntimeMetrics:
    def __init__(self) -> None:
        self.openai_calls_total = 0
        self.openai_errors_total = 0
        self.openai_latency_ms_sum = 0
        self.openai_latency_ms_max = 0
        self.chat_requests_total = 0
        self.chat_rate_limited_total = 0

    def observe_openai(self, latency_ms: int, ok: bool) -> None:
        self.openai_calls_total += 1
        if not ok:
            self.openai_errors_total += 1
        self.openai_latency_ms_sum += max(0, int(latency_ms))
        self.openai_latency_ms_max = max(self.openai_latency_ms_max, int(latency_ms))

    def observe_chat_request(self, rate_limited: bool = False) -> None:
        self.chat_requests_total += 1
        if rate_limited:
            self.chat_rate_limited_total += 1

    def snapshot(self) -> Dict[str, Any]:
        avg = 0
        if self.openai_calls_total:
            avg = round(self.openai_latency_ms_sum / self.openai_calls_total, 2)
        return {
            "chat_requests_total": self.chat_requests_total,
            "chat_rate_limited_total": self.chat_rate_limited_total,
            "openai_calls_total": self.openai_calls_total,
            "openai_errors_total": self.openai_errors_total,
            "openai_latency_ms_avg": avg,
            "openai_latency_ms_max": self.openai_latency_ms_max,
        }

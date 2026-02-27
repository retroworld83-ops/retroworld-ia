from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple


class InMemoryRateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> Tuple[bool, int]:
        now = time.time()
        q = self._hits[key]
        threshold = now - self.window_seconds
        while q and q[0] < threshold:
            q.popleft()
        if len(q) >= self.limit:
            retry_after = int(max(1, self.window_seconds - (now - q[0])))
            return False, retry_after
        q.append(now)
        return True, 0

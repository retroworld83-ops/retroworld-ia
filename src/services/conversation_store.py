from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class ConversationStore:
    def __init__(self, backend: str, conv_dir: Path, sqlite_path: Path):
        self.backend = (backend or "json").strip().lower()
        self.conv_dir = conv_dir
        self.sqlite_path = sqlite_path
        self.conv_dir.mkdir(parents=True, exist_ok=True)
        if self.backend == "sqlite":
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created TEXT NOT NULL,
                    updated TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _empty(self, conv_id: str) -> Dict[str, Any]:
        return {
            "id": conv_id,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [],
            "meta": {},
        }

    def _json_path(self, conv_id: str) -> Path:
        return self.conv_dir / f"{conv_id}.json"

    def load(self, conv_id: str) -> Dict[str, Any]:
        if self.backend == "sqlite":
            with sqlite3.connect(self.sqlite_path) as conn:
                row = conn.execute("SELECT payload FROM conversations WHERE id=?", (conv_id,)).fetchone()
            if not row:
                return self._empty(conv_id)
            try:
                return json.loads(row[0])
            except Exception:
                return self._empty(conv_id)

        p = self._json_path(conv_id)
        if not p.exists():
            return self._empty(conv_id)
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return self._empty(conv_id)

    def save(self, conv: Dict[str, Any]) -> None:
        cid = str(conv.get("id", "unknown"))
        if self.backend == "sqlite":
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            created = str(conv.get("created") or now)
            payload = json.dumps(conv, ensure_ascii=False)
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    """
                    INSERT INTO conversations(id, created, updated, payload)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      updated=excluded.updated,
                      payload=excluded.payload
                    """,
                    (cid, created, now, payload),
                )
                conn.commit()
            return

        p = self._json_path(cid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(conv, ensure_ascii=False, indent=2), "utf-8")

    def list_all(self) -> List[Dict[str, Any]]:
        if self.backend == "sqlite":
            with sqlite3.connect(self.sqlite_path) as conn:
                rows = conn.execute("SELECT payload FROM conversations ORDER BY updated DESC").fetchall()
            out: List[Dict[str, Any]] = []
            for (payload,) in rows:
                try:
                    out.append(json.loads(payload))
                except Exception:
                    continue
            return out

        out = []
        for p in sorted(self.conv_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                out.append(json.loads(p.read_text("utf-8")))
            except Exception:
                continue
        return out

    def count(self) -> int:
        if self.backend == "sqlite":
            with sqlite3.connect(self.sqlite_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
            return int(row[0] if row else 0)
        return len(list(self.conv_dir.glob("*.json")))

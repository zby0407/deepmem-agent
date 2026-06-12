"""Memory history tracking — adapted from mem0/memory/storage.py.

Records every ADD/UPDATE/DELETE operation on memories for audit trail.
Uses the same SQLite database as LocalExternalMemoryProvider.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class MemoryHistoryManager:
    """Tracks memory change history (ADD/UPDATE/DELETE) in SQLite.

    Adapted from mem0's SQLiteManager — simplified to only the history table
    (no messages table, no thread lock — our voice runtime is single-threaded async).
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS voice_memory_history (
                id           TEXT PRIMARY KEY,
                memory_id    TEXT,
                old_memory   TEXT,
                new_memory   TEXT,
                event        TEXT,
                created_at   REAL,
                updated_at   REAL,
                is_deleted   INTEGER DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vmh_memory_id
            ON voice_memory_history(memory_id)
        """)
        self._conn.commit()

    def add_history(
        self,
        memory_id: str,
        old_memory: str | None,
        new_memory: str | None,
        event: str,
        *,
        created_at: float | None = None,
        updated_at: float | None = None,
        is_deleted: int = 0,
    ) -> None:
        """Record a memory change event.

        Args:
            memory_id: The memory's ID.
            old_memory: Previous text (None for ADD).
            new_memory: New text (None for DELETE).
            event: "ADD", "UPDATE", or "DELETE".
            created_at: Original creation timestamp (epoch).
            updated_at: Timestamp of this change (epoch).
            is_deleted: 1 if this is a DELETE event.
        """
        try:
            self._conn.execute(
                """
                INSERT INTO voice_memory_history (
                    id, memory_id, old_memory, new_memory, event,
                    created_at, updated_at, is_deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    str(memory_id),
                    old_memory,
                    new_memory,
                    event,
                    created_at,
                    updated_at,
                    is_deleted,
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.debug("[MemoryHistory] add_history failed: %s", e)

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """Get the change history for a specific memory.

        Returns:
            List of history records ordered by time ascending.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted
                FROM voice_memory_history
                WHERE memory_id = ?
                ORDER BY created_at ASC, updated_at ASC
                """,
                (str(memory_id),),
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "memory_id": r[1],
                    "old_memory": r[2] or "",
                    "new_memory": r[3] or "",
                    "event": r[4],
                    "created_at": r[5],
                    "updated_at": r[6],
                    "is_deleted": bool(r[7]),
                }
                for r in rows
            ]
        except Exception:
            return []

    def get_recent_history(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent history records across all memories for a user.

        Joins with voice_memories to filter by user_id.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT h.id, h.memory_id, h.old_memory, h.new_memory, h.event,
                       h.created_at, h.updated_at, h.is_deleted
                FROM voice_memory_history h
                JOIN voice_memories m ON m.id = CAST(h.memory_id AS INTEGER)
                WHERE m.user_id = ?
                ORDER BY h.updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "memory_id": r[1],
                    "old_memory": r[2] or "",
                    "new_memory": r[3] or "",
                    "event": r[4],
                    "created_at": r[5],
                    "updated_at": r[6],
                    "is_deleted": bool(r[7]),
                }
                for r in rows
            ]
        except Exception:
            return []

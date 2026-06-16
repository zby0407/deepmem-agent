"""LocalExternalMemoryProvider — mem0-style self-contained memory for the voice runtime.

Features:
- Automatic LLM-based fact extraction from conversation turns
- MD5 hash deduplication + LLM conflict resolution
- Semantic search via DashScope text-embedding-v3 embeddings
- Hybrid search: semantic cosine similarity + FTS5 keyword matching
- Temporal grounding: all facts stored with absolute dates
- Bi-temporal model (Zep/Graphiti): valid_at/invalid_at/expired_at
- 2-factor retention decay (novel): age × confirmation boost
- Contradiction detection (Zep pattern): invalidation over deletion
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import struct
import time
from datetime import date
from pathlib import Path
from typing import Any

from .memory_extract import _md5, extract_and_dedup, semantic_dedup
from .memory_history import MemoryHistoryManager
from .memory_provider import MemoryProvider
from .scoring import ENTITY_BOOST_WEIGHT, get_bm25_params, normalize_bm25, score_and_rank
from .config import cfg

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "voice_memory.sqlite3"


def _encode_embedding(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_embedding(data: bytes, dim: int = 1024) -> list[float]:
    return list(struct.unpack(f"{dim}f", data))


class LocalExternalMemoryProvider(MemoryProvider):
    """Self-contained local memory with SQLite + embeddings.

    - Automatic fact extraction via LLM after each turn (mem0-style)
    - MD5 hash dedup + LLM conflict resolution
    - Semantic search via embeddings + FTS5 keyword search
    - Frozen snapshot: top semantic memories baked into system prompt
    - Cognitive science-based 3-scope system (semantic/episodic/procedural)
    - Tools: memory_search (hybrid search)
    """

    def __init__(
        self,
        user_id: str = "",
        db_path: str | Path = DB_PATH,
        llm_client: Any = None,
        model: str | None = None,
        embedding_client: Any = None,
        skip_embedding: bool = False,
    ):
        self._user_id = user_id
        self._session_id = ""
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._cached_context: str = ""
        self._background_task: asyncio.Task | None = None
        # Frozen snapshot
        self._frozen_identity_lines: list[str] = []
        self._frozen_system_built: bool = False
        # LLM + embedding
        self._llm = llm_client
        self._model = model or cfg("model.llm", "qwen-flash")
        self._embedder = embedding_client
        # Recent extraction buffer for dedup within session
        self._recent_hashes: set[str] = set()
        self._recent_texts: list[str] = []  # for LLM dedup prompt
        self._session_history: list[dict[str, str]] = []  # last_k_messages
        self._skip_embedding = skip_embedding

    @property
    def name(self) -> str:
        return "local"

    @property
    def memory_layer(self) -> str:
        return "external"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        if "user_id" in kwargs:
            self._user_id = kwargs["user_id"]
        self._ensure_db()
        if not self._frozen_system_built:
            self._build_frozen_snapshot()

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS voice_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT 'episodic',
                created_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                hash TEXT,
                embedding BLOB
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_voice_mem_user
            ON voice_memories(user_id, scope)
        """)
        # FTS5 index for search
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS voice_memories_fts
            USING fts5(content, scope, user_id, content='voice_memories', content_rowid='id')
        """)
        # Triggers to keep FTS in sync
        self._conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS voice_mem_ai AFTER INSERT ON voice_memories BEGIN
                INSERT INTO voice_memories_fts(rowid, content, scope, user_id)
                VALUES (new.id, new.content, new.scope, new.user_id);
            END;
            CREATE TRIGGER IF NOT EXISTS voice_mem_ad AFTER DELETE ON voice_memories BEGIN
                INSERT INTO voice_memories_fts(voice_memories_fts, rowid, content, scope, user_id)
                VALUES ('delete', old.id, old.content, old.scope, old.user_id);
            END;
            CREATE TRIGGER IF NOT EXISTS voice_mem_au AFTER UPDATE ON voice_memories BEGIN
                INSERT INTO voice_memories_fts(voice_memories_fts, rowid, content, scope, user_id)
                VALUES ('delete', old.id, old.content, old.scope, old.user_id);
                INSERT INTO voice_memories_fts(rowid, content, scope, user_id)
                VALUES (new.id, new.content, new.scope, new.user_id);
            END;
        """)
        self._conn.commit()
        # Schema migration: add hash/embedding columns if missing
        self._migrate_schema()
        # Initialize history manager
        self._history = MemoryHistoryManager(self._conn)
        # Backfill embeddings for existing memories
        self._backfill_embeddings()

    def _migrate_schema(self) -> None:
        """Add new columns to existing tables.

        Bi-temporal model (Zep/Graphiti EntityEdge pattern):
        - valid_at: when the fact became true in the real world (event time)
        - invalid_at: when the fact stopped being true (NULL = currently active)
        - expired_at: when the record was superseded in the DB (graph-maintenance time)

        Confidence + decay (novel — beyond Zep/Mem0):
        - confidence: extraction confidence score (0-1)
        - confirmation_count: how many times this fact was reinforced
        - last_used_at: last access timestamp for idle decay
        - decay_k: per-memory decay rate (scope-dependent)
        """
        try:
            cols = {row[1] for row in self._conn.execute("PRAGMA table_info(voice_memories)").fetchall()}
            if "hash" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN hash TEXT")
            if "embedding" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN embedding BLOB")
            if "linked_memory_ids" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN linked_memory_ids TEXT")
            if "updated_at" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN updated_at REAL")
            if "attributed_to" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN attributed_to TEXT")
            # Bi-temporal fields (Zep/Graphiti pattern)
            if "confidence" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN confidence REAL DEFAULT 0.7")
            if "confirmation_count" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN confirmation_count INTEGER DEFAULT 0")
            if "last_used_at" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN last_used_at REAL")
            if "decay_k" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN decay_k REAL DEFAULT 0.02")
            if "valid_at" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN valid_at REAL")
            if "invalid_at" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN invalid_at REAL")
            if "expired_at" not in cols:
                self._conn.execute("ALTER TABLE voice_memories ADD COLUMN expired_at REAL")
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_voice_mem_hash
                ON voice_memories(user_id, hash)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_voice_mem_active
                ON voice_memories(user_id, invalid_at)
            """)
            self._conn.commit()
        except Exception:
            pass

    def _backfill_embeddings(self) -> None:
        """Embed existing memories that lack embeddings."""
        if not self._embedder or not self._conn:
            return
        try:
            rows = self._conn.execute(
                "SELECT id, content FROM voice_memories "
                "WHERE user_id = ? AND embedding IS NULL AND scope != 'conversation' "
                "ORDER BY created_at DESC LIMIT 50",
                (self._user_id,),
            ).fetchall()
            if not rows:
                return
            count = 0
            for row in rows:
                try:
                    vec = self._embedder.embed_single(row["content"])
                    if vec and not all(v == 0 for v in vec):
                        blob = _encode_embedding(vec)
                        self._conn.execute(
                            "UPDATE voice_memories SET embedding = ? WHERE id = ?",
                            (blob, row["id"]),
                        )
                        count += 1
                except Exception:
                    continue
            if count:
                self._conn.commit()
                logger.info("[MemoryExtract] Backfilled %d embeddings", count)
        except Exception as e:
            logger.debug("[MemoryExtract] backfill failed: %s", e)

    # ------------------------------------------------------------------
    # System prompt / frozen snapshot
    # ------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        parts = []
        if self._frozen_system_built and self._frozen_identity_lines:
            frozen_block = "[Session Context - stable across turns]"
            lines = "\n".join(f"  - {line}" for line in self._frozen_identity_lines)
            frozen_block += f"\nKnown identity:\n{lines}"
            parts.append(frozen_block)

        parts.append(
            "# Persistent Memory\n"
            "Your memory system automatically extracts and recalls relevant facts. "
            "The <memory-context> block contains relevant memories for the current query."
        )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Prefetch — hybrid semantic + keyword search
    # ------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._user_id or not self._conn:
            return self._cached_context
        try:
            if not self._frozen_system_built:
                self._build_frozen_snapshot()

            results = self._hybrid_search(query, limit=cfg("retrieval.prefetch_limit", 5))

            # Always-inject rules (hindsight) — forced into every turn
            always_rules = self._get_always_inject_rules(limit=cfg("retrieval.always_inject_hindsight_limit", 3))
            if always_rules:
                seen_ids = {r.get("id") for r in results}
                for rule in always_rules:
                    if rule["id"] not in seen_ids:
                        results.append(rule)

            if results:
                self._cached_context = "\n".join(r["content"] for r in results)
            else:
                # Fallback: recent non-conversation memories
                self._cached_context = self._recent_memory_pack(limit=5)
        except Exception as e:
            logger.debug("LocalExternalMemoryProvider prefetch failed: %s", e)
        return self._cached_context

    def _get_always_inject_rules(self, limit: int = 3) -> list[dict]:
        """Fetch hindsight rules that should be injected every turn.

        These are procedural memories with metadata.kind='hindsight_rule',
        produced by the consolidator's deduction phase.
        """
        try:
            rows = self._conn.execute(
                "SELECT id, content FROM voice_memories "
                "WHERE user_id = ? AND scope = 'procedural' "
                "AND invalid_at IS NULL "
                "AND json_extract(metadata, '$.kind') = 'hindsight_rule' "
                "ORDER BY created_at DESC LIMIT ?",
                (self._user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass  # prefetch is synchronous

    def _build_frozen_snapshot(self) -> None:
        try:
            rows = self._conn.execute(
                "SELECT content FROM voice_memories "
                "WHERE user_id = ? AND scope = 'semantic' AND invalid_at IS NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (self._user_id, cfg("retrieval.frozen_snapshot_limit", 5)),
            ).fetchall()
            self._frozen_identity_lines = [r["content"] for r in rows]
            self._frozen_system_built = True
            logger.info(
                "LocalExternalMemoryProvider frozen snapshot: %d semantic lines",
                len(self._frozen_identity_lines),
            )
        except Exception as e:
            logger.debug("LocalExternalMemoryProvider frozen snapshot failed: %s", e)

    def _recent_memory_pack(self, limit: int = 5) -> str:
        try:
            rows = self._conn.execute(
                "SELECT content FROM voice_memories "
                "WHERE user_id = ? AND scope != 'conversation' AND invalid_at IS NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (self._user_id, limit),
            ).fetchall()
            return "\n".join(r["content"] for r in rows) if rows else ""
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Hybrid search: semantic + FTS5
    # ------------------------------------------------------------------

    def _hybrid_search(self, query: str, limit: int = 5) -> list[dict]:
        """Hybrid search: FTS5 pre-filter → mem0 scoring × retention decay.

        Optimized pipeline:
        1. FTS5 keyword search → candidate_ids (pre-filter)
        2. Semantic search only on candidates (not full scan)
        3. BM25 sigmoid normalization (mem0 scoring.py)
        4. Linked-memory boost (mem0 entity_boost)
        5. Score and rank (mem0 score_and_rank)
        6. Apply retention decay (simplified 2-factor: age + confirmation)
        """
        fetch_limit = max(limit * cfg("retrieval.fts5_fetch_multiplier", 4), cfg("retrieval.fts5_fetch_min", 20))

        # Step 1: FTS5 search → results + candidate_ids for semantic pre-filter
        fts_results, candidate_ids = self._search_db(query, limit=fetch_limit)

        # Step 2: Semantic search on pre-filtered candidates (not full scan)
        semantic_results = self._semantic_search(
            query, limit=fetch_limit, candidate_ids=candidate_ids,
        )

        # Step 3: BM25 sigmoid normalization
        bm25_scores: dict[str, float] = {}
        if fts_results:
            midpoint, steepness = get_bm25_params(query)
            for r in fts_results:
                raw_rank = r.get("rank", 0)
                if raw_rank and raw_rank < 0:
                    bm25_scores[str(r["id"])] = normalize_bm25(-raw_rank, midpoint, steepness)

        # Step 4: Linked-memory boosts
        entity_boosts = self._compute_link_boosts(semantic_results, fts_results)

        # Step 5: Score and rank using mem0's formula
        results = score_and_rank(semantic_results, bm25_scores, entity_boosts, threshold=cfg("retrieval.score_threshold", 0.15), top_k=limit)

        # Step 6: Apply retention decay (simplified 2-factor)
        # retention = e^(-k*age) * min(1.8, 1+ln(1+confirmations)*0.08)
        # No last_used_at tracking — not worth the write overhead per search
        now = time.time()
        for r in results:
            # Retention fields already in result from semantic/fts search
            created_at = r.get("created_at") or now
            k = r.get("decay_k") or 0.02
            confirmations = r.get("confirmation_count") or 0
            age_days = max(0.0, (now - created_at) / 86400)
            age_decay = math.exp(-k * age_days)
            access_boost = min(
                cfg("retrieval.retention_access_boost_cap", 1.8),
                1.0 + math.log1p(confirmations) * cfg("retrieval.retention_confirmation_factor", 0.08),
            )
            retention = age_decay * access_boost
            r["retention_score"] = round(retention, 4)
            confidence = r.get("confidence") or 0.7
            r["score"] = round(r.get("score", 0) * retention * confidence, 4)

        # Sort by final score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:limit]

    def _compute_link_boosts(self, semantic_results: list[dict], fts_results: list[dict]) -> dict[str, float]:
        """Boost memories linked to search results via linked_memory_ids."""
        boosts: dict[str, float] = {}
        all_results = semantic_results + fts_results
        for r in all_results:
            links = r.get("linked_memory_ids")
            if not links:
                # Try loading from DB if not in result
                rid = r.get("id")
                if rid:
                    try:
                        row = self._conn.execute(
                            "SELECT linked_memory_ids FROM voice_memories WHERE id = ?",
                            (rid,),
                        ).fetchone()
                        if row and row["linked_memory_ids"]:
                            links = json.loads(row["linked_memory_ids"])
                    except Exception:
                        pass
            if links:
                for linked_id in links:
                    boosts[str(linked_id)] = boosts.get(str(linked_id), 0) + ENTITY_BOOST_WEIGHT * cfg("retrieval.linked_boost_factor", 0.5)
        return boosts

    def _semantic_search(self, query: str, limit: int = 5,
                         candidate_ids: list[int] | None = None) -> list[dict]:
        """Embed query and find most similar memories by cosine similarity.

        If candidate_ids is provided, only compute embeddings for those candidates
        (FTS5 pre-filter optimization). Otherwise falls back to full scan.
        """
        if not self._embedder:
            return []
        try:
            query_vec = self._embedder.embed_single(query)
            if not query_vec or all(v == 0 for v in query_vec):
                return []

            if candidate_ids:
                # Pre-filtered: only compute embeddings for FTS5 candidates
                placeholders = ",".join("?" for _ in candidate_ids)
                rows = self._conn.execute(
                    f"SELECT id, content, scope, embedding, linked_memory_ids, "
                    f"confidence, confirmation_count, created_at, decay_k "
                    f"FROM voice_memories "
                    f"WHERE id IN ({placeholders}) AND embedding IS NOT NULL "
                    f"AND invalid_at IS NULL",
                    candidate_ids,
                ).fetchall()
            else:
                # Fallback: full scan (for small memory sets)
                rows = self._conn.execute(
                    "SELECT id, content, scope, embedding, linked_memory_ids, "
                    "confidence, confirmation_count, created_at, decay_k "
                    "FROM voice_memories "
                    "WHERE user_id = ? AND embedding IS NOT NULL AND scope != 'conversation' "
                    "AND invalid_at IS NULL",
                    (self._user_id,),
                ).fetchall()

            scored = []
            for row in rows:
                try:
                    mem_vec = _decode_embedding(row["embedding"])
                    score = _cosine_similarity(query_vec, mem_vec)
                    if score >= cfg("retrieval.semantic_floor", 0.15):
                        links = None
                        if row["linked_memory_ids"]:
                            try:
                                links = json.loads(row["linked_memory_ids"])
                            except Exception:
                                pass
                        scored.append({
                            "id": row["id"], "content": row["content"],
                            "scope": row["scope"], "score": score,
                            "linked_memory_ids": links,
                            "confidence": row["confidence"] or 0.7,
                            "confirmation_count": row["confirmation_count"] or 0,
                            "created_at": row["created_at"],
                            "decay_k": row["decay_k"] or 0.02,
                        })
                except Exception:
                    continue

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:limit]
        except Exception as e:
            logger.debug("Semantic search failed: %s", e)
            return []

    def _search_db(self, query: str, limit: int = 5) -> tuple[list[dict], list[int]]:
        """FTS5 keyword search, filtering out invalidated memories (Zep pattern).

        Returns (results, candidate_ids) — candidate_ids for semantic search pre-filter.
        """
        if not self._conn:
            return [], []
        try:
            rows = self._conn.execute(
                "SELECT m.id, m.content, m.scope, fts.rank, "
                "m.confidence, m.confirmation_count, m.created_at, m.decay_k "
                "FROM voice_memories_fts fts "
                "JOIN voice_memories m ON m.id = fts.rowid "
                "WHERE fts.user_id = ? AND voice_memories_fts MATCH ? "
                "AND m.invalid_at IS NULL "
                "ORDER BY rank LIMIT ?",
                (self._user_id, query, limit),
            ).fetchall()
            if rows:
                results = [dict(r) for r in rows]
                candidate_ids = [r["id"] for r in results]
                return results, candidate_ids
        except Exception:
            pass
        try:
            rows = self._conn.execute(
                "SELECT id, content, scope, confidence, confirmation_count, "
                "created_at, decay_k FROM voice_memories "
                "WHERE user_id = ? AND content LIKE ? AND scope != 'conversation' "
                "AND invalid_at IS NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (self._user_id, f"%{query}%", limit),
            ).fetchall()
            results = [dict(r) for r in rows]
            candidate_ids = [r["id"] for r in results]
            return results, candidate_ids
        except Exception:
            return [], []

    # ------------------------------------------------------------------
    # Sync turn — automatic fact extraction (mem0-style)
    # ------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Extract facts from conversation turn and store them (mem0 V3 pipeline).

        Single LLM call: extract + dedup in one pass.
        Semantic dedup via embeddings to catch near-duplicates.
        """
        if not self._user_id or not self._conn:
            return
        if not user_content and not assistant_content:
            return

        # Track session history for context
        self._session_history.append({"role": "user", "content": user_content or ""})
        self._session_history.append({"role": "assistant", "content": assistant_content or ""})
        # Keep last 20 messages
        if len(self._session_history) > cfg("buffers.session_history", 20):
            self._session_history = self._session_history[-cfg("buffers.session_history", 20):]

        # Fire-and-forget async extraction
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._extract_and_store(user_content, assistant_content))
        except RuntimeError:
            pass

    async def _extract_and_store(self, user_content: str, assistant_content: str) -> None:
        """V3 extraction pipeline — single LLM call with semantic dedup."""
        try:
            today = date.today().strftime("%Y-%m-%d")

            # Get existing memories for prompt context + hash dedup
            existing = self._get_existing_for_dedup(limit=cfg("buffers.existing_dedup", 20))
            existing_hashes = {m["hash"] for m in existing if m.get("hash")}
            existing_for_prompt = [
                {"id": str(m["id"]), "text": m["content"]}
                for m in existing[:cfg("buffers.dedup_prompt_existing_cap", 10)]
            ]

            # Build existing embedding map for semantic dedup
            existing_embeddings: dict[str, list[float]] = {}
            if self._embedder:
                for m in existing[:cfg("buffers.existing_dedup", 20)]:
                    if m.get("embedding"):
                        try:
                            existing_embeddings[str(m["id"])] = _decode_embedding(m["embedding"])
                        except Exception:
                            pass

            # Phase 1: Single LLM call — extract facts with built-in dedup
            extracted = await extract_and_dedup(
                self._llm, self._model,
                user_content, assistant_content,
                existing_for_prompt,
                recently_extracted=self._recent_texts[-20:],
                last_k_messages=self._session_history[-10:],
                observation_date=today,
            )

            if not extracted:
                logger.debug("[MemoryExtract] No facts extracted from turn")
                return

            # Phase 2: Hash dedup (exact match)
            new_texts = []
            new_hashes = set()
            for mem in extracted:
                h = _md5(mem["text"])
                if h in existing_hashes or h in self._recent_hashes:
                    continue
                new_texts.append(mem["text"])
                new_hashes.add(h)

            if not new_texts:
                logger.debug("[MemoryExtract] All extracted facts are duplicates (hash)")
                return

            # Phase 3: Semantic dedup (embedding similarity >= 0.90)
            if existing_embeddings:
                new_texts = semantic_dedup(
                    new_texts, existing_embeddings, self._embedder,
                )
                # Rebuild hashes to match surviving texts
                new_hashes = {_md5(t) for t in new_texts}

            if not new_texts:
                logger.debug("[MemoryExtract] All extracted facts are duplicates (semantic)")
                return

            # Phase 4: Store new facts with scope from extraction + resolve contradictions
            # Uses linked_memory_ids from extraction (no extra LLM call)
            added = 0
            for mem in extracted:
                text = mem["text"]
                h = _md5(text)
                if h not in new_hashes:
                    continue
                scope = mem.get("scope", "episodic")
                mem_id = self._store_memory(
                    text, scope,
                    linked_memory_ids=mem.get("linked_memory_ids", []),
                    attributed_to=mem.get("attributed_to", "user"),
                    confidence=mem.get("confidence", 0.7),
                )
                if mem_id:
                    self._recent_hashes.add(h)
                    self._recent_texts.append(text)
                    added += 1
                    # Deterministic contradiction resolution using linked_memory_ids
                    # No extra LLM call — the extraction prompt already identified
                    # which existing memories are related via linked_memory_ids
                    self._resolve_linked_contradictions(
                        mem_id, text, mem.get("linked_memory_ids", []),
                    )

            # Keep recent_texts bounded
            if len(self._recent_texts) > cfg("buffers.recent_texts", 50):
                self._recent_texts = self._recent_texts[-cfg("buffers.recent_texts", 50):]

            logger.info("[MemoryExtract] Turn processed: %d added", added)

            if added:
                self._frozen_system_built = False

        except Exception as e:
            logger.warning("[MemoryExtract] Extraction pipeline failed: %s", e)

    # ------------------------------------------------------------------
    # Contradiction resolution — deterministic, no extra LLM call
    # ------------------------------------------------------------------

    def _resolve_linked_contradictions(
        self, new_mem_id: int, new_text: str, linked_ids: list,
    ) -> None:
        """Resolve contradictions using linked_memory_ids from extraction.

        The extraction prompt already identifies which existing memories are
        related via linked_memory_ids. We use this to deterministically:
        - Supersede old memories that are contradicted by the new fact
        - Reinforce old memories that are confirmed by the new fact

        No extra LLM call — this is pure text comparison on the links
        the extraction LLM already provided (Zep invalidation pattern).
        """
        if not linked_ids or not self._conn:
            return
        now = time.time()
        new_normalized = self._normalize_for_compare(new_text)

        for linked_id in linked_ids:
            try:
                linked_id_int = int(linked_id)
            except (ValueError, TypeError):
                continue
            row = self._conn.execute(
                "SELECT id, content, created_at FROM voice_memories "
                "WHERE id = ? AND invalid_at IS NULL",
                (linked_id_int,),
            ).fetchone()
            if not row:
                continue

            old_text = row["content"]
            old_normalized = self._normalize_for_compare(old_text)

            # Compute text overlap to decide reinforce vs supersede
            overlap = self._text_overlap(new_normalized, old_normalized)

            if overlap >= cfg("retrieval.contradiction_reinforce_overlap", 0.7):
                # High overlap = same fact, reinforce (Zep episode count pattern)
                self._conn.execute(
                    "UPDATE voice_memories SET confirmation_count = confirmation_count + 1 WHERE id = ?",
                    (linked_id_int,),
                )
                self._history.add_history(
                    linked_id_int, old_text, old_text, "REINFORCE",
                    created_at=row["created_at"], updated_at=now,
                )
                logger.info("[Resolve] id=%d reinforced (overlap=%.2f)", linked_id_int, overlap)
            elif overlap < cfg("retrieval.contradiction_supersede_overlap", 0.3):
                # Low overlap = different fact about same topic = contradiction
                # Zep pattern: set invalid_at (never delete)
                self._conn.execute(
                    "UPDATE voice_memories SET invalid_at = ?, expired_at = ? WHERE id = ?",
                    (now, now, linked_id_int),
                )
                self._history.add_history(
                    linked_id_int, old_text, None, "SUPERSEDE",
                    created_at=row["created_at"], updated_at=now,
                )
                logger.info("[Resolve] id=%d superseded by id=%d (overlap=%.2f)",
                           linked_id_int, new_mem_id, overlap)
            # else: medium overlap, related but not contradictory — leave both

        self._conn.commit()

    @staticmethod
    def _normalize_for_compare(text: str) -> set[str]:
        """Extract content words for overlap comparison."""
        import re as _re
        # Split on whitespace and punctuation, keep CJK chars and alphanumeric
        tokens = _re.findall(r'[一-鿿]+|[a-zA-Z0-9]+', (text or "").lower())
        return set(tokens)

    @staticmethod
    def _text_overlap(a: set[str], b: set[str]) -> float:
        """Jaccard-like overlap between two token sets."""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    # Decay rates per scope — loaded from config.yaml
    # Cognitive science basis: Tulving 1972, Anderson ACT-R
    @staticmethod
    def _scope_decay_k(scope: str) -> float:
        decay_map = cfg("memory.scope_decay", {})
        return decay_map.get(scope, 0.02)

    def _compute_retention(self, mem: dict) -> float:
        """2-factor retention: age decay × confirmation boost.

        retention = e^(-k*age_days) * min(1.8, 1 + ln(1+confirmations)*0.08)
        """
        now = time.time()
        created_at = mem.get("created_at") or now
        k = mem.get("decay_k") or 0.02
        confirmations = mem.get("confirmation_count") or 0
        age_days = max(0.0, (now - created_at) / 86400)
        age_decay = math.exp(-k * age_days)
        access_boost = min(
            cfg("retrieval.retention_access_boost_cap", 1.8),
            1.0 + math.log1p(confirmations) * cfg("retrieval.retention_confirmation_factor", 0.08),
        )
        return age_decay * access_boost

    def _store_memory(
        self,
        content: str,
        scope: str,
        linked_memory_ids: list[str] | None = None,
        attributed_to: str | None = None,
        confidence: float = 0.7,
        metadata: dict | None = None,
    ) -> int | None:
        """Store a memory with bi-temporal fields, confidence, and decay parameters.

        Bi-temporal model (Zep/Graphiti):
        - valid_at = now (when the fact was recorded)
        - invalid_at = NULL (currently active)
        - expired_at = NULL (not superseded)

        Decay (cognitive science-based):
        - decay_k set per scope (semantic=0.0005, episodic=0.02, procedural=0.005)
        - confirmation_count starts at 0
        """
        if not self._conn:
            return None
        h = _md5(content)
        now = time.time()
        links_json = json.dumps(linked_memory_ids) if linked_memory_ids else None
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"
        decay_k = self._scope_decay_k(scope)
        try:
            cursor = self._conn.execute(
                """INSERT INTO voice_memories
                   (user_id, content, scope, created_at, updated_at, hash,
                    linked_memory_ids, attributed_to, metadata,
                    confidence, confirmation_count, last_used_at, decay_k,
                    valid_at, invalid_at, expired_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, NULL, NULL)""",
                (self._user_id, content, scope, now, now, h,
                 links_json, attributed_to, meta_json,
                 confidence, now, decay_k, now),
            )
            self._conn.commit()
            mem_id = cursor.lastrowid

            # Record ADD history
            self._history.add_history(
                mem_id, None, content, "ADD",
                created_at=now, updated_at=now,
            )

            # Embed (skip during bulk ingestion for speed — backfill later)
            if not self._skip_embedding:
                self._embed_memory(mem_id, content)
            return mem_id
        except Exception as e:
            logger.debug("[MemoryExtract] store failed: %s", e)
            return None

    def _embed_memory(self, mem_id: int, content: str) -> None:
        """Embed a memory and store the vector. Runs synchronously (fast enough)."""
        if not self._embedder or not self._conn:
            return
        try:
            vec = self._embedder.embed_single(content)
            if vec and not all(v == 0 for v in vec):
                blob = _encode_embedding(vec)
                self._conn.execute(
                    "UPDATE voice_memories SET embedding = ? WHERE id = ?",
                    (blob, mem_id),
                )
                self._conn.commit()
        except Exception as e:
            logger.debug("[MemoryExtract] embedding failed for id=%d: %s", mem_id, e)

    def _backfill_embeddings(self, batch_size: int = 5) -> int:
        """Embed all memories that have NULL embeddings using batch API. Returns count embedded."""
        if not self._embedder or not self._conn:
            return 0
        rows = self._conn.execute(
            "SELECT id, content FROM voice_memories WHERE embedding IS NULL AND user_id = ?",
            (self._user_id,),
        ).fetchall()
        if not rows:
            return 0
        count = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [r["content"][:4000] for r in batch]
            ids = [r["id"] for r in batch]
            try:
                vecs = self._embedder.embed(texts)
                for mem_id, vec in zip(ids, vecs):
                    if vec and not all(v == 0 for v in vec):
                        blob = _encode_embedding(vec)
                        self._conn.execute(
                            "UPDATE voice_memories SET embedding = ? WHERE id = ?",
                            (blob, mem_id),
                        )
                        count += 1
            except Exception as e:
                logger.debug("[MemoryExtract] backfill batch failed: %s", e)
        self._conn.commit()
        return count

    def _get_existing_for_dedup(self, limit: int = 20) -> list[dict]:
        """Get recent active memories for dedup context.

        Only returns memories where invalid_at IS NULL (Zep pattern —
        invalidated memories are excluded from active dedup).
        """
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                "SELECT id, content, hash, embedding, confidence, confirmation_count "
                "FROM voice_memories "
                "WHERE user_id = ? AND scope != 'conversation' AND invalid_at IS NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (self._user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # CRUD operations — adapted from mem0's Memory.update()/delete()
    # ------------------------------------------------------------------

    def get_memory(self, mem_id: int) -> dict | None:
        """Get a single memory by ID, including bi-temporal fields."""
        if not self._conn:
            return None
        try:
            row = self._conn.execute(
                "SELECT id, content, scope, created_at, updated_at, hash, "
                "linked_memory_ids, attributed_to, metadata, "
                "confidence, confirmation_count, last_used_at, decay_k, "
                "valid_at, invalid_at, expired_at "
                "FROM voice_memories WHERE id = ? AND user_id = ?",
                (mem_id, self._user_id),
            ).fetchone()
            if row:
                d = dict(row)
                d["retention_score"] = round(self._compute_retention(d), 4)
                return d
            return None
        except Exception:
            return None

    def update_memory(self, mem_id: int, new_content: str) -> bool:
        """Update a memory's content. Adapted from mem0's _update_memory().

        Steps (same as mem0):
        1. Fetch existing memory
        2. Record UPDATE history
        3. Recompute hash
        4. Re-embed
        5. UPDATE in DB (FTS5 auto-sync trigger handles the index)
        """
        if not self._conn:
            return False
        try:
            existing = self.get_memory(mem_id)
            if not existing:
                return False

            old_content = existing["content"]
            if old_content == new_content:
                return True  # no change

            now = time.time()
            new_hash = _md5(new_content)

            # Record history before update
            self._history.add_history(
                mem_id, old_content, new_content, "UPDATE",
                created_at=existing.get("created_at"),
                updated_at=now,
            )

            # Update content, hash, updated_at
            self._conn.execute(
                "UPDATE voice_memories SET content = ?, hash = ?, updated_at = ? WHERE id = ?",
                (new_content, new_hash, now, mem_id),
            )
            self._conn.commit()

            # Re-embed
            self._embed_memory(mem_id, new_content)

            # Invalidate frozen snapshot
            self._frozen_system_built = False

            logger.info("[MemoryUpdate] id=%d updated", mem_id)
            return True
        except Exception as e:
            logger.debug("[MemoryUpdate] failed for id=%d: %s", mem_id, e)
            return False

    def delete_memory(self, mem_id: int) -> bool:
        """Delete a memory. Adapted from mem0's _delete_memory().

        Steps (same as mem0):
        1. Fetch existing memory
        2. Record DELETE history
        3. DELETE from DB (FTS5 auto-sync trigger handles the index)
        """
        if not self._conn:
            return False
        try:
            existing = self.get_memory(mem_id)
            if not existing:
                return False

            now = time.time()

            # Record history before delete
            self._history.add_history(
                mem_id, existing["content"], None, "DELETE",
                created_at=existing.get("created_at"),
                updated_at=now,
                is_deleted=1,
            )

            # Delete
            self._conn.execute("DELETE FROM voice_memories WHERE id = ?", (mem_id,))
            self._conn.commit()

            # Invalidate frozen snapshot
            self._frozen_system_built = False

            logger.info("[MemoryDelete] id=%d deleted", mem_id)
            return True
        except Exception as e:
            logger.debug("[MemoryDelete] failed for id=%d: %s", mem_id, e)
            return False

    def get_memory_history(self, mem_id: int) -> list[dict]:
        """Get change history for a specific memory."""
        if not hasattr(self, '_history'):
            return []
        return self._history.get_history(str(mem_id))

    def get_recent_history(self, limit: int = 20) -> list[dict]:
        """Get recent history records for this user."""
        if not hasattr(self, '_history'):
            return []
        return self._history.get_recent_history(self._user_id, limit=limit)

    # ------------------------------------------------------------------
    # Tools — memory_search + memory_update + memory_delete + memory_history
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "memory_search",
                "description": "Search the user's persistent memory for facts, preferences, and prior context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_update",
                "description": "Update an existing memory by ID. Use when correcting or refining a stored fact.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "description": "The memory ID to update"},
                        "content": {"type": "string", "description": "The new content for this memory"},
                    },
                    "required": ["memory_id", "content"],
                },
            },
            {
                "name": "memory_delete",
                "description": "Delete a memory by ID. Use when a stored fact is wrong or no longer relevant.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "description": "The memory ID to delete"},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "memory_history",
                "description": "Get the change history for a specific memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "description": "The memory ID to get history for"},
                    },
                    "required": ["memory_id"],
                },
            },
        ]

    async def ahandle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        user_id = kwargs.get("user_id") or self._user_id
        if not user_id or not self._conn:
            return json.dumps({"ok": False, "error": "no user_id or db"})

        if tool_name == "memory_search":
            query = args.get("query", "")
            results = self._hybrid_search(query, limit=cfg("retrieval.prefetch_limit", 5))
            return json.dumps({"ok": True, "results": results}, ensure_ascii=False)

        if tool_name == "memory_update":
            mem_id = args.get("memory_id")
            content = args.get("content", "")
            if not mem_id or not content:
                return json.dumps({"ok": False, "error": "memory_id and content required"})
            success = self.update_memory(int(mem_id), content)
            return json.dumps({"ok": success}, ensure_ascii=False)

        if tool_name == "memory_delete":
            mem_id = args.get("memory_id")
            if not mem_id:
                return json.dumps({"ok": False, "error": "memory_id required"})
            success = self.delete_memory(int(mem_id))
            return json.dumps({"ok": success}, ensure_ascii=False)

        if tool_name == "memory_history":
            mem_id = args.get("memory_id")
            if not mem_id:
                return json.dumps({"ok": False, "error": "memory_id required"})
            history = self.get_memory_history(int(mem_id))
            return json.dumps({"ok": True, "history": history}, ensure_ascii=False)

        return json.dumps({"ok": False, "error": f"unknown tool: {tool_name}"})

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_all_memories(self, limit: int = 50, include_invalidated: bool = False) -> list[dict]:
        """Get all stored memories for the user (excluding raw conversation turns).

        By default only returns active memories (invalid_at IS NULL).
        Pass include_invalidated=True to also see superseded memories.
        """
        if not self._conn:
            return []
        try:
            invalid_filter = "" if include_invalidated else "AND invalid_at IS NULL"
            rows = self._conn.execute(
                f"SELECT id, content, scope, created_at, updated_at, "
                f"linked_memory_ids, attributed_to, "
                f"confidence, confirmation_count, last_used_at, decay_k, "
                f"valid_at, invalid_at, expired_at "
                f"FROM voice_memories "
                f"WHERE user_id = ? AND scope != 'conversation' {invalid_filter} "
                f"ORDER BY created_at DESC LIMIT ?",
                (self._user_id, limit),
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                # Add computed retention score
                d["retention_score"] = round(self._compute_retention(d), 4)
                result.append(d)
            return result
        except Exception:
            return []

    def shutdown(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

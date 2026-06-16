"""VoiceMemoryConsolidator — background memory consolidation (dreaming cycle).

References:
- Letta sleeptime agent: runs in background, reorganizes memory blocks
  (source: letta/prompts/system_prompts/sleeptime_v2.py)
- Zep build_communities(): entity clustering for pattern detection
  (source: graphiti_core/graphiti.py)
- Zep episode_mentions_reranker: frequency = reliability signal
  (source: graphiti_core/search/search_utils.py)
- vehicle_memory/consolidation.py: DEDUCTION_PROMPT + INDUCTION_PROMPT

Two phases:
1. Deduction: sweep for contradictions missed by inline detection, stale cleanup
2. Induction: cross-memory pattern recognition (behavioral patterns)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from .config import cfg

logger = logging.getLogger(__name__)

# Deduction prompt — loaded from config.yaml
# See config.example.yaml for the schema and minimal template.
DEDUCTION_PROMPT = cfg("prompts.deduction", "")

# Induction prompt — loaded from config.yaml
INDUCTION_PROMPT = cfg("prompts.induction", "")


def _extract_json(raw: str) -> Any:
    """Extract JSON from LLM output, handling markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try finding JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


class VoiceMemoryConsolidator:
    """Background consolidation worker (dreaming cycle).

    References:
    - Letta sleeptime agent: runs in background, reorganizes memory blocks
      Tools: memory_replace, memory_insert, memory_rethink, memory_finish_edits
      Prompt: "Not every observation warrants a memory edit" but aim for "high recall"

    - Zep build_communities(): entity clustering for higher-level abstractions
      Clears existing communities, rebuilds via clustering algorithm

    - Zep episode_mentions_reranker: frequency = reliability
      Edges appearing in more episodes are implicitly more confident

    - vehicle_memory/consolidation.py ConsolidationWorker:
      Triggers: ≥50 new conclusions OR ≥8 hours since last
      Two phases: Deduction Specialist + Induction Specialist

    Our adaptation:
    - Triggers: ≥10 new memories OR ≥4 hours since last consolidation
    - Deduction: conflict detection, stale cleanup, hidden inference
    - Induction: cross-memory pattern recognition
    - New patterns stored with slow decay (k=0.001, like Letta core memory)
    """

    def __init__(
        self,
        memory_provider: Any,
        *,
        llm_client: Any = None,
        model: str | None = None,
        check_interval: float | None = None,
        memory_threshold: int | None = None,
        time_threshold_hours: float | None = None,
    ):
        self._provider = memory_provider
        self._llm = llm_client
        self._model = model or cfg("model.llm", "qwen-flash")
        self.check_interval = check_interval if check_interval is not None else cfg("consolidation.check_interval", 600.0)
        self.memory_threshold = memory_threshold if memory_threshold is not None else cfg("consolidation.memory_threshold", 10)
        self.time_threshold = (time_threshold_hours if time_threshold_hours is not None else cfg("consolidation.time_threshold_hours", 4.0)) * 3600
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_consolidation: float = 0.0
        self._last_memory_count: int = 0

    async def start(self) -> None:
        """Start the background consolidation loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "[Consolidator] Started (check_interval=%.0fs, threshold=%d memories or %.0fh)",
            self.check_interval, self.memory_threshold, self.time_threshold / 3600,
        )

    async def stop(self) -> None:
        """Stop the background consolidation loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[Consolidator] Stopped")

    async def _run_loop(self) -> None:
        """Background loop — check periodically if consolidation should run."""
        while self._running:
            try:
                await self._check_and_consolidate()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[Consolidator] Error: %s", e, exc_info=True)
                await asyncio.sleep(self.check_interval)

    async def _check_and_consolidate(self) -> None:
        """Check trigger conditions and run consolidation if needed.

        Trigger logic (from vehicle_memory/consolidation.py):
        - ≥ memory_threshold new memories since last consolidation
        - ≥ time_threshold hours since last consolidation
        """
        if not self._provider._conn:
            return

        try:
            current_count = self._provider._conn.execute(
                "SELECT COUNT(*) FROM voice_memories "
                "WHERE user_id = ? AND invalid_at IS NULL AND scope != 'conversation'",
                (self._provider._user_id,),
            ).fetchone()[0]

            elapsed = time.time() - self._last_consolidation
            new_count = current_count - self._last_memory_count

            if new_count >= self.memory_threshold and elapsed >= self.time_threshold:
                logger.info(
                    "[Consolidator] Triggered: %d new memories, %.0fh elapsed",
                    new_count, elapsed / 3600,
                )
                await self.consolidate()
                self._last_consolidation = time.time()
                self._last_memory_count = current_count
        except Exception as e:
            logger.debug("[Consolidator] Check failed: %s", e)

    async def consolidate(self) -> dict[str, Any]:
        """Run the full consolidation cycle.

        Two phases (from vehicle_memory/consolidation.py):
        1. Deduction: conflict detection, stale cleanup, hidden inference
        2. Induction: cross-memory pattern recognition

        Returns summary of actions taken.
        """
        result = {
            "conflicts_resolved": 0,
            "stale_marked": 0,
            "new_deductions": 0,
            "patterns_found": 0,
            "hindsight_rules": 0,
            "hindsight_pruned": 0,
        }

        if not self._llm or not self._provider._conn:
            return result

        # Fetch all active memories
        memories = self._provider._conn.execute(
            "SELECT id, content, scope, confidence, confirmation_count, created_at "
            "FROM voice_memories "
            "WHERE user_id = ? AND invalid_at IS NULL AND scope != 'conversation' "
            "ORDER BY created_at DESC LIMIT ?",
            (self._provider._user_id, cfg("consolidation.max_memories", 50)),
        ).fetchall()

        if not memories:
            return result

        memory_dicts = [dict(m) for m in memories]

        # --- Phase 1: Deduction ---
        if len(memories) >= 2:
            deduction_result = await self._run_deduction(memory_dicts)
            result["conflicts_resolved"] = deduction_result.get("conflicts_resolved", 0)
            result["stale_marked"] = deduction_result.get("stale_marked", 0)
            result["new_deductions"] = deduction_result.get("new_deductions", 0)
            result["hindsight_rules"] = deduction_result.get("hindsight_rules", 0)

        # --- Phase 2: Induction ---
        if len(memories) >= cfg("consolidation.induction_min_memories", 4):
            induction_result = await self._run_induction(memory_dicts)
            result["patterns_found"] = induction_result.get("patterns_found", 0)

        # --- Phase 3: Hindsight lifecycle management (Mem0 decay + Zep invalidation) ---
        pruned = await self._prune_hindsight_rules()
        result["hindsight_pruned"] = pruned

        logger.info(
            "[Consolidator] Complete: %d conflicts, %d stale, %d deductions, %d patterns, %d hindsight, %d pruned",
            result["conflicts_resolved"], result["stale_marked"],
            result["new_deductions"], result["patterns_found"],
            result["hindsight_rules"], pruned,
        )
        return result

    async def _run_deduction(self, memories: list[dict]) -> dict[str, int]:
        """Deduction phase: conflict detection + stale cleanup + hidden inference.

        References:
        - vehicle_memory/consolidation.py DEDUCTION_PROMPT
        - Zep resolve_edge_contradictions(): invalid_at pattern
        """
        result = {"conflicts_resolved": 0, "stale_marked": 0, "new_deductions": 0, "hindsight_rules": 0}

        # Format memories for prompt
        memory_text = "\n".join(
            f"[{m['id']}] {m['content']} "
            f"(confidence={m.get('confidence', 0.7):.2f}, "
            f"confirmations={m.get('confirmation_count', 0)})"
            for m in memories
        )

        prompt = DEDUCTION_PROMPT.format(memories=memory_text)

        try:
            raw = ""
            async for chunk in self._llm.chat_stream(
                [{"role": "user", "content": prompt}], model=self._model
            ):
                raw += chunk

            data = _extract_json(raw)
            if not data or not isinstance(data, dict):
                return result

            now = time.time()

            lock = getattr(self._provider, '_write_lock', None)
            if lock:
                await lock.acquire()
            try:
                # Process conflicts (Zep invalidation pattern)
                for conflict in data.get("conflicts", []):
                    keep_id = conflict.get("keep_id")
                    invalidate_id = conflict.get("invalidate_id")
                    reason = conflict.get("reason", "")
                    if keep_id and invalidate_id:
                        self._provider._conn.execute(
                            "UPDATE voice_memories SET invalid_at = ?, expired_at = ? WHERE id = ?",
                            (now, now, invalidate_id),
                        )
                        self._provider._history.add_history(
                            invalidate_id, None, None, "CONSOLIDATE_INVALIDATE",
                            created_at=now, updated_at=now,
                        )
                        result["conflicts_resolved"] += 1
                        logger.info("[Consolidator] Conflict: %d invalidated, %d kept: %s",
                                   invalidate_id, keep_id, reason)

                # Process stale memories
                for stale in data.get("stale", []):
                    memory_id = stale.get("memory_id")
                    reason = stale.get("reason", "")
                    if memory_id:
                        self._provider._conn.execute(
                            "UPDATE voice_memories SET expired_at = ? WHERE id = ?",
                            (now, memory_id),
                        )
                        result["stale_marked"] += 1
                        logger.info("[Consolidator] Stale: %d — %s", memory_id, reason)

                # Store new deductions (Letta rethink pattern: derive new insights)
                for deduction in data.get("new_deductions", []):
                    content = (deduction.get("content") or "").strip()
                    if not content:
                        continue
                    confidence = min(max(float(deduction.get("confidence", cfg("memory.deduction_default_confidence", 0.6))), 0), cfg("memory.deduction_confidence_cap", 0.9))
                    evidence_ids = deduction.get("evidence_ids", [])
                    mem_id = await self._provider._store_memory(
                        content, "episodic",
                        linked_memory_ids=[str(eid) for eid in evidence_ids if eid],
                        confidence=confidence,
                    )
                    if mem_id:
                        # Set slow decay for derived deductions (Letta core memory pattern)
                        self._provider._conn.execute(
                            "UPDATE voice_memories SET decay_k = ? WHERE id = ?",
                            (cfg("memory.deduction_decay_k", 0.001), mem_id),
                        )
                        result["new_deductions"] += 1

                # Store hindsight rules (reflection patterns for better recall)
                for rule in data.get("hindsight_rules", []):
                    content = (rule.get("content") or "").strip()
                    if not content:
                        continue
                    evidence_ids = rule.get("evidence_ids", [])
                    mem_id = await self._provider._store_memory(
                        content, "procedural",
                        linked_memory_ids=[str(eid) for eid in evidence_ids if eid],
                        confidence=cfg("memory.hindsight_confidence", 0.8),
                        metadata={"kind": "hindsight_rule",
                                  "hit_count": 0, "injected_count": 0},
                    )
                    if mem_id:
                        # Hindsight rules: very slow decay (semantic-like permanence)
                        self._provider._conn.execute(
                            "UPDATE voice_memories SET decay_k = ? WHERE id = ?",
                            (cfg("memory.hindsight_decay_k", 0.0005), mem_id),
                        )
                        result["hindsight_rules"] += 1
                        logger.info("[Consolidator] Hindsight rule: %s", content[:60])

                self._provider._conn.commit()
            finally:
                if lock:
                    lock.release()

        except Exception as e:
            logger.debug("[Consolidator] Deduction failed: %s", e)

        return result

    async def _run_induction(self, memories: list[dict]) -> dict[str, int]:
        """Induction phase: cross-memory pattern recognition.

        References:
        - vehicle_memory/consolidation.py INDUCTION_PROMPT
        - Zep build_communities(): higher-level abstractions
        - Zep episode_mentions_reranker: frequency = reliability
        """
        result = {"patterns_found": 0}

        memory_text = "\n".join(
            f"[{m['id']}] {m['content']} "
            f"(type={m.get('scope', 'episodic')}, "
            f"confidence={m.get('confidence', 0.7):.2f})"
            for m in memories
        )

        prompt = INDUCTION_PROMPT.format(memories=memory_text)

        try:
            raw = ""
            async for chunk in self._llm.chat_stream(
                [{"role": "user", "content": prompt}], model=self._model
            ):
                raw += chunk

            patterns = _extract_json(raw)
            if not isinstance(patterns, list):
                return result

            lock = getattr(self._provider, '_write_lock', None)
            if lock:
                await lock.acquire()
            try:
                # Map memory_type from induction prompt to scope
                _TYPE_TO_SCOPE = {
                    "habit": "procedural",
                    "preference": "procedural",
                    "relationship": "semantic",
                }

                for pattern in patterns:
                    content = (pattern.get("content") or "").strip()
                    evidence_ids = pattern.get("evidence_ids", [])
                    if not content or len(evidence_ids) < 2:
                        continue

                    evidence_count = len(evidence_ids)
                    confidence = min(
                        cfg("consolidation.induction_confidence_base", 0.5)
                        + cfg("consolidation.induction_confidence_per_evidence", 0.08) * evidence_count,
                        cfg("consolidation.induction_confidence_cap", 0.92),
                    )
                    memory_type = pattern.get("memory_type", "habit")
                    scope = _TYPE_TO_SCOPE.get(memory_type, "procedural")

                    mem_id = await self._provider._store_memory(
                        content, scope,
                        linked_memory_ids=[str(eid) for eid in evidence_ids if eid],
                        confidence=confidence,
                    )
                    if mem_id:
                        # Derived patterns: slow decay (Letta core memory pattern)
                        # confirmation_count = evidence count (Zep episode count pattern)
                        self._provider._conn.execute(
                            "UPDATE voice_memories SET decay_k = ?, "
                            "confirmation_count = ? WHERE id = ?",
                            (cfg("memory.deduction_decay_k", 0.001), evidence_count, mem_id),
                        )
                        result["patterns_found"] += 1
                        logger.info("[Consolidator] Pattern: %s (type=%s, evidence=%d)", content[:60], memory_type, evidence_count)

                self._provider._conn.commit()
            finally:
                if lock:
                    lock.release()

        except Exception as e:
            logger.debug("[Consolidator] Induction failed: %s", e)

        return result

    async def _prune_hindsight_rules(self) -> int:
        """Layer 3: Retire ineffective hindsight rules (Zep invalidation pattern).

        Prunes rules that have:
        1. Low hit rate after enough injections: hit_count / injected_count < 0.05
           and injected_count >= 30 (rule is consistently irrelevant)
        2. Stale: created > 30 days ago and hit_count == 0
           (rule was never useful)

        Uses Zep's bi-temporal invalidation (set invalid_at, never delete).
        """
        if not self._provider._conn:
            return 0

        now = time.time()
        pruned = 0
        stale_threshold = now - cfg("hindsight.prune_stale_days", 30) * 86400

        try:
            rows = self._provider._conn.execute(
                "SELECT id, content, metadata, created_at "
                "FROM voice_memories "
                "WHERE user_id = ? AND json_extract(metadata, '$.kind') = 'hindsight_rule' "
                "AND invalid_at IS NULL",
                (self._provider._user_id,),
            ).fetchall()

            lock = getattr(self._provider, '_write_lock', None)
            if lock:
                await lock.acquire()
            try:
                for row in rows:
                    meta = json.loads(row["metadata"] or "{}")
                    hit = meta.get("hit_count", 0)
                    injected = meta.get("injected_count", 0)
                    created_at = row["created_at"] or now

                    should_invalidate = False
                    reason = ""

                    # Rule 1: Low hit rate after enough data points
                    if injected >= cfg("hindsight.prune_min_injections", 30) and hit / injected < cfg("hindsight.prune_hit_rate", 0.05):
                        should_invalidate = True
                        reason = f"low hit rate ({hit}/{injected}={hit/injected:.1%})"

                    # Rule 2: Stale and never hit
                    elif created_at < stale_threshold and hit == 0:
                        should_invalidate = True
                        reason = f"stale ({(now - created_at) / 86400:.0f}d, 0 hits)"

                    if should_invalidate:
                        self._provider._conn.execute(
                            "UPDATE voice_memories SET invalid_at = ?, expired_at = ? WHERE id = ?",
                            (now, now, row["id"]),
                        )
                        if hasattr(self._provider, '_history'):
                            self._provider._history.add_history(
                                row["id"], row["content"], None, "HINDSIGHT_PRUNE",
                                created_at=created_at, updated_at=now,
                            )
                        pruned += 1
                        logger.info("[Consolidator] Pruned hindsight rule %d: %s — %s",
                                    row["id"], row["content"][:40], reason)

                if pruned:
                    self._provider._conn.commit()
            finally:
                if lock:
                    lock.release()

        except Exception as e:
            logger.debug("[Consolidator] Hindsight pruning failed: %s", e)

        return pruned

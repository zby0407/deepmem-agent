"""Memory extraction pipeline — adapted from mem0's V3 additive extraction.

Single LLM call per turn: extracts facts AND handles dedup in one pass.
Uses mem0's ADDITIVE_EXTRACTION_PROMPT with few-shot examples.
Semantic dedup via embedding similarity (not just hash).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from typing import Any

from .config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# V3 Additive Extraction Prompt — loaded from config.yaml
# The prompt template uses {placeholders} filled at runtime.
# See config.example.yaml for the schema and minimal template.
# Production-tuned prompts are in config.yaml (gitignored).
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = cfg("prompts.extraction", "")


def _md5(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


async def extract_and_dedup(
    llm_client: Any,
    model: str,
    user_msg: str,
    assistant_msg: str,
    existing_memories: list[dict[str, str]],
    recently_extracted: list[str] | None = None,
    last_k_messages: list[dict[str, str]] | None = None,
    observation_date: str | None = None,
) -> list[dict[str, Any]]:
    """Single LLM call: extract facts with built-in dedup.

    Uses mem0's V3 additive approach — one LLM call does extraction + dedup.
    Returns list of {"text": "...", "attributed_to": "user|assistant", "linked_memory_ids": [...]}
    """
    if not llm_client:
        return []

    today = observation_date or date.today().strftime("%Y-%m-%d")

    # Format existing memories
    existing_str = "（无）"
    if existing_memories:
        lines = [f'[{{"id": "{m["id"]}", "text": "{m["text"]}"}}]' for m in existing_memories[:cfg("buffers.dedup_prompt_existing_cap", 10)]]
        existing_str = "\n".join(lines)

    # Format recently extracted
    recently_str = "（无）"
    if recently_extracted:
        recently_str = "\n".join(f"- {r}" for r in recently_extracted[-20:])

    # Format last k messages
    last_k_str = "（无）"
    if last_k_messages:
        lines = []
        for msg in last_k_messages[-10:]:
            role = msg.get("role", "")
            content = msg.get("content", "")[:200]  # truncate
            if role and content:
                lines.append(f"{role}: {content}")
        if lines:
            last_k_str = "\n".join(lines)

    # Format new messages
    new_msgs = []
    if user_msg:
        new_msgs.append(f'{{"role": "user", "content": "{user_msg}"}}')
    if assistant_msg:
        new_msgs.append(f'{{"role": "assistant", "content": "{assistant_msg}"}}')
    new_messages_str = f'[{", ".join(new_msgs)}]' if new_msgs else "[]"

    prompt = EXTRACTION_PROMPT.format(
        observation_date=today,
        existing_memories=existing_str,
        recently_extracted=recently_str,
        last_k_messages=last_k_str,
        new_messages=new_messages_str,
    )

    try:
        result = await llm_client.chat_with_tools(
            [{"role": "user", "content": prompt}],
            model=model,
        )
        content = result.get("content") or ""
        parsed = _parse_json(content)
        memories = parsed.get("memory", [])
        if not isinstance(memories, list):
            return []
        # Validate and filter
        valid = []
        for m in memories:
            if not isinstance(m, dict):
                continue
            text = (m.get("text") or "").strip()
            if not text:
                continue
            scope = m.get("scope", "episodic")
            if scope not in ("semantic", "episodic", "procedural"):
                scope = "episodic"
            valid.append({
                "text": text,
                "scope": scope,
                "attributed_to": m.get("attributed_to", "user"),
                "linked_memory_ids": m.get("linked_memory_ids", []),
            })
        return valid
    except Exception as e:
        logger.warning("[MemoryExtract] extraction failed: %s", e)
        return []


def semantic_dedup(
    new_texts: list[str],
    existing_embeddings: dict[str, list[float]],
    embedder: Any,
    threshold: float | None = None,
) -> list[str]:
    """Remove semantically similar duplicates using embedding similarity.

    Returns only new_texts that are NOT semantically equivalent to existing memories.
    """
    if threshold is None:
        threshold = cfg("retrieval.semantic_dedup_threshold", 0.90)
    if not embedder or not existing_embeddings or not new_texts:
        return new_texts

    try:
        new_vecs = embedder.embed(new_texts)
    except Exception:
        return new_texts

    existing_items = list(existing_embeddings.items())
    result = []
    for i, text in enumerate(new_texts):
        if i >= len(new_vecs):
            result.append(text)
            continue
        new_vec = new_vecs[i]
        is_dup = False
        for _, exist_vec in existing_items:
            score = _cosine_similarity(new_vec, exist_vec)
            if score >= threshold:
                is_dup = True
                break
        if not is_dup:
            result.append(text)
    return result


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}

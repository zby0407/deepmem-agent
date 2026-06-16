"""Scoring utilities for hybrid retrieval — adapted from mem0/utils/scoring.py.

Provides:
- BM25 normalization: Sigmoid normalization of raw FTS5/BM25 scores to [0, 1].
- BM25 parameter selection: Query-length-adaptive sigmoid parameters.
- Additive scoring: Combined scoring with semantic + BM25 + linked-memory boost.
"""

from __future__ import annotations

import math
from typing import Any

from .config import cfg


def get_bm25_params(query: str) -> tuple[float, float]:
    """Get BM25 sigmoid parameters based on query length.

    Longer queries tend to have higher raw BM25 scores, so we adjust
    the sigmoid midpoint and steepness accordingly.

    Returns:
        (midpoint, steepness) for sigmoid normalization.
    """
    num_terms = len(query.split())
    bm25 = cfg("scoring.bm25") or {}

    if num_terms <= 3:
        p = bm25.get("short", {})
        return p.get("midpoint", 5.0), p.get("steepness", 0.7)
    elif num_terms <= 6:
        p = bm25.get("medium", {})
        return p.get("midpoint", 7.0), p.get("steepness", 0.6)
    elif num_terms <= 9:
        p = bm25.get("long_short", {})
        return p.get("midpoint", 9.0), p.get("steepness", 0.5)
    elif num_terms <= 15:
        p = bm25.get("long_medium", {})
        return p.get("midpoint", 10.0), p.get("steepness", 0.5)
    else:
        p = bm25.get("long_long", {})
        return p.get("midpoint", 12.0), p.get("steepness", 0.5)


def normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    """Normalize BM25 score to [0, 1] using logistic sigmoid.

    Args:
        raw_score: Raw BM25 score (positive, typically 0-20+).
        midpoint: Score at which sigmoid outputs 0.5.
        steepness: Controls how quickly sigmoid transitions.

    Returns:
        Normalized score in range [0, 1].
    """
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


ENTITY_BOOST_WEIGHT = cfg("retrieval.entity_boost_weight", 0.5)


def score_and_rank(
    semantic_results: list[dict[str, Any]],
    bm25_scores: dict[str, float],
    entity_boosts: dict[str, float],
    threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Score candidates additively and return top-k results.

    For each candidate:
        semantic_score is taken from the result's score field.
        combined = (semantic + bm25 + entity_boost) / max_possible

    Threshold gates the semantic score BEFORE combining -- candidates
    below the threshold are excluded even if BM25/entity would boost them.

    The divisor adapts based on which signals are active:
        - Semantic only: max_possible = 1.0
        - Semantic + BM25: max_possible = 2.0
        - Semantic + BM25 + entity: max_possible = 2.5
        - Semantic + entity (no BM25): max_possible = 1.5

    Returns:
        List of scored result dicts sorted by combined score descending.
    """
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)

    max_possible = 1.0
    if has_bm25:
        max_possible += 1.0
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT

    scored: list[dict[str, Any]] = []

    for result in semantic_results:
        mem_id = result.get("id")
        if mem_id is None:
            continue

        semantic_score = result.get("score", 0.0)
        if semantic_score < threshold:
            continue

        mem_id_str = str(mem_id)
        bm25_score = bm25_scores.get(mem_id_str, 0.0)
        entity_boost = entity_boosts.get(mem_id_str, 0.0)

        raw_combined = semantic_score + bm25_score + entity_boost
        combined = min(raw_combined / max_possible, 1.0)

        scored.append(
            {
                "id": mem_id_str,
                "content": result.get("content", ""),
                "scope": result.get("scope", ""),
                "score": combined,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

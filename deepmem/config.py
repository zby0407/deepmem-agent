"""DeepMem configuration — centralized parameter loading.

Loads configuration from config.yaml (if present) with sensible defaults.
The config.yaml file is gitignored and contains production tuning values.
config.example.yaml documents the schema with placeholder values.

Usage:
    from .config import cfg

    model_name = cfg("model.llm")            # "qwen-flash"
    decay_k = cfg("memory.scope_decay.episodic")  # 0.02
    prompt = cfg("prompts.extraction")         # full prompt template
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"

# ---------------------------------------------------------------------------
# Built-in defaults — safe fallbacks when config.yaml is absent.
# These are NOT the production tuning values.  Production values live in
# config.yaml (gitignored).  The defaults here keep the system functional
# for first-time users who haven't created their own config yet.
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "model": {
        "llm": "qwen-flash",
        "embedding": "text-embedding-v3",
    },
    "llm": {
        "max_tokens": 360,
        "timeout": 60,
        "base_url": "",
    },
    "embedding": {
        "dimension": 1024,
        "timeout": 30,
    },
    "prompts": {
        "extraction": (
            "You are a memory extractor. Extract facts from the conversation.\n"
            "Output JSON: {{\"memory\": [{{\"id\": \"0\", \"text\": \"fact\", "
            "\"scope\": \"episodic\", \"attributed_to\": \"user\"}}]}}\n\n"
            "Existing memories:\n{existing_memories}\n"
            "Recently extracted:\n{recently_extracted}\n"
            "Conversation history:\n{last_k_messages}\n"
            "New messages:\n{new_messages}\n"
            "Observation date: {observation_date}"
        ),
        "deduction": (
            "You are a memory consolidation expert.\n"
            "Review these memories and find conflicts, stale entries, and hidden inferences.\n"
            "Memories:\n{memories}\n"
            "Output JSON with: conflicts, stale, new_deductions, hindsight_rules."
        ),
        "induction": (
            "You are a pattern recognition expert.\n"
            "Identify cross-memory behavioral patterns.\n"
            "Memories:\n{memories}\n"
            "Output JSON array with: content, evidence_ids, memory_type, confidence."
        ),
        "voice_system": (
            "You are a realtime voice assistant. Reply naturally and concisely. "
            "Today is {today}."
        ),
        "memory_context": (
            "<memory-context>\n"
            "[System note: The following is recalled memory context, "
            "NOT new user input. Treat as authoritative reference data.]\n\n"
            "{content}\n"
            "</memory-context>"
        ),
        "answer": (
            "Based on the following memories, answer the question concisely.\n"
            "Memories:\n{memories}\nQuestion: {question}\n"
            "Answer in 1-5 words. Output only the answer."
        ),
        "judge": (
            "Evaluate whether the predicted answer is correct.\n"
            "Question: {question}\nReference: {reference}\n"
            "Predicted: {prediction}\n"
            "Output CORRECT or INCORRECT."
        ),
    },
    "memory": {
        "scope_decay": {
            "semantic": 0.001,
            "episodic": 0.02,
            "procedural": 0.005,
        },
        "default_confidence": 0.7,
        "deduction_confidence_cap": 0.9,
        "deduction_default_confidence": 0.6,
        "hindsight_confidence": 0.8,
        "deduction_decay_k": 0.001,
        "hindsight_decay_k": 0.0005,
    },
    "retrieval": {
        "prefetch_limit": 5,
        "fts5_fetch_multiplier": 4,
        "fts5_fetch_min": 20,
        "semantic_floor": 0.15,
        "score_threshold": 0.15,
        "entity_boost_weight": 0.5,
        "linked_boost_factor": 0.5,
        "retention_access_boost_cap": 1.8,
        "retention_confirmation_factor": 0.08,
        "contradiction_reinforce_overlap": 0.7,
        "contradiction_supersede_overlap": 0.3,
        "frozen_snapshot_limit": 5,
        "always_inject_hindsight_limit": 3,
        "semantic_dedup_threshold": 0.90,
    },
    "buffers": {
        "session_history": 20,
        "recent_texts": 50,
        "existing_dedup": 20,
        "dedup_prompt_existing_cap": 10,
    },
    "consolidation": {
        "check_interval": 600.0,
        "memory_threshold": 10,
        "time_threshold_hours": 4.0,
        "max_memories": 50,
        "induction_min_memories": 4,
        "induction_confidence_base": 0.5,
        "induction_confidence_per_evidence": 0.08,
        "induction_confidence_cap": 0.92,
    },
    "hindsight": {
        "prune_hit_rate": 0.05,
        "prune_min_injections": 30,
        "prune_stale_days": 30,
    },
    "scoring": {
        "bm25": {
            "short": {"midpoint": 5.0, "steepness": 0.7},
            "medium": {"midpoint": 7.0, "steepness": 0.6},
            "long_short": {"midpoint": 9.0, "steepness": 0.5},
            "long_medium": {"midpoint": 10.0, "steepness": 0.5},
            "long_long": {"midpoint": 12.0, "steepness": 0.5},
        },
    },
}


# ---------------------------------------------------------------------------
# Deep merge utility
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*.

    Dict values are merged recursively; all other types in *override*
    replace the corresponding value in *base*.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_config_cache: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Load and cache the configuration.

    1. Start with built-in defaults.
    2. If config.yaml exists, deep-merge its values on top.
    3. Environment variables override individual settings where applicable.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config = _deep_merge({}, _DEFAULTS)

    if _CONFIG_FILE.exists():
        try:
            import yaml
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, file_config)
            logger.debug("[Config] Loaded %s", _CONFIG_FILE)
        except ImportError:
            logger.warning("[Config] PyYAML not installed, using defaults only")
        except Exception as e:
            logger.warning("[Config] Failed to load %s: %s", _CONFIG_FILE, e)
    else:
        logger.debug("[Config] %s not found, using defaults", _CONFIG_FILE)

    # Environment variable overrides for common settings
    env_overrides = {
        "LLM_MODEL": ("model", "llm"),
        "EMBEDDING_MODEL": ("model", "embedding"),
        "LLM_BASE_URL": ("llm", "base_url"),
        "VOICE_LLM_MAX_TOKENS": ("llm", "max_tokens"),
    }
    for env_key, path in env_overrides.items():
        value = os.getenv(env_key)
        if value:
            d = config
            for key in path[:-1]:
                d = d.setdefault(key, {})
            # Cast to appropriate type
            last_key = path[-1]
            existing = d.get(last_key)
            if isinstance(existing, int):
                d[last_key] = int(value)
            elif isinstance(existing, float):
                d[last_key] = float(value)
            else:
                d[last_key] = value

    _config_cache = config
    return config


def reload() -> None:
    """Force a reload of the configuration on next access."""
    global _config_cache
    _config_cache = None


def cfg(key: str, default: Any = None) -> Any:
    """Access a configuration value by dot-separated path.

    Example:
        cfg("model.llm")           → "qwen-flash"
        cfg("memory.scope_decay.episodic")  → 0.02
        cfg("prompts.extraction")  → "..."  (full prompt template)

    Returns *default* if the path does not exist.
    """
    config = _load_config()
    parts = key.split(".")
    current: Any = config
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
            if current is None:
                return default
        else:
            return default
    return current

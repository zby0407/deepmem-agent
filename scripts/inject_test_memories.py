#!/usr/bin/env python3
"""Inject realistic test memories into voice_memory.sqlite3 for hindsight testing.

Scenarios covered:
  1. Contradiction pairs (old vs new fact, same topic)
  2. Decay across scopes (semantic / episodic / procedural)
  3. Confirmation count boosting retention
  4. Hindsight rules (always_inject)
  5. Bi-temporal invalidation (fact was true, now false)
  6. Linked memory clusters
  7. Time-staggered creation to exercise retention formula
"""

import hashlib
import json
import math
import random
import sqlite3
import struct
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "voice_memory.sqlite3"

# --- helpers ---

def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def _fake_embedding(seed: int, dim: int = 1024) -> bytes:
    """Deterministic random unit vector (so FTS+embedding pipeline can still compare)."""
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec))
    vec = [v / norm for v in vec]
    return struct.pack(f"{dim}f", *vec)

def _days_ago(days: int) -> float:
    return time.time() - days * 86400

def _hours_ago(hours: int) -> float:
    return time.time() - hours * 3600

# --- test data definitions ---
# Each entry: (user_id, content, scope, created_days_ago, attrs)
#   attrs is a dict that maps to optional columns.

USER = "browser"  # primary test user (must match frontend user_id)

RECORDS = [
    # -- Scenario 1: Contradiction pair --
    # Old fact (30 days ago)
    {
        "content": "小满最喜欢的颜色是蓝色",
        "scope": "semantic",
        "days_ago": 30,
        "confidence": 0.8,
        "confirmation_count": 3,
        "attributed_to": "user",
    },
    # New fact (2 days ago) -- contradicts #1
    {
        "content": "小满最喜欢的颜色是粉色",
        "scope": "semantic",
        "days_ago": 2,
        "confidence": 0.85,
        "confirmation_count": 0,
        "attributed_to": "user",
        "linked_memory_ids": ["1"],  # will reference rowid 1 after insert
    },
    # Another old fact that should be superseded
    {
        "content": "小满住在北京市朝阳区",
        "scope": "semantic",
        "days_ago": 90,
        "confidence": 0.9,
        "confirmation_count": 5,
        "attributed_to": "user",
    },
    # Current fact -- contradicts #3
    {
        "content": "小满住在南京市鼓楼区",
        "scope": "semantic",
        "days_ago": 10,
        "confidence": 0.95,
        "confirmation_count": 1,
        "attributed_to": "user",
        "linked_memory_ids": ["3"],
    },

    # -- Scenario 2: Decay across scopes --
    # Old semantic -- should retain well (k=0.0005, half-life ~1386d)
    {
        "content": "小满是一名软件工程师",
        "scope": "semantic",
        "days_ago": 120,
        "confidence": 0.95,
        "confirmation_count": 8,
        "attributed_to": "user",
    },
    # Old episodic -- moderate decay (k=0.02, half-life ~35d)
    {
        "content": "小满在学习React前端框架",
        "scope": "episodic",
        "days_ago": 60,
        "confidence": 0.7,
        "confirmation_count": 2,
        "attributed_to": "user",
    },
    # Old procedural -- slow decay (k=0.005, half-life ~139d)
    {
        "content": "小满喜欢在早上喝拿铁咖啡",
        "scope": "procedural",
        "days_ago": 20,
        "confidence": 0.6,
        "confirmation_count": 1,
        "attributed_to": "user",
    },
    # Recent preference -- should score high due to freshness
    {
        "content": "小满最近改喝美式咖啡了",
        "scope": "procedural",
        "days_ago": 3,
        "confidence": 0.75,
        "confirmation_count": 0,
        "attributed_to": "user",
    },

    # -- Scenario 3: High confirmation count --
    {
        "content": "小满喜欢在周末去公园跑步",
        "scope": "procedural",
        "days_ago": 45,
        "confidence": 0.85,
        "confirmation_count": 12,
        "attributed_to": "user",
    },
    {
        "content": "小满的生日是8月15日",
        "scope": "semantic",
        "days_ago": 100,
        "confidence": 0.99,
        "confirmation_count": 15,
        "attributed_to": "user",
    },

    # -- Scenario 4: Hindsight rules --
    {
        "content": "当用户提到出行或天气时，助手应主动提供天气预报和穿衣建议",
        "scope": "procedural",
        "days_ago": 15,
        "confidence": 0.9,
        "confirmation_count": 0,
        "attributed_to": "assistant",
        "metadata": json.dumps({
            "kind": "hindsight_rule",
            "use_mode": "always_inject",
            "hit_count": 7,
            "injected_count": 25,
        }),
        "decay_k": 0.0005,
    },
    {
        "content": "当用户表达情绪低落时，助手应先共情再提供建议，不要直接给解决方案",
        "scope": "procedural",
        "days_ago": 40,
        "confidence": 0.85,
        "confirmation_count": 0,
        "attributed_to": "assistant",
        "metadata": json.dumps({
            "kind": "hindsight_rule",
            "use_mode": "always_inject",
            "hit_count": 3,
            "injected_count": 50,
        }),
        "decay_k": 0.0005,
    },
    # An ineffective hindsight rule (low hit rate, should be pruned)
    {
        "content": "当用户提到数字时，助手应将其转换为中文大写",
        "scope": "procedural",
        "days_ago": 35,
        "confidence": 0.6,
        "confirmation_count": 0,
        "attributed_to": "assistant",
        "metadata": json.dumps({
            "kind": "hindsight_rule",
            "use_mode": "always_inject",
            "hit_count": 0,
            "injected_count": 40,
        }),
        "decay_k": 0.0005,
    },

    # -- Scenario 5: Bi-temporal (invalidated fact) --
    {
        "content": "小满每周三晚上有瑜伽课",
        "scope": "procedural",
        "days_ago": 50,
        "confidence": 0.8,
        "confirmation_count": 6,
        "attributed_to": "user",
        "valid_at": _days_ago(50),
        "invalid_at": _days_ago(10),  # stopped 10 days ago
    },
    {
        "content": "小满改成了每周四晚上上舞蹈课",
        "scope": "procedural",
        "days_ago": 10,
        "confidence": 0.85,
        "confirmation_count": 1,
        "attributed_to": "user",
        "linked_memory_ids": ["14"],
    },

    # -- Scenario 6: Linked memory cluster (travel planning) --
    {
        "content": "小满计划6月中旬去日本旅行",
        "scope": "episodic",
        "days_ago": 5,
        "confidence": 0.9,
        "confirmation_count": 0,
        "attributed_to": "user",
    },
    {
        "content": "小满想去东京和京都两个城市",
        "scope": "episodic",
        "days_ago": 5,
        "confidence": 0.85,
        "confirmation_count": 0,
        "attributed_to": "user",
        "linked_memory_ids": ["16"],
    },
    {
        "content": "小满对日本料理特别期待，尤其是寿司和拉面",
        "scope": "procedural",
        "days_ago": 4,
        "confidence": 0.8,
        "confirmation_count": 0,
        "attributed_to": "user",
        "linked_memory_ids": ["16", "17"],
    },
    {
        "content": "小满的旅行预算在1.5万元左右",
        "scope": "episodic",
        "days_ago": 3,
        "confidence": 0.7,
        "confirmation_count": 0,
        "attributed_to": "user",
        "linked_memory_ids": ["16", "17"],
    },
    {
        "content": "小满需要办理日本签证，护照还有8个月有效期",
        "scope": "episodic",
        "days_ago": 2,
        "confidence": 0.9,
        "confirmation_count": 0,
        "attributed_to": "user",
        "linked_memory_ids": ["16"],
    },

    # -- Scenario 7: Habit pattern memories --
    {
        "content": "小满每天晚上11点前睡觉",
        "scope": "procedural",
        "days_ago": 30,
        "confidence": 0.75,
        "confirmation_count": 8,
        "attributed_to": "assistant",
    },
    {
        "content": "小满习惯用番茄工作法管理时间",
        "scope": "procedural",
        "days_ago": 25,
        "confidence": 0.7,
        "confirmation_count": 4,
        "attributed_to": "user",
    },
    {
        "content": "小满喜欢在下雨天听爵士乐",
        "scope": "procedural",
        "days_ago": 15,
        "confidence": 0.65,
        "confirmation_count": 2,
        "attributed_to": "user",
    },
    {
        "content": "小满养了一只叫豆豆的橘猫",
        "scope": "semantic",
        "days_ago": 80,
        "confidence": 0.95,
        "confirmation_count": 10,
        "attributed_to": "user",
    },
    {
        "content": "小满的猫豆豆今年3岁了",
        "scope": "semantic",
        "days_ago": 60,
        "confidence": 0.8,
        "confirmation_count": 3,
        "attributed_to": "user",
        "linked_memory_ids": ["24"],
    },

    # -- Scenario 8: Recent conversation context --
    {
        "content": "小满最近在准备下周的项目汇报",
        "scope": "episodic",
        "days_ago": 1,
        "confidence": 0.85,
        "confirmation_count": 0,
        "attributed_to": "user",
    },
    {
        "content": "小满希望助手帮忙练习英语口语",
        "scope": "episodic",
        "days_ago": 1,
        "confidence": 0.8,
        "confirmation_count": 0,
        "attributed_to": "user",
    },
    {
        "content": "小满觉得PPT制作很头疼",
        "scope": "procedural",
        "days_ago": 1,
        "confidence": 0.7,
        "confirmation_count": 0,
        "attributed_to": "user",
    },
    {
        "content": "小满喜欢用简洁风格的PPT模板",
        "scope": "procedural",
        "days_ago": 0.5,
        "confidence": 0.75,
        "confirmation_count": 0,
        "attributed_to": "user",
    },
]

# -- History records for key events --
# (memory_id, event, old_memory, new_memory, created_at_offset_days)
HISTORY_EVENTS = [
    # Contradiction superseded
    (1, "SUPERSEDE", "小满最喜欢的颜色是蓝色", "小满最喜欢的颜色是粉色", 2),
    (3, "SUPERSEDE", "小满住在北京市朝阳区", "小满住在南京市鼓楼区", 10),
    # Activity change
    (14, "CONSOLIDATE_INVALIDATE", "小满每周三晚上有瑜伽课", "小满改成了每周四晚上上舞蹈课", 10),
    # Reinforcements
    (5, "REINFORCE", None, "小满是一名软件工程师", 90),
    (5, "REINFORCE", None, "小满是一名软件工程师", 60),
    (5, "REINFORCE", None, "小满是一名软件工程师", 30),
    (9, "REINFORCE", None, "小满喜欢在周末去公园跑步", 30),
    (9, "REINFORCE", None, "小满喜欢在周末去公园跑步", 20),
    (10, "REINFORCE", None, "小满的生日是8月15日", 80),
    (10, "REINFORCE", None, "小满的生日是8月15日", 50),
    (24, "REINFORCE", None, "小满养了一只叫豆豆的橘猫", 60),
    (24, "REINFORCE", None, "小满养了一只叫豆豆的橘猫", 40),
    # Hindsight rule pruning candidate
    (13, "HINDSIGHT_PRUNE", None, "当用户提到数字时，助手应将其转换为中文大写", 0),
]


def inject():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    existing = conn.execute("SELECT MAX(id) FROM voice_memories").fetchone()[0] or 0
    print(f"Existing max id: {existing}, inserting {len(RECORDS)} new records...")

    now = time.time()
    inserted_ids = {}

    for i, rec in enumerate(RECORDS):
        content = rec["content"]
        scope = rec["scope"]
        created_at = _days_ago(rec["days_ago"])
        updated_at = created_at + random.uniform(0, 3600)
        confidence = rec.get("confidence", 0.7)
        confirmation_count = rec.get("confirmation_count", 0)
        attributed_to = rec.get("attributed_to", "user")
        metadata = rec.get("metadata", "{}")
        decay_k = rec.get("decay_k",
                          0.0005 if scope == "semantic"
                          else 0.005 if scope == "procedural"
                          else 0.02)
        valid_at = rec.get("valid_at", created_at)
        invalid_at = rec.get("invalid_at")
        hash_val = _md5(content)

        # Generate deterministic fake embedding from content hash
        seed = int(hash_val[:8], 16)
        embedding = _fake_embedding(seed)

        # Resolve linked_memory_ids (placeholder indices -> actual rowids)
        raw_links = rec.get("linked_memory_ids")
        if raw_links:
            resolved = []
            for ref in raw_links:
                idx = int(ref)
                if idx in inserted_ids:
                    resolved.append(str(inserted_ids[idx]))
                else:
                    resolved.append(str(existing + idx))  # fallback
            linked = json.dumps(resolved)
        else:
            linked = None

        last_used_at = _days_ago(max(0, rec["days_ago"] - random.uniform(0, 2)))

        cursor = conn.execute(
            """INSERT INTO voice_memories
               (user_id, content, scope, created_at, updated_at, metadata, hash,
                embedding, linked_memory_ids, attributed_to, confidence,
                confirmation_count, last_used_at, decay_k, valid_at, invalid_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (USER, content, scope, created_at, updated_at, metadata, hash_val,
             embedding, linked, attributed_to, confidence,
             confirmation_count, last_used_at, decay_k, valid_at, invalid_at),
        )
        rowid = cursor.lastrowid
        inserted_ids[i + 1] = rowid
        print(f"  [{rowid}] ({scope:12s}) {content[:40]}...")

    conn.commit()

    # -- Insert history events --
    print(f"\nInserting {len(HISTORY_EVENTS)} history events...")
    for mem_idx, event, old_mem, new_mem, offset_days in HISTORY_EVENTS:
        actual_id = inserted_ids.get(mem_idx, existing + mem_idx)
        created_at = _days_ago(offset_days + 1)
        updated_at = _days_ago(offset_days)
        conn.execute(
            """INSERT OR IGNORE INTO voice_memory_history
               (id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), str(actual_id), old_mem, new_mem, event,
             created_at, updated_at, 1 if event in ("SUPERSEDE", "CONSOLIDATE_INVALIDATE") else 0),
        )
    conn.commit()

    # -- Summary --
    total = conn.execute("SELECT COUNT(*) FROM voice_memories").fetchone()[0]
    by_scope = conn.execute(
        "SELECT scope, COUNT(*) as cnt FROM voice_memories GROUP BY scope"
    ).fetchall()
    history_count = conn.execute("SELECT COUNT(*) FROM voice_memory_history").fetchone()[0]

    print(f"\n=== Done ===")
    print(f"Total memories: {total}")
    for row in by_scope:
        print(f"  {row['scope']:15s}: {row['cnt']}")
    print(f"History events: {history_count}")

    # Print retention preview
    print(f"\n=== Retention Preview (as of now) ===")
    rows = conn.execute(
        "SELECT id, content, scope, confidence, confirmation_count, decay_k, created_at FROM voice_memories WHERE user_id=? ORDER BY id",
        (USER,),
    ).fetchall()
    for r in rows:
        age_days = (now - r["created_at"]) / 86400
        retention = math.exp(-r["decay_k"] * age_days) * min(1.8, 1 + math.log(1 + r["confirmation_count"]) * 0.08)
        print(f"  [{r['id']:3d}] retention={retention:.3f}  conf={r['confidence']:.2f}  ({r['scope']:12s}) {r['content'][:35]}")

    conn.close()


if __name__ == "__main__":
    inject()

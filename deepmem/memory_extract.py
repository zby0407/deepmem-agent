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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# V3 Additive Extraction Prompt — from mem0 ADDITIVE_EXTRACTION_PROMPT
# Simplified for voice runtime (Chinese-first, shorter context)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """你是一个记忆提取器 — 从对话中提取所有值得长期记住的事实，并为每条事实分类。

# 记忆分类（认知科学三分类）

每条记忆必须归入以下三类之一：

- **semantic**（语义记忆）：用户的稳定个人事实。姓名、生日、职业、家庭成员、住址、宠物、长期不会改变的身份信息。特点是"几乎永久有效"。
- **episodic**（情景记忆）：有时间边界的事件、计划、近期活动。出差计划、昨晚的聚餐、正在学什么、下周的会议。特点是"几周到几个月后过时"。
- **procedural**（程序记忆）：习惯、惯例、稳定偏好、行为规律。每天喝咖啡、周末跑步、喜欢简洁风格、不吃辣。特点是"一旦形成，长期保持"。

判定规则：
- 如果这条信息一年后大概率还成立 → semantic
- 如果这条信息有明确时间窗口或已过期 → episodic
- 如果这条信息描述的是重复行为或稳定偏好 → procedural

# 规则

1. 从用户和助手的消息中提取事实。用户消息包含个人信息、偏好、计划；助手消息包含建议、方案、研究的信息。
2. 事实必须自包含（用"用户"代替"你/我"），15-80字，上下文丰富
3. 所有相对时间必须转为绝对日期（观察日期：{observation_date}）
4. 跳过问候、寒暄、无信息量的回复
5. 提取所有维度：个人信息、偏好、计划、习惯、关系、经历、建议
6. 如果新信息与已有记忆语义等价且无新内容，跳过
7. 当新记忆与已有记忆相关时（同一主题、更新偏好、后续事件），在 linked_memory_ids 中引用已有记忆的 ID

# 已有记忆
{existing_memories}

# 最近已提取的记忆（会话内去重参考）
{recently_extracted}

# 最近对话历史（用于指代消解）
{last_k_messages}

# 新对话
{new_messages}

# 观察日期
{observation_date}

# 穷举提取清单
输出前检查：
1. 是否从对话中每个不同主题都提取了记忆？
2. 是否提取了对话中段和末段的信息，而不是只提取开头？
3. 每条用户消息中的每个具体事实是否都有对应提取？
4. 每条记忆是否都正确分类为 semantic/episodic/procedural？

# 输出格式
只返回 JSON，不要其他文字：
{{"memory": [
  {{"id": "0", "text": "事实内容", "scope": "semantic", "attributed_to": "user", "linked_memory_ids": []}},
  {{"id": "1", "text": "事实内容", "scope": "episodic", "attributed_to": "assistant"}}
]}}
如果没有值得提取的事实：{{"memory": []}}

# 示例

## 示例1：多主题提取（含分类）
新对话：
用户：我叫小满，刚升职为产品经理。我老婆小红和我昨天去了海底捞庆祝。
助手：恭喜！海底捞是个庆祝的好地方。

输出：
{{"memory": [
  {{"id": "0", "text": "用户名叫小满，刚升职为产品经理", "scope": "semantic", "attributed_to": "user"}},
  {{"id": "1", "text": "用户的妻子叫小红，他们昨天去海底捞庆祝升职", "scope": "episodic", "attributed_to": "user"}}
]}}

## 示例2：偏好 = procedural
新对话：
用户：能推荐一些好看的科幻电影吗？我喜欢《星际穿越》。
助手：推荐《降临》《银翼杀手2049》《火星救援》，都是口碑很好的硬科幻。

输出：
{{"memory": [
  {{"id": "0", "text": "用户喜欢看科幻电影，尤其喜欢《星际穿越》", "scope": "procedural", "attributed_to": "user"}},
  {{"id": "1", "text": "用户被推荐了《降临》《银翼杀手2049》《火星救援》等硬科幻电影", "scope": "episodic", "attributed_to": "assistant"}}
]}}

## 示例3：无需提取
新对话：
用户：早上好！
助手：早上好！有什么需要帮忙的吗？

输出：{{"memory": []}}

## 示例4：去重 — 跳过已存在的
已有记忆：[{{"id": "0", "text": "用户叫小满，是产品经理"}}]
新对话：
用户：我之前说过我是产品经理对吧？

输出：{{"memory": []}}

## 示例5：绝对时间 + 习惯 = procedural
新对话：
用户：我下周三要去北京出差。我习惯出差前一晚整理行李。
（假设今天是2026-05-29）

输出：
{{"memory": [
  {{"id": "0", "text": "用户2026年6月3日要去北京出差", "scope": "episodic", "attributed_to": "user"}},
  {{"id": "1", "text": "用户习惯出差前一晚整理行李", "scope": "procedural", "attributed_to": "user"}}
]}}
"""


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
        lines = [f'[{{"id": "{m["id"]}", "text": "{m["text"]}"}}]' for m in existing_memories[:10]]
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
    threshold: float = 0.90,
) -> list[str]:
    """Remove semantically similar duplicates using embedding similarity.

    Returns only new_texts that are NOT semantically equivalent to existing memories.
    """
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

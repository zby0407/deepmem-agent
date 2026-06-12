"""Benchmark LocalExternalMemoryProvider against LOCOMO and LongMemEval.

Usage:
    python benchmarks/locomo_benchmark.py --benchmark locomo --limit 5
    python benchmarks/locomo_benchmark.py --benchmark longmemeval --limit 10
    python benchmarks/locomo_benchmark.py --benchmark both --limit 3 --output results.json
    python benchmarks/locomo_benchmark.py --benchmark locomo --limit 1 --max-turns 50  # fast test
    python benchmarks/locomo_benchmark.py --benchmark locomo --data-dir /path/to/data
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

# Add parent dir to path for `deepmem` package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from deepmem.local_memory import LocalExternalMemoryProvider
from deepmem.embedding import EmbeddingClient
from deepmem.llm import build_llm_client

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "the", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "you",
}

ANSWER_PROMPT = """Based on the following memories, answer the question. Be extremely concise.

Memories:
{memories}

Question: {question}

Rules:
- Answer in 1-5 words maximum
- Use exact names, dates, places from the memories
- For dates: use the format from the memories (e.g. "7 May 2023", "2022", "June 2023")
- For "when" questions: give the specific date/time mentioned
- If unsure or not found: output "unknown"
- Output ONLY the answer, nothing else"""

JUDGE_PROMPT = """You are evaluating whether a predicted answer is correct.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Rules:
- Accept different phrasing if the meaning is preserved
- Accept a more detailed answer if it contains the reference answer
- Accept abbreviations, synonyms, or paraphrases
- For dates: accept equivalent formats (e.g. "7 May 2023" = "May 7, 2023" = "May 7th 2023")
- For names: accept partial matches if the key name is correct
- If the predicted answer is "unknown" or empty, it's INCORRECT

Output ONLY one word: CORRECT or INCORRECT"""


def token_set(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 1 and t not in STOPWORDS
    }


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = token_set(prediction)
    ref_tokens = token_set(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    pred_norm = re.sub(r"[,\s]+", " ", prediction.lower().strip()).strip()
    ref_norm = re.sub(r"[,\s]+", " ", reference.lower().strip()).strip()
    if ref_norm in pred_norm or pred_norm in ref_norm:
        return 1.0
    pred_tokens = token_set(prediction)
    ref_tokens = token_set(reference)
    if not ref_tokens:
        return 0.0
    if ref_tokens.issubset(pred_tokens):
        return 1.0
    return 0.0


def evidence_hit(retrieved_contents: list[str], gold_texts: list[str]) -> float:
    if not gold_texts:
        return 0.0
    for gold in gold_texts:
        gold_tokens = token_set(gold)
        if not gold_tokens:
            continue
        for content in retrieved_contents:
            content_tokens = token_set(content)
            overlap = len(gold_tokens & content_tokens)
            if overlap >= len(gold_tokens) * 0.5:
                return 1.0
    return 0.0


async def judge_answer(llm_caller, question: str, reference: str, prediction: str) -> float:
    if not prediction or prediction.lower() in ("unknown", ""):
        return 0.0
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, prediction=prediction)
    try:
        result = await llm_caller(prompt)
        return 1.0 if "correct" in result.lower().strip() else 0.0
    except Exception:
        return 0.0


async def generate_answer(llm_caller, memories: list[str], question: str) -> str:
    if not memories:
        return "unknown"
    memories_text = "\n".join(f"- {m}" for m in memories)
    prompt = ANSWER_PROMPT.format(memories=memories_text, question=question)
    try:
        result = await llm_caller(prompt)
        return result.strip().strip('"').strip("'")
    except Exception:
        return "unknown"


@dataclass
class BenchmarkResult:
    benchmark: str
    sample_id: str
    question_id: str
    question_type: str
    question: str
    reference_answer: str
    generated_answer: str
    em: float
    f1: float
    judge_score: float
    evidence_hit: float
    memories_retrieved: int
    latency_ms: float


async def make_llm_caller():
    client = build_llm_client()

    async def call_llm(prompt: str) -> str:
        result = await client.chat_with_tools(
            [{"role": "user", "content": prompt}],
            model=client.model,
        )
        return result.get("content") or ""

    return call_llm


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


async def batch_extract_and_store(
    provider: LocalExternalMemoryProvider,
    turn_pairs: list[tuple[str, str]],
    batch_size: int = 10,
) -> int:
    """Batch multiple turns into fewer LLM calls for extraction.

    Instead of 1 LLM call per turn, we batch batch_size turns into a single
    extraction call. This reduces API calls from N to N/batch_size, and gives
    the LLM more context for better extraction quality.

    Context compression: the extraction prompt caps existing memories at 10
    and recently_extracted at 20, so the prompt stays bounded regardless of
    how many total memories exist.
    """
    import re as _re
    from deepmem.memory_extract import EXTRACTION_PROMPT, _md5
    from deepmem.local_memory import _encode_embedding
    from datetime import date as _date

    today = _date.today().strftime("%Y-%m-%d")
    total_added = 0

    def _parse_json_from_llm(raw: str) -> Any:
        text = raw.strip()
        if text.startswith("```"):
            text = _re.sub(r"^```(?:json)?", "", text).strip()
            text = _re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for s, e in [("{", "}"), ("[", "]")]:
            start = text.find(s)
            end = text.rfind(e)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return None

    for batch_start in range(0, len(turn_pairs), batch_size):
        batch = turn_pairs[batch_start:batch_start + batch_size]

        # Format batch as a multi-turn JSON array
        new_msgs = []
        for user_content, assistant_content in batch:
            if user_content:
                new_msgs.append(f'{{"role": "user", "content": "{_truncate(user_content, 300)}"}}')
            if assistant_content:
                new_msgs.append(f'{{"role": "assistant", "content": "{_truncate(assistant_content, 300)}"}}')

        if not new_msgs:
            continue
        new_messages_str = f'[{", ".join(new_msgs)}]'

        # Get dedup context (bounded: top 10 existing, last 20 recent)
        existing = provider._get_existing_for_dedup(limit=20)
        existing_hashes = {m["hash"] for m in existing if m.get("hash")}
        existing_str = "（无）"
        if existing:
            lines = [f'[{{"id": "{m["id"]}", "text": "{m["content"][:100]}"}}]' for m in existing[:10]]
            existing_str = "\n".join(lines)

        recently_str = "（无）"
        if provider._recent_texts:
            recently_str = "\n".join(f"- {r}" for r in provider._recent_texts[-20:])

        last_k_str = "（无）"
        if provider._session_history:
            lines = []
            for msg in provider._session_history[-10:]:
                role = msg.get("role", "")
                content = msg.get("content", "")[:200]
                if role and content:
                    lines.append(f"{role}: {content}")
            if lines:
                last_k_str = "\n".join(lines)

        # Build prompt directly (bypass extract_and_dedup to avoid JSON double-wrapping)
        prompt = EXTRACTION_PROMPT.format(
            observation_date=today,
            existing_memories=existing_str,
            recently_extracted=recently_str,
            last_k_messages=last_k_str,
            new_messages=new_messages_str,
        )

        # Call LLM
        try:
            result = await asyncio.wait_for(
                provider._llm.chat_with_tools(
                    [{"role": "user", "content": prompt}],
                    model=provider._model,
                ),
                timeout=45,
            )
            content = result.get("content") or ""
            parsed = _parse_json_from_llm(content)
            memories = parsed.get("memory", []) if parsed else []
            if not isinstance(memories, list):
                memories = []
        except asyncio.TimeoutError:
            print(f"    [WARN] Batch {batch_start//batch_size+1} timed out", flush=True)
            continue
        except Exception as e:
            print(f"    [WARN] Batch {batch_start//batch_size+1} failed: {e}", flush=True)
            continue

        if not memories:
            continue

        # Store extracted facts
        import time as _time
        batch_added = 0
        for mem in memories:
            if not isinstance(mem, dict):
                continue
            text = (mem.get("text") or "").strip()
            if not text or len(text) < 10:
                continue
            h = _md5(text)
            if h in existing_hashes or h in provider._recent_hashes:
                continue
            now = _time.time()
            try:
                cursor = provider._conn.execute(
                    """INSERT INTO voice_memories
                       (user_id, content, scope, created_at, updated_at, hash,
                        linked_memory_ids, attributed_to, metadata,
                        confidence, confirmation_count, last_used_at, decay_k,
                        valid_at, invalid_at, expired_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, NULL, NULL)""",
                    (provider._user_id, text, "learning", now, now, h,
                     json.dumps(mem.get("linked_memory_ids", [])),
                     mem.get("attributed_to", "user"), "{}",
                     mem.get("confidence", 0.7), now, 0.02, now),
                )
                provider._conn.commit()
                mem_id = cursor.lastrowid
                provider._recent_hashes.add(h)
                provider._recent_texts.append(text)
                total_added += 1
                batch_added += 1

                # Embed
                if provider._embedder:
                    try:
                        vec = provider._embedder.embed_single(text)
                        if vec and not all(v == 0 for v in vec):
                            blob = _encode_embedding(vec)
                            provider._conn.execute(
                                "UPDATE voice_memories SET embedding = ? WHERE id = ?",
                                (blob, mem_id),
                            )
                            provider._conn.commit()
                    except Exception:
                        pass
            except Exception:
                pass

        batch_num = batch_start // batch_size + 1
        total_batches = (len(turn_pairs) + batch_size - 1) // batch_size
        print(f"    Batch {batch_num}/{total_batches}: {batch_added} memories", flush=True)

        if len(provider._recent_texts) > 50:
            provider._recent_texts = provider._recent_texts[-50:]

    return total_added


async def ingest_locomo_conversations(
    provider: LocalExternalMemoryProvider,
    sample: dict,
    max_turns: int | None = None,
) -> set[str]:
    """Ingest all sessions from a LoCoMo sample into the memory provider.

    Returns set of speaker names found.
    """
    conversation = sample.get("conversation", {})
    speakers: set[str] = set()
    turn_pairs: list[tuple[str, str]] = []

    for key, turns in conversation.items():
        if not re.fullmatch(r"session_\d+", str(key)) or not isinstance(turns, list):
            continue
        session_date = conversation.get(f"{key}_date_time")

        i = 0
        while i < len(turns):
            turn = turns[i]
            speaker = str(turn.get("speaker", "unknown")).strip()
            text = " ".join(str(turn.get("text", "")).split())
            if not text:
                i += 1
                continue
            speakers.add(speaker)

            if session_date:
                text = f"[Date: {session_date}] {text}"

            if i + 1 < len(turns):
                next_turn = turns[i + 1]
                next_speaker = str(next_turn.get("speaker", "unknown")).strip()
                next_text = " ".join(str(next_turn.get("text", "")).split())
                if next_speaker != speaker and next_text:
                    if session_date:
                        next_text = f"[Date: {session_date}] {next_text}"
                    turn_pairs.append((f"{speaker}: {text}", f"{next_speaker}: {next_text}"))
                    speakers.add(next_speaker)
                    i += 2
                    continue

            turn_pairs.append((f"{speaker}: {text}", ""))
            i += 1

    if max_turns:
        turn_pairs = turn_pairs[:max_turns]

    total = len(turn_pairs)
    for idx, (user_content, assistant_content) in enumerate(turn_pairs):
        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"  Ingesting turn {idx + 1}/{total}...", flush=True)
        try:
            await asyncio.wait_for(
                provider._extract_and_store(user_content, assistant_content),
                timeout=30,
            )
        except asyncio.TimeoutError:
            print(f"  [WARN] Turn {idx + 1} extraction timed out", flush=True)
        except Exception as e:
            print(f"  [WARN] Turn {idx + 1} extraction failed: {e}", flush=True)
        await asyncio.sleep(0.3)

    return speakers


async def ingest_longmemeval_sessions(
    provider: LocalExternalMemoryProvider,
    sample: dict,
    max_turns: int | None = None,
):
    """Ingest all sessions from a LongMemEval sample."""
    sessions = sample.get("haystack_sessions", [])
    turn_pairs: list[tuple[str, str]] = []

    for session_msgs in sessions:
        i = 0
        while i < len(session_msgs):
            msg = session_msgs[i]
            role = msg.get("role", "user")
            content = " ".join(str(msg.get("content", "")).split())
            if not content:
                i += 1
                continue

            if i + 1 < len(session_msgs):
                next_msg = session_msgs[i + 1]
                next_role = next_msg.get("role", "assistant")
                next_content = " ".join(str(next_msg.get("content", "")).split())
                if next_role != role and next_content:
                    if role == "user":
                        turn_pairs.append((content, next_content))
                    else:
                        turn_pairs.append((next_content, content))
                    i += 2
                    continue

            if role == "user":
                turn_pairs.append((content, ""))
            else:
                turn_pairs.append(("", content))
            i += 1

    if max_turns:
        turn_pairs = turn_pairs[:max_turns]

    total = len(turn_pairs)
    for idx, (user_content, assistant_content) in enumerate(turn_pairs):
        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"  Ingesting turn {idx + 1}/{total}...", flush=True)
        try:
            await asyncio.wait_for(
                provider._extract_and_store(user_content, assistant_content),
                timeout=30,
            )
        except asyncio.TimeoutError:
            print(f"  [WARN] Turn {idx + 1} extraction timed out", flush=True)
        except Exception:
            pass
        await asyncio.sleep(0.3)


def retrieve_memories(provider: LocalExternalMemoryProvider, query: str, limit: int = 20) -> list[str]:
    """Retrieve memories with deduplication, using a higher limit for better recall."""
    results = provider._hybrid_search(query, limit=limit)
    return [r["content"] for r in results]


async def retrieve_with_expansion(
    llm_caller,
    provider: LocalExternalMemoryProvider,
    question: str,
    limit: int = 20,
) -> list[str]:
    """Multi-query retrieval: decompose complex questions into sub-queries.

    For multi-hop questions like "What did X do after meeting Y?",
    we generate sub-queries: "X meeting Y", "X activity after meeting Y"
    and merge results.
    """
    # Original query
    results = provider._hybrid_search(question, limit=limit)
    seen = {r["content"] for r in results}
    all_contents = [r["content"] for r in results]

    # Generate sub-queries for multi-hop questions
    decompose_prompt = f"""Generate 2-3 search queries to find information for this question.
Each query should focus on a different aspect of the question.
Output one query per line, nothing else.

Question: {question}"""

    try:
        sub_queries_text = await asyncio.wait_for(llm_caller(decompose_prompt), timeout=10)
        sub_queries = [q.strip() for q in sub_queries_text.strip().split("\n") if q.strip() and len(q.strip()) > 5]
    except Exception:
        sub_queries = []

    for sq in sub_queries[:3]:
        sub_results = provider._hybrid_search(sq, limit=5)
        for r in sub_results:
            if r["content"] not in seen:
                seen.add(r["content"])
                all_contents.append(r["content"])

    return all_contents[:limit * 2]


async def _eval_one_qa(
    llm_caller,
    provider: LocalExternalMemoryProvider,
    sem: asyncio.Semaphore,
    sample_id: str,
    qa_idx: int,
    qa: dict,
) -> BenchmarkResult | None:
    question = str(qa.get("question", ""))
    answer = str(qa.get("answer", ""))
    category = str(qa.get("category", "unknown"))
    if not question or not answer:
        return None

    async with sem:
        start = time.time()
        memories = await retrieve_with_expansion(llm_caller, provider, question)
        latency = (time.time() - start) * 1000

        generated = await generate_answer(llm_caller, memories, question)
        em = exact_match(generated, answer)
        f1 = f1_score(generated, answer)
        judge = await judge_answer(llm_caller, question, answer, generated)
        ev_hit = evidence_hit(memories, [answer])

    return BenchmarkResult(
        benchmark="locomo",
        sample_id=sample_id,
        question_id=f"{sample_id}:{qa_idx}",
        question_type=category,
        question=question,
        reference_answer=answer,
        generated_answer=generated[:200],
        em=em, f1=f1, judge_score=judge, evidence_hit=ev_hit,
        memories_retrieved=len(memories),
        latency_ms=latency,
    )


async def _run_one_locomo_sample(
    llm_caller,
    sample: dict,
    sample_idx: int,
    total: int,
    qa_limit: int | None,
    max_turns: int | None,
    qa_concurrency: int = 5,
) -> list[BenchmarkResult]:
    """Ingest + evaluate a single LOCOMO sample."""
    sample_id = str(sample.get("sample_id", sample_idx))
    print(f"\n[LOCOMO] Sample {sample_idx + 1}/{total}: {sample_id}", flush=True)

    db_path = tempfile.mktemp(suffix=".sqlite3")
    embedder = EmbeddingClient()
    llm_client = build_llm_client()
    provider = LocalExternalMemoryProvider(
        user_id=f"locomo_{sample_id}",
        db_path=db_path,
        llm_client=llm_client,
        embedding_client=embedder,
        skip_embedding=True,
    )
    provider.initialize(session_id=f"locomo_{sample_id}")

    t0 = time.time()
    speakers = await ingest_locomo_conversations(provider, sample, max_turns)
    ingest_time = time.time() - t0
    mem_count = provider._conn.execute(
        "SELECT COUNT(*) FROM voice_memories WHERE user_id = ? AND invalid_at IS NULL",
        (f"locomo_{sample_id}",),
    ).fetchone()[0]
    print(f"  [{sample_id}] Ingested in {ingest_time:.1f}s, {mem_count} memories stored", flush=True)

    t1 = time.time()
    embedded = provider._backfill_embeddings()
    backfill_time = time.time() - t1
    print(f"  [{sample_id}] Backfilled {embedded} embeddings in {backfill_time:.1f}s", flush=True)

    qas = (sample.get("qa") or [])[:qa_limit]
    sem = asyncio.Semaphore(qa_concurrency)
    tasks = [
        _eval_one_qa(llm_caller, provider, sem, sample_id, qa_idx, qa)
        for qa_idx, qa in enumerate(qas)
    ]
    qa_results = await asyncio.gather(*tasks)
    results = [r for r in qa_results if r is not None]
    print(f"  [{sample_id}] Evaluated {len(qas)} questions", flush=True)
    provider.shutdown()
    return results


async def run_locomo(
    llm_caller,
    data: list[dict],
    limit: int,
    qa_limit: int | None,
    max_turns: int | None,
    concurrency: int = 5,
    sample_concurrency: int = 1,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    samples = data[:limit]
    total = len(samples)
    sample_sem = asyncio.Semaphore(sample_concurrency)

    async def _run_with_sem(idx, sample):
        async with sample_sem:
            return await _run_one_locomo_sample(
                llm_caller, sample, idx, total, qa_limit, max_turns, concurrency,
            )

    tasks = [_run_with_sem(i, s) for i, s in enumerate(samples)]
    sample_results = await asyncio.gather(*tasks)
    for sr in sample_results:
        results.extend(sr)
    return results


async def run_longmemeval(
    llm_caller,
    data: list[dict],
    limit: int,
    max_turns: int | None,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for sample_idx, sample in enumerate(data[:limit]):
        question_id = str(sample.get("question_id", sample_idx))
        print(f"\n[LongMemEval] Sample {sample_idx + 1}/{min(limit, len(data))}: {question_id}", flush=True)

        db_path = tempfile.mktemp(suffix=".sqlite3")
        embedder = EmbeddingClient()
        llm_client = build_llm_client()
        provider = LocalExternalMemoryProvider(
            user_id=f"longmem_{question_id}",
            db_path=db_path,
            llm_client=llm_client,
            embedding_client=embedder,
            skip_embedding=True,
        )
        provider.initialize(session_id=f"longmem_{question_id}")

        t0 = time.time()
        await ingest_longmemeval_sessions(provider, sample, max_turns)
        ingest_time = time.time() - t0
        mem_count = provider._conn.execute(
            "SELECT COUNT(*) FROM voice_memories WHERE user_id = ? AND invalid_at IS NULL",
            (f"longmem_{question_id}",),
        ).fetchone()[0]
        print(f"  Ingested in {ingest_time:.1f}s, {mem_count} memories stored", flush=True)

        # Backfill embeddings after ingestion
        t1 = time.time()
        embedded = provider._backfill_embeddings()
        backfill_time = time.time() - t1
        print(f"  Backfilled {embedded} embeddings in {backfill_time:.1f}s", flush=True)

        question = str(sample.get("question", ""))
        answer = str(sample.get("answer", ""))
        q_type = str(sample.get("question_type", "unknown"))
        if not question or not answer:
            continue

        start = time.time()
        memories = await retrieve_with_expansion(llm_caller, provider, question)
        latency = (time.time() - start) * 1000

        generated = await generate_answer(llm_caller, memories, question)
        em = exact_match(generated, answer)
        f1 = f1_score(generated, answer)
        judge = await judge_answer(llm_caller, question, answer, generated)
        ev_hit = evidence_hit(memories, [answer])

        results.append(BenchmarkResult(
            benchmark="longmemeval",
            sample_id=question_id,
            question_id=question_id,
            question_type=q_type,
            question=question,
            reference_answer=answer,
            generated_answer=generated[:200],
            em=em, f1=f1, judge_score=judge, evidence_hit=ev_hit,
            memories_retrieved=len(memories),
            latency_ms=latency,
        ))

        provider.shutdown()

    return results


def print_summary(results: list[BenchmarkResult], name: str):
    if not results:
        print(f"\n{name}: No results", flush=True)
        return

    em_scores = [r.em for r in results]
    f1_scores = [r.f1 for r in results]
    judge_scores = [r.judge_score for r in results]
    ev_scores = [r.evidence_hit for r in results]
    latencies = [r.latency_ms for r in results]
    mem_counts = [r.memories_retrieved for r in results]

    print(f"\n{'=' * 60}", flush=True)
    print(f"  {name} Results ({len(results)} questions)", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Exact Match:      {mean(em_scores):.3f}", flush=True)
    print(f"  F1 Score:         {mean(f1_scores):.3f}", flush=True)
    print(f"  Judge Score:      {mean(judge_scores):.3f}", flush=True)
    print(f"  Evidence Hit:     {mean(ev_scores):.3f}", flush=True)
    print(f"  Avg Memories:     {mean(mem_counts):.1f}", flush=True)
    print(f"  Avg Latency:      {mean(latencies):.0f}ms", flush=True)
    print(f"{'=' * 60}", flush=True)

    types: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        types.setdefault(r.question_type, []).append(r)

    if len(types) > 1:
        print(f"\n  By Type:", flush=True)
        for qtype, type_results in sorted(types.items()):
            t_em = mean([r.em for r in type_results])
            t_f1 = mean([r.f1 for r in type_results])
            t_judge = mean([r.judge_score for r in type_results])
            t_ev = mean([r.evidence_hit for r in type_results])
            print(f"    {qtype:20s}  EM={t_em:.3f}  F1={t_f1:.3f}  Judge={t_judge:.3f}  Hit={t_ev:.3f}  (n={len(type_results)})", flush=True)


def main():
    parser = argparse.ArgumentParser(description="LocalExternalMemoryProvider Benchmark")
    parser.add_argument("--benchmark", choices=["locomo", "longmemeval", "both"], default="both")
    parser.add_argument("--limit", type=int, default=5, help="Max samples to evaluate")
    parser.add_argument("--qa-limit", type=int, default=None, help="Max QA per LOCOMO sample")
    parser.add_argument("--max-turns", type=int, default=None, help="Max turns to ingest per sample (for fast testing)")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent LLM calls for QA evaluation")
    parser.add_argument("--sample-concurrency", type=int, default=1, help="Number of LOCOMO samples to process in parallel")
    parser.add_argument("--sample-ids", type=str, default=None, help="Comma-separated sample IDs to run (e.g. conv-49,conv-50)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory containing locomo/ and longmemeval/ subdirs")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent.parent
    if args.data_dir:
        data_root = Path(args.data_dir)
    else:
        data_root = package_root / "data"

    locomo_path = data_root / "locomo" / "locomo10.json"
    longmem_path = data_root / "longmemeval" / "longmemeval_s.json"

    async def run():
        llm_caller = await make_llm_caller()
        results: list[BenchmarkResult] = []

        if args.benchmark in ("locomo", "both"):
            data = json.loads(locomo_path.read_text())
            if args.sample_ids:
                ids = set(s.strip() for s in args.sample_ids.split(","))
                data = [s for s in data if str(s.get("sample_id", "")) in ids]
                print(f"\nRunning LOCOMO benchmark (samples={args.sample_ids}, max_turns={args.max_turns})", flush=True)
            else:
                print(f"\nRunning LOCOMO benchmark (limit={args.limit}, max_turns={args.max_turns})", flush=True)
            locomo_results = await run_locomo(llm_caller, data, len(data), args.qa_limit, args.max_turns, args.concurrency, args.sample_concurrency)
            results.extend(locomo_results)
            print_summary(locomo_results, "LOCOMO")

        if args.benchmark in ("longmemeval", "both"):
            data = json.loads(longmem_path.read_text())
            print(f"\nRunning LongMemEval benchmark (limit={args.limit}, max_turns={args.max_turns})", flush=True)
            longmem_results = await run_longmemeval(llm_caller, data, args.limit, args.max_turns)
            results.extend(longmem_results)
            print_summary(longmem_results, "LongMemEval")

        if args.output and results:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(
                [
                    {
                        "benchmark": r.benchmark,
                        "sample_id": r.sample_id,
                        "question_id": r.question_id,
                        "question_type": r.question_type,
                        "question": r.question,
                        "reference_answer": r.reference_answer,
                        "generated_answer": r.generated_answer,
                        "em": r.em, "f1": r.f1, "judge_score": r.judge_score, "evidence_hit": r.evidence_hit,
                        "memories_retrieved": r.memories_retrieved,
                        "latency_ms": r.latency_ms,
                    }
                    for r in results
                ],
                ensure_ascii=False, indent=2,
            ))
            print(f"\nResults saved to {output_path}", flush=True)

    asyncio.run(run())


if __name__ == "__main__":
    main()

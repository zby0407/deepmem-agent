"""OpenAI-compatible LLM client for DeepMem agent.

A lightweight, standalone LLM client that works with any OpenAI-compatible
endpoint: OpenAI, DashScope (Qwen), Ollama, vLLM, etc.

This replaces the internal services.py dependency for open-source use.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import urllib.request
from typing import Any, AsyncIterator

from .config import cfg

logger = logging.getLogger(__name__)

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


class LLMClient:
    """Async OpenAI-compatible chat completions client.

    Works with OpenAI, DashScope (Qwen), Ollama, vLLM, or any
    OpenAI-compatible endpoint.

    Usage:
        client = LLMClient(api_key="sk-...", base_url="https://api.openai.com/v1")
        result = await client.chat_with_tools([{"role": "user", "content": "Hi"}])
        print(result["content"])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or cfg("llm.base_url", "")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL") or cfg("model.llm", "qwen-flash")
        self.max_tokens = int(max_tokens or os.getenv("VOICE_LLM_MAX_TOKENS") or cfg("llm.max_tokens", 360))

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def chat_with_tools(
        self,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        """Non-streaming chat completion with optional tool support.

        Returns:
            dict with keys: content (str|None), tool_calls (list|None), finish_reason (str)
        """
        url = self._chat_completions_url()
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")

        def _do_request() -> dict:
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            response_data = await asyncio.to_thread(_do_request)

            choice = (response_data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            return {
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls"),
                "finish_reason": choice.get("finish_reason", "stop"),
            }
        except Exception as e:
            logger.error("LLM request failed: %s", e)
            return {"content": None, "tool_calls": None, "finish_reason": "error"}

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Streaming chat completion. Yields content chunks as strings."""
        url = self._chat_completions_url()
        body = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_tokens,
        }

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content") or ""
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error("LLM stream failed: %s", e)


def build_llm_client(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Factory function compatible with the original build_llm_client interface.

    Auto-detects provider from environment variables:
    - LLM_PROVIDER=openai or OPENAI_API_KEY → OpenAI
    - LLM_PROVIDER=dashscope or DASHSCOPE_API_KEY → DashScope (Qwen)
    - LLM_BASE_URL set → custom endpoint (Ollama, vLLM, etc.)
    """
    provider = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()

    if base_url:
        return LLMClient(api_key=api_key, base_url=base_url, model=model)

    if provider in ("openai", "openai_compatible", "compatible") or os.getenv("OPENAI_API_KEY"):
        return LLMClient(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            model=model or os.getenv("LLM_MODEL") or cfg("model.llm", "gpt-4o-mini"),
        )

    # Default: DashScope (Qwen) or any OpenAI-compatible endpoint
    return LLMClient(api_key=api_key, model=model)

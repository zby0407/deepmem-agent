"""Embedding client for DeepMem agent.

Supports DashScope text-embedding-v3 (Alibaba Cloud) and any
OpenAI-compatible embedding endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.request

logger = logging.getLogger(__name__)

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

# Default DashScope embedding URL
_DASHSCOPE_EMBEDDING_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
)


class EmbeddingClient:
    """Synchronous embedding client.

    Supports two modes:
    1. DashScope mode (default): Uses DashScope's native text-embedding API.
       Set DASHSCOPE_API_KEY environment variable.
    2. OpenAI-compatible mode: Uses /v1/embeddings endpoint.
       Set EMBEDDING_BASE_URL and EMBEDDING_API_KEY environment variables.

    The client auto-detects the mode based on environment variables.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        dimension: int = 1024,
    ):
        self.api_key = (
            api_key
            or os.getenv("EMBEDDING_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("LLM_API_KEY")
            or ""
        )
        self.model = model or os.getenv("EMBEDDING_MODEL") or "text-embedding-v3"
        self.dimension = dimension
        self._base_url = base_url or os.getenv("EMBEDDING_BASE_URL") or ""

    def _is_openai_compatible(self) -> bool:
        """Check if we should use OpenAI-compatible /v1/embeddings format."""
        provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
        if provider in ("openai", "openai_compatible"):
            return True
        if self._base_url and "dashscope" not in self._base_url:
            return True
        if os.getenv("OPENAI_API_KEY") and not os.getenv("DASHSCOPE_API_KEY"):
            return True
        return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts synchronously."""
        if not texts:
            return []
        if not self.api_key:
            logger.warning("No API key for embeddings, returning zero vectors")
            return [[0.0] * self.dimension for _ in texts]

        if self._is_openai_compatible():
            return self._embed_openai(texts)
        return self._embed_dashscope(texts)

    def _embed_dashscope(self, texts: list[str]) -> list[list[float]]:
        """DashScope native embedding API."""
        url = self._base_url or _DASHSCOPE_EMBEDDING_URL
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = json.dumps({
            "model": self.model,
            "input": {"texts": texts},
            "parameters": {"dimension": self.dimension},
        }).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
                data = json.loads(resp.read())
            embeddings = data.get("output", {}).get("embeddings", [])
            result: list[list[float]] = [[] for _ in texts]
            for item in embeddings:
                idx = item.get("text_index", 0)
                vec = item.get("embedding", [])
                if 0 <= idx < len(texts):
                    result[idx] = vec
            for i in range(len(result)):
                if not result[i]:
                    result[i] = [0.0] * self.dimension
            return result
        except Exception as e:
            logger.error("DashScope embedding failed: %s", e)
            return [[0.0] * self.dimension for _ in texts]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """OpenAI-compatible /v1/embeddings API."""
        base = (self._base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        url = f"{base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = json.dumps({
            "model": self.model,
            "input": texts,
        }).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
                data = json.loads(resp.read())
            embeddings = data.get("data", [])
            result: list[list[float]] = [[] for _ in texts]
            for item in embeddings:
                idx = item.get("index", 0)
                vec = item.get("embedding", [])
                if 0 <= idx < len(texts):
                    result[idx] = vec
            for i in range(len(result)):
                if not result[i]:
                    result[i] = [0.0] * self.dimension
            return result
        except Exception as e:
            logger.error("OpenAI embedding failed: %s", e)
            return [[0.0] * self.dimension for _ in texts]

    def embed_single(self, text: str) -> list[float]:
        results = self.embed([text])
        return results[0] if results else [0.0] * self.dimension

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    async def aembed_single(self, text: str) -> list[float]:
        return self.embed_single(text)

    async def close(self):
        pass

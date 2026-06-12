"""DeepMem Agent — Hermes-inspired memory system for conversational AI.

A self-contained memory agent package for long-term conversational memory
with cognitive-science-inspired retention, hybrid retrieval, and background
consolidation. Designed for LoCoMo benchmark evaluation.

Key components:
- LocalExternalMemoryProvider: SQLite-backed memory with hybrid search
- MemoryManager: Provider orchestration and lifecycle management
- RealtimeVoiceHarness: Turn-level conversation orchestration
- EmbeddingClient: Pluggable embedding (DashScope / OpenAI)
- LLMClient: Pluggable LLM (DashScope / OpenAI / Ollama)
"""

from .embedding import EmbeddingClient
from .harness import RealtimeVoiceHarness, VoiceTurnResult
from .llm import LLMClient, build_llm_client
from .local_memory import LocalExternalMemoryProvider
from .memory_manager import MemoryManager
from .memory_provider import MemoryProvider

__all__ = [
    "EmbeddingClient",
    "LLMClient",
    "LocalExternalMemoryProvider",
    "MemoryManager",
    "MemoryProvider",
    "RealtimeVoiceHarness",
    "VoiceTurnResult",
    "build_llm_client",
]

__version__ = "0.1.0"

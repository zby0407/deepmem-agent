"""Realtime voice harness adapted from Hermes Agent's conversation lifecycle.

The core ordering mirrors Hermes's AIAgent memory integration:
on_turn_start -> prefetch_all -> fenced memory context in prompt ->
sync_all -> queue_prefetch_all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

from .memory_manager import MemoryManager, build_memory_context_block
from .config import cfg


class VoiceLLM(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        extra_tool_map: dict[str, Any] | None = None,
    ) -> str:
        """Complete one realtime voice turn."""


@dataclass
class VoiceTurnResult:
    user_text: str
    assistant_text: str
    turn_number: int
    memory_context: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)


class RealtimeVoiceHarness:
    """Hermes-style harness specialized for realtime voice turns."""

    def __init__(
        self,
        session_id: str,
        memory_manager: MemoryManager,
        llm: VoiceLLM,
        *,
        platform: str = "voice",
        system_prompt: str | None = None,
        enable_tools: bool = True,
        extra_tool_providers: list[Any] | None = None,
    ):
        self.session_id = session_id
        self.memory_manager = memory_manager
        self.llm = llm
        self.platform = platform
        self.enable_tools = enable_tools
        self._extra_tool_providers = extra_tool_providers or []
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif self.enable_tools:
            extra_hints = ""
            if self._extra_tool_providers:
                for provider in self._extra_tool_providers:
                    name = getattr(provider, '__class__', type(provider)).__name__
                    if name == "LarkToolProvider":
                        extra_hints += (
                            "\n- 飞书消息: 用户提到飞书/消息时，用 lark_read_messages 读取，lark_reply_message 回复。"
                        )
                    elif name == "VocabProvider":
                        extra_hints += (
                            "\n- 背单词: 用户说背单词/学单词/复习单词时，必须先调用 vocab_review 获取单词列表，"
                            "然后用英译中、中译英、填空等方式逐一提问。用户回答后调用 vocab_rate(word, quality) 评分。"
                            "用户要加单词时调用 vocab_add。查看进度调用 vocab_stats。"
                        )
            self.system_prompt = cfg(
                "prompts.voice_system_tools",
                "You are a realtime voice assistant. Reply naturally and concisely. "
                "Today is {today}. Use tools when appropriate.{extra_hints}",
            ).format(today=today, extra_hints=extra_hints)
        else:
            self.system_prompt = cfg(
                "prompts.voice_system",
                "You are a realtime voice assistant. Reply naturally and concisely. "
                "Today is {today}.",
            ).format(today=today)
        self.turn_number = 0
        self.active_partial_text = ""
        self.pending_file_context = None  # Set by VoiceRuntimeLLMProcessor on FileContextFrame
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
            }
        ]

    def _build_system_prompt(self) -> str:
        memory_prompt = self.memory_manager.build_system_prompt()
        if not memory_prompt:
            return self.system_prompt
        return f"{self.system_prompt}\n\n{memory_prompt}"

    async def handle_transcript_partial(self, text: str) -> None:
        self.active_partial_text = text or ""

    async def handle_interruption(self) -> None:
        self.active_partial_text = ""

    async def handle_transcript_final(self, text: str) -> VoiceTurnResult:
        user_text = (text or "").strip()
        if not user_text:
            return VoiceTurnResult(
                user_text="",
                assistant_text="",
                turn_number=self.turn_number,
                messages=list(self.messages),
            )

        self.active_partial_text = ""
        self.turn_number += 1
        self.memory_manager.on_turn_start(
            self.turn_number,
            user_text,
            platform=self.platform,
        )
        raw_memory_context = self.memory_manager.prefetch_all(user_text, session_id=self.session_id)
        fenced_memory_context = build_memory_context_block(raw_memory_context)

        turn_messages = [*self.messages]
        if fenced_memory_context:
            memory_insert_at = 1 if turn_messages else 0
            turn_messages.insert(
                memory_insert_at,
                {"role": "system", "content": fenced_memory_context},
            )
        # Inject file context if pending
        file_description = ""
        if self.pending_file_context:
            fc = self.pending_file_context
            file_description = fc.description
            turn_messages.append({
                "role": "system",
                "content": f"[用户上传了文件: {fc.filename}（{fc.file_type}）]\n{fc.description}",
            })
            self.pending_file_context = None
        turn_messages.append({"role": "user", "content": user_text})
        tools = self.memory_manager.get_all_tool_schemas() if self.enable_tools else []
        logger.info("[Harness] Turn %d: %d tools available", self.turn_number, len(tools))

        # Inject extra tool schemas (e.g. Lark)
        extra_tool_map: dict[str, Any] = {}
        for provider in self._extra_tool_providers:
            for schema in provider.get_tool_schemas():
                name = schema.get("name", "")
                if name:
                    tools.append(schema)
                    extra_tool_map[name] = provider

        assistant_text = await self.llm.complete(
            turn_messages, tools=tools, extra_tool_map=extra_tool_map or None,
        )
        assistant_text = (assistant_text or "").strip()

        # Include file context in the recorded user message for memory extraction
        full_user_text = user_text
        if file_description:
            full_user_text = f"{user_text}\n\n[附件内容]\n{file_description}"
        self.messages.append({"role": "user", "content": full_user_text})
        self.messages.append({"role": "assistant", "content": assistant_text})

        self.memory_manager.sync_all(full_user_text, assistant_text, session_id=self.session_id)
        self.memory_manager.queue_prefetch_all(user_text, session_id=self.session_id)

        return VoiceTurnResult(
            user_text=user_text,
            assistant_text=assistant_text,
            turn_number=self.turn_number,
            memory_context=raw_memory_context,
            messages=turn_messages,
        )

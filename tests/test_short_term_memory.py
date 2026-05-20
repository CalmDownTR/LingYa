from __future__ import annotations

import pytest

from lingya.memory.short_term import Message, ShortTermMemory


class TestShortTermMemory:
    @pytest.fixture
    def memory(self) -> ShortTermMemory:
        return ShortTermMemory(max_messages=6)

    def test_add_appends_message(self, memory):
        msg = Message(role="user", content="hello")
        memory.add(msg)
        assert len(memory) == 1

    def test_get_messages_returns_all(self, memory):
        memory.add(Message(role="user", content="hello"))
        memory.add(Message(role="assistant", content="hi"))
        result = memory.get_messages()
        assert len(result) == 2
        assert result[0].content == "hello"
        assert result[1].content == "hi"

    def test_get_messages_last_n(self, memory):
        for i in range(5):
            memory.add(Message(role="user", content=f"msg{i}"))
        result = memory.get_messages(last_n=3)
        assert len(result) == 3
        assert result[0].content == "msg2"
        assert result[1].content == "msg3"
        assert result[2].content == "msg4"

    def test_get_conversation_text(self, memory):
        memory.add(Message(role="user", content="hello"))
        memory.add(Message(role="assistant", content="hi there"))
        text = memory.get_conversation_text()
        assert "user: hello" in text
        assert "assistant: hi there" in text
        assert text == "user: hello\nassistant: hi there"

    def test_hard_cap_enforcement_pops_oldest(self, memory):
        for i in range(10):
            memory.add(Message(role="user", content=f"msg{i}"))
        assert len(memory) == 6  # capped at max_messages
        messages = memory.get_messages()
        assert messages[0].content == "msg4"
        assert messages[-1].content == "msg9"

    def test_prepend_inserts_at_front(self, memory):
        memory.add(Message(role="user", content="first"))
        memory.add(Message(role="assistant", content="second"))
        memory.prepend(Message(role="system", content="summary"))
        messages = memory.get_messages()
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[0].content == "summary"
        assert messages[1].content == "first"
        assert messages[2].content == "second"

    def test_pop_compressible_removes_from_front(self, memory):
        for i in range(5):
            memory.add(Message(role="user", content=f"msg{i}"))
        popped = memory.pop_compressible(3)
        assert len(popped) == 3
        assert popped[0].content == "msg0"
        assert popped[2].content == "msg2"
        assert len(memory) == 2
        assert memory.get_messages()[0].content == "msg3"

    def test_pop_compressible_more_than_available(self, memory):
        memory.add(Message(role="user", content="only"))
        popped = memory.pop_compressible(5)
        assert len(popped) == 1
        assert len(memory) == 0

    def test_pop_compressible_zero(self, memory):
        memory.add(Message(role="user", content="msg"))
        popped = memory.pop_compressible(0)
        assert len(popped) == 0
        assert len(memory) == 1

    def test_clear_empties_all(self, memory):
        for i in range(3):
            memory.add(Message(role="user", content=f"msg{i}"))
        memory.clear()
        assert len(memory) == 0
        assert memory.get_messages() == []

    def test_token_count_estimate(self, memory):
        memory.add(Message(role="user", content="hello world"))  # 11 chars
        memory.add(Message(role="assistant", content="hi"))      # 2 chars
        # 13 chars // 2 = 6
        assert memory.token_count_estimate() == 6

    def test_message_default_timestamp(self):
        msg = Message(role="user", content="test")
        assert msg.role == "user"
        assert msg.content == "test"
        assert msg.timestamp is not None

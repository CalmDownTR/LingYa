from __future__ import annotations

import readline  # noqa: F401 — fix CJK backspace in input()
from datetime import datetime, timezone

from langchain.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from lingya.persona.config import PersonaConfig
from lingya.reflection import generate_opening_line
from lingya.storage.db import Database


class LingYaCLI:
    def __init__(
        self,
        agent,
        db: Database,
        model: BaseChatModel,
        persona_config: PersonaConfig,
        memory=None,
    ) -> None:
        self.agent = agent
        self.db = db
        self._model = model
        self._persona_config = persona_config
        self.memory = memory
        self.console = Console()
        self._conv_id: int | None = None
        self._thread_id: str = "default"

    async def run(self) -> None:
        self._print_welcome()
        await self._ensure_conversation()
        await self._show_opening()

        while True:
            try:
                self.console.print()
                user_input = input("You: ")
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Goodbye.[/]")
                return

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                await self._handle_command(user_input)
            else:
                with self.console.status("[dim]Thinking...[/]"):
                    response = await self._invoke_agent(user_input)
                self.console.print()
                self.console.print(Panel.fit(
                    Markdown(response),
                    border_style="green",
                    title="LingYa",
                    title_align="left",
                ))

    async def _invoke_agent(self, user_input: str) -> str:
        result = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            {"configurable": {"thread_id": self._thread_id}},
        )

        # Extract the last AI response, skipping tool results
        messages = result.get("messages", [])
        ais = [m for m in messages if isinstance(m, AIMessage)]
        response_text = ais[-1].text if ais else ""

        # Persist this turn
        if self._conv_id:
            await self.db.add_turn(self._conv_id, "user", user_input)
            if response_text:
                await self.db.add_turn(self._conv_id, "ai", response_text)
            await self.db.update_conversation_timestamp(self._conv_id)

        return response_text

    async def _ensure_conversation(self) -> None:
        if self._conv_id is not None:
            return
        title = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self._conv_id = await self.db.create_conversation(title)
        self._thread_id = str(self._conv_id)

    async def _handle_command(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        match command:
            case "/exit" | "/quit":
                self.console.print("[dim]Goodbye.[/]")
                raise EOFError()
            case "/help":
                self._show_help()
            case "/sessions":
                await self._cmd_sessions()
            case "/new":
                await self._cmd_new_session()
            case "/switch":
                await self._cmd_switch(arg)
            case "/memories":
                self._cmd_memories()
            case "/forget":
                self._cmd_forget(arg)
            case "/remember":
                self._cmd_remember(arg)
            case _:
                self.console.print(f"[yellow]Unknown command: {command}[/]. Type /help for available commands.")

    async def _cmd_sessions(self) -> None:
        sessions = await self.db.list_conversations()
        if not sessions:
            self.console.print("[dim]No sessions found.[/]")
            return
        current = self._conv_id
        self.console.print()
        for s in sessions:
            marker = "[bold cyan]→[/]" if s["id"] == current else "  "
            self.console.print(
                f"{marker} [bold]#{s['id']}[/] {s['title']} "
                f"[dim]({s['updated_at']})[/]"
            )

    async def _cmd_new_session(self) -> None:
        title = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self._conv_id = await self.db.create_conversation(title)
        self._thread_id = str(self._conv_id)
        self.console.print(f"[green]Created session #{self._conv_id}: {title}[/]")

    async def _cmd_switch(self, arg: str) -> None:
        if not arg:
            self.console.print("[yellow]Usage: /switch <session_id>[/]")
            return
        try:
            session_id = int(arg)
        except ValueError:
            self.console.print("[yellow]Session ID must be a number.[/]")
            return
        conv = await self.db.get_conversation(session_id)
        if conv is None:
            self.console.print(f"[yellow]Session #{session_id} does not exist.[/]")
            return
        self._conv_id = session_id
        self._thread_id = str(session_id)
        self.console.print(f"[green]Switched to session #{session_id}: {conv['title']}[/]")

    def _cmd_memories(self) -> None:
        if self.memory is None:
            self.console.print("[yellow]Memory system not available.[/]")
            return
        items = self.memory.list_all()
        if not items:
            self.console.print("[dim]No memories stored yet.[/]")
            return
        self.console.print()
        for i, item in enumerate(items, 1):
            self.console.print(f"  [bold][{i}][/] {item['text']}")

    def _cmd_forget(self, arg: str) -> None:
        if self.memory is None:
            self.console.print("[yellow]Memory system not available.[/]")
            return
        if not arg:
            self.console.print("[yellow]Usage: /forget <index>[/]")
            return
        try:
            index = int(arg)
        except ValueError:
            self.console.print("[yellow]Index must be a number (use /memories to see indices).[/]")
            return
        items = self.memory.list_all()
        if index < 1 or index > len(items):
            self.console.print(f"[yellow]Index {index} out of range (1-{len(items)}).[/]")
            return
        mem_id = items[index - 1]["id"]
        self.memory.delete(mem_id)
        self.console.print(f"[green]Forgot: {items[index - 1]['text']}[/]")

    def _cmd_remember(self, arg: str) -> None:
        if self.memory is None:
            self.console.print("[yellow]Memory system not available.[/]")
            return
        if not arg:
            self.console.print("[yellow]Usage: /remember <text>[/]")
            return
        mem_id = self.memory.store(arg)
        self.console.print(f"[green]Remembered: {arg} (id: {mem_id})[/]")

    def _print_welcome(self) -> None:
        self.console.print()

    async def _show_opening(self) -> None:
        """Generate and display LingYa's opening line for this session."""
        # Find the most recent conversation that isn't this one
        sessions = await self.db.list_conversations()
        prev_session = None
        for s in sessions:
            if s["id"] != self._conv_id:
                prev_session = s
                break

        transcript = None
        if prev_session is not None:
            turns = await self.db.get_turns(prev_session["id"], limit=6)
            if turns:
                lines = [f"{'LingYa' if t['role'] == 'ai' else 'User'}: {t['content']}" for t in turns]
                transcript = "\n".join(lines)

        with self.console.status("[dim]Waking up...[/]"):
            line = await generate_opening_line(
                self._model, self._persona_config, transcript
            )
        if line is None:
            return  # Silent fallback

        self.console.print(Panel.fit(
            line,
            border_style="magenta",
            title="LingYa",
            title_align="left",
        ))

    def _show_help(self) -> None:
        help_text = """
**Available Commands:**

| Command | Description |
|---------|-------------|
| `/sessions` | List all sessions |
| `/new` | Start a new session |
| `/switch <id>` | Switch to a session by ID |
| `/memories` | List all stored memories |
| `/forget <index>` | Delete a memory by its index |
| `/remember <text>` | Manually add a memory |
| `/help` | Show this help |
| `/exit`, `/quit` | Exit LingYa |

Anything else is treated as a conversation message.
"""
        self.console.print(Markdown(help_text))

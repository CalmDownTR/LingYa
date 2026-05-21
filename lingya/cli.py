from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from lingya.personality.engine import PersonalityEngine
from lingya.storage.db import Database


class LingYaCLI:
    def __init__(self, agent, personality_engine: PersonalityEngine, db: Database) -> None:
        self.agent = agent
        self.personality_engine = personality_engine
        self.db = db
        self.console = Console()
        self._conv_id: int | None = None
        self._thread_id: str = "default"

    async def run(self) -> None:
        self._print_welcome()
        await self._ensure_conversation()

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/]")
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
        # Log user turn
        if self._conv_id:
            await self.db.log_turn(self._conv_id, "user", user_input)

        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            {"configurable": {"thread_id": self._thread_id}},
        )

        # Extract final AI response
        messages = result.get("messages", [])
        response_text = ""
        if messages:
            last = messages[-1]
            content = last.content if hasattr(last, "content") else str(last)
            response_text = content if isinstance(content, str) else str(content)

        # Log assistant turn
        if self._conv_id and response_text:
            await self.db.log_turn(self._conv_id, "assistant", response_text)

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
            case "/personality":
                self._cmd_personality()
            case "/sessions":
                await self._cmd_sessions()
            case "/new":
                await self._cmd_new_session()
            case "/switch":
                await self._cmd_switch(arg)
            case _:
                self.console.print(f"[yellow]Unknown command: {command}[/]. Type /help for available commands.")

    def _cmd_personality(self) -> None:
        p = self.personality_engine.personality
        prompt = p.to_system_prompt()
        self.console.print(Panel(prompt, title="Current Personality", border_style="blue"))

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
                f"[dim]({s.get('turn_count', 0)} turns, {s['updated_at']})[/]"
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

    def _print_welcome(self) -> None:
        p = self.personality_engine.personality
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold]{p.name}[/] — {p.role}\n"
            f"Tone: {p.tone}\n\n"
            "Type /help for available commands.",
            border_style="blue",
            title="Welcome",
        ))

    def _show_help(self) -> None:
        help_text = """
**Available Commands:**

| Command | Description |
|---------|-------------|
| `/personality` | View current personality |
| `/sessions` | List all sessions |
| `/new` | Start a new session |
| `/switch <id>` | Switch to a session by ID |
| `/help` | Show this help |
| `/exit`, `/quit` | Exit LingYa |

Anything else is treated as a conversation message.
"""
        self.console.print(Markdown(help_text))

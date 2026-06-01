from __future__ import annotations

import json
import readline  # noqa: F401 — fix CJK backspace in input()
from datetime import date, datetime, timezone

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from lingya.reflection import generate_opening_line
from lingya.storage.db import Database


class LingYaCLI:
    def __init__(
        self,
        agent,
        db: Database,
        model: BaseChatModel,
        engine,  # MindEngine
        memory=None,
        data_dir: str = "./data",
        diary_period_days: int = 1,
        ws_client=None,  # Optional GatewayClient for WS mode
    ) -> None:
        self.agent = agent
        self.db = db
        self._model = model
        self._engine = engine
        self.memory = memory
        self._data_dir = data_dir
        self._diary_period_days = diary_period_days
        self._ws_client = ws_client
        self.console = Console()
        self._conv_id: int | None = None
        self._thread_id: str = "default"

    async def run(self) -> None:
        self._print_welcome()
        await self._ensure_conversation()
        await self._show_opening()

        new_diary = await self._maybe_generate_diary()
        if new_diary:
            self.console.print("[dim]📔 我写了一篇新日记。输入 /diary 查看。[/]")

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

    async def run_ws(self) -> None:
        """Run the CLI in WebSocket mode — all logic delegated to Gateway.

        Uses self._ws_client for all communication. Local resources
        (agent, db, model, engine, memory) are not available in this mode.
        """
        self._print_welcome()

        # No opening line in WS mode — the model lives in the daemon.
        # No diary check in WS mode — the daemon handles diary generation.

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
                await self._handle_command_ws(user_input)
            else:
                await self._handle_chat_ws(user_input)

    async def _handle_chat_ws(self, user_input: str) -> None:
        """Send a chat message via WebSocket and display the response."""
        with self.console.status("[dim]Thinking...[/]"):
            response = await self._ws_client.send({
                "type": "chat",
                "payload": {"text": user_input},
            })
        self.console.print()
        self._display_ws_response(response)

    async def _handle_command_ws(self, cmd: str) -> None:
        """Handle slash commands in WebSocket mode."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        match command:
            case "/exit" | "/quit":
                self.console.print("[dim]Goodbye.[/]")
                raise EOFError()
            case "/help":
                self._show_help()
            case "/diary":
                await self._cmd_diary_ws(arg)
            case "/memories":
                await self._cmd_memories_ws()
            case "/mind":
                await self._cmd_mind_ws(arg)
            case "/sessions" | "/new" | "/switch" | "/forget" | "/remember":
                self.console.print(
                    "[yellow]This command is not available in WebSocket mode. "
                    "Use the direct CLI (python main.py) for session management.[/]"
                )
            case _:
                self.console.print(
                    f"[yellow]Unknown command: {command}[/]. "
                    "Type /help for available commands."
                )

    async def _cmd_diary_ws(self, arg: str) -> None:
        """Handle /diary in WebSocket mode."""
        if arg == "list":
            payload = {"action": "list"}
        elif arg.isdigit():
            payload = {"action": "read", "index": int(arg) - 1}
        elif arg:
            self.console.print("[yellow]Usage: /diary | /diary list | /diary <number>[/]")
            return
        else:
            payload = {"action": "read", "index": 0}

        with self.console.status("[dim]Querying diary...[/]"):
            response = await self._ws_client.send({
                "type": "diary",
                "payload": payload,
            })
        self._display_ws_response(response)

    async def _cmd_memories_ws(self) -> None:
        """Handle /memories in WebSocket mode — lists all memories."""
        with self.console.status("[dim]Querying memories...[/]"):
            response = await self._ws_client.send({
                "type": "memory",
                "payload": {"action": "list"},
            })
        self._display_ws_response(response)

    async def _cmd_mind_ws(self, arg: str) -> None:
        """Handle /mind in WebSocket mode — query mind state."""
        query = arg if arg in ("state", "tone") else "state"
        with self.console.status("[dim]Querying mind state...[/]"):
            response = await self._ws_client.send({
                "type": "mind",
                "payload": {"query": query},
            })
        self._display_ws_response(response)

    def _display_ws_response(self, response: dict) -> None:
        """Display a WebSocket response in the CLI."""
        resp_type = response.get("type", "")

        if resp_type == "error":
            self.console.print(
                f"[yellow]Error: {response.get('payload', {}).get('message', 'Unknown error')}[/]"
            )

        elif resp_type == "diary_response":
            payload = response.get("payload", {})
            action = payload.get("action", "")
            if action == "list":
                diaries = payload.get("diaries", [])
                if not diaries:
                    self.console.print("[dim]No diaries yet.[/]")
                    return
                self.console.print()
                self.console.print("[bold]LingYa's Diary[/]")
                self.console.print("─" * 40)
                for i, d in enumerate(diaries, 1):
                    date_str = d.get("date", "unknown")
                    preview = d.get("preview", "")
                    self.console.print(f"  [bold][{i}][/] {date_str}  [dim]{preview}[/]")
                self.console.print("─" * 40)
                self.console.print(f"{len(diaries)} diaries. Use /diary <number> to read.")
            elif action == "read":
                date_str = payload.get("date", "unknown")
                content = payload.get("content", "")
                self.console.print()
                self.console.print("[bold]LingYa's Diary[/]")
                self.console.print("━" * 40)
                self.console.print(f"[bold]{date_str}[/]")
                self.console.print()
                self.console.print(Markdown(content))
                self.console.print("━" * 40)

        elif resp_type == "memory_response":
            payload = response.get("payload", {})
            action = payload.get("action", "")
            if action == "list":
                memories = payload.get("memories", [])
                if not memories:
                    self.console.print("[dim]No memories stored yet.[/]")
                    return
                self.console.print()
                for i, item in enumerate(memories, 1):
                    self.console.print(f"  [bold][{i}][/] {item['text']}")
            elif action == "search":
                results = payload.get("results", [])
                if not results:
                    self.console.print("[dim]No matching memories.[/]")
                    return
                self.console.print()
                for i, r in enumerate(results, 1):
                    self.console.print(f"  [bold][{i}][/] {r['text']}")

        elif resp_type == "mind_state":
            payload = response.get("payload", {})
            if "tone" in payload and "pad" not in payload:
                # Tone-only response
                tone = payload.get("tone", {})
                self.console.print()
                self.console.print("[bold]Current Tone[/]")
                self.console.print(f"  Warmth:    {tone.get('warmth', '?')}")
                self.console.print(f"  Formality: {tone.get('formality', '?')}")
                self.console.print(f"  Humor:     {tone.get('humor', '?')}")
            else:
                # Full state
                pad = payload.get("pad", {})
                ocean = payload.get("ocean", {})
                tone = payload.get("tone", {})
                self.console.print()
                self.console.print("[bold]Mind State[/]")
                self.console.print(f"  Emotion:  {payload.get('emotion', '?')} "
                                   f"(intensity: {payload.get('emotion_intensity', '?')})")
                self.console.print(f"  PAD:      P={pad.get('pleasure', '?'):.2f} "
                                   f"A={pad.get('arousal', '?'):.2f} "
                                   f"D={pad.get('dominance', '?'):.2f}")
                self.console.print(f"  IPC:      {payload.get('ipc_state', '?')} "
                                   f"(agency={payload.get('ipc_agency', '?'):.2f}, "
                                   f"communion={payload.get('ipc_communion', '?'):.2f})")
                self.console.print(f"  OCEAN:    O={ocean.get('openness', '?'):.2f} "
                                   f"C={ocean.get('conscientiousness', '?'):.2f} "
                                   f"E={ocean.get('extraversion', '?'):.2f} "
                                   f"A={ocean.get('agreeableness', '?'):.2f} "
                                   f"N={ocean.get('neuroticism', '?'):.2f}")
                self.console.print(f"  Tone:     warmth={tone.get('warmth', '?')} "
                                   f"formality={tone.get('formality', '?')} "
                                   f"humor={tone.get('humor', '?')}")
                self.console.print(f"  Turn:     {payload.get('turn_counter', '?')}")
        else:
            # Unknown or chat response — display as markdown
            resp_text = response.get("payload", {}).get("text", json.dumps(response, ensure_ascii=False, default=str))
            self.console.print(Panel.fit(
                Markdown(resp_text),
                border_style="green",
                title="LingYa",
                title_align="left",
            ))

    async def _invoke_agent(self, user_input: str) -> str:
        # Prepend dynamic tone/mood fragment as SystemMessage
        fragment = self._engine.get_prompt_fragment()
        messages: list = [HumanMessage(content=user_input)]
        if fragment:
            messages.insert(0, SystemMessage(content=fragment))

        result = await self.agent.ainvoke(
            {"messages": messages},
            {"configurable": {"thread_id": self._thread_id}},
        )

        # Extract the last AI response, skipping tool results
        msgs = result.get("messages", [])
        ais = [m for m in msgs if isinstance(m, AIMessage)]
        response_text = ais[-1].text if ais else ""

        # MindEngine: process user event + check response alignment
        await self._engine.process_event({
            "event_type": "outcome",
            "valence": "neutral",
            "focus": "self",
            "description": user_input,
            "content": user_input,
        })
        if response_text:
            await self._engine.check_response_alignment(response_text)

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
            case "/diary":
                self._cmd_diary(arg)
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

    def _cmd_diary(self, arg: str) -> None:
        from lingya.diary import get_diary_dir, list_diaries, read_diary

        diary_dir = get_diary_dir(self._data_dir)

        if arg == "list":
            diaries = list_diaries(diary_dir)
            if not diaries:
                self.console.print("[dim]还没有日记。[/]")
                return
            self.console.print()
            self.console.print("[bold]📔 LingYa 的日记[/]")
            self.console.print("─" * 40)
            for i, d in enumerate(diaries, 1):
                date_str = d["date"].strftime("%Y年%m月%d日")
                preview = d.get("preview", "")
                self.console.print(f"  [bold][{i}][/] {date_str}  [dim]{preview}[/]")
            self.console.print("─" * 40)
            self.console.print(f"共 {len(diaries)} 篇日记。输入 /diary <编号> 查看。")
        elif arg.isdigit():
            index = int(arg) - 1
            result = read_diary(diary_dir, index)
            if result is None:
                self.console.print(f"[yellow]没有第 {arg} 篇日记。[/]")
                return
            diary_date, content = result
            self._show_diary(diary_date, content)
        elif arg:
            self.console.print("[yellow]用法: /diary | /diary list | /diary <编号>[/]")
        else:
            result = read_diary(diary_dir, 0)
            if result is None:
                self.console.print("[dim]还没有日记。LingYa 会在适当的时候写日记。[/]")
                return
            diary_date, content = result
            self._show_diary(diary_date, content)

    def _show_diary(self, diary_date, content: str) -> None:
        self.console.print()
        self.console.print("[bold]📔 LingYa 的日记[/]")
        self.console.print("━" * 40)
        self.console.print(f"[bold]{diary_date.strftime('%Y年%m月%d日')}[/]")
        self.console.print()
        self.console.print(Markdown(content))
        self.console.print("━" * 40)

    async def _maybe_generate_diary(self) -> bool:
        """Check if a diary should be generated, generate it if so.

        Returns True if a new diary was written.
        """
        from lingya.diary import (
            format_transcript,
            generate_diary,
            get_diary_dir,
            get_last_diary_date,
            has_deep_conversation,
            save_diary,
            should_generate_diary,
        )

        diary_dir = get_diary_dir(self._data_dir)

        if not should_generate_diary(diary_dir, self._diary_period_days):
            return False

        last_date = get_last_diary_date(diary_dir)
        since = last_date.isoformat() if last_date else "1970-01-01"

        turns = await self.db.get_turns_since(since, limit=200)

        if not has_deep_conversation(turns):
            return False

        transcript = format_transcript(turns)

        with self.console.status("[dim]Writing in diary...[/]"):
            try:
                content = await generate_diary(
                    self._model, self._engine.config, transcript
                )
            except Exception:
                return False

        if not content:
            return False

        save_diary(diary_dir, date.today(), content)
        return True

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
                self._model, self._engine.config, transcript
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

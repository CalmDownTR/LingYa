from __future__ import annotations

import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


class LingYaCLI:
    """Thin CLI client — all logic lives in the Gateway daemon."""

    def __init__(self, ws_client) -> None:
        self._ws_client = ws_client
        self.console = Console()

    async def run(self) -> None:
        self._print_welcome()

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
                await self._handle_chat(user_input)

    async def _handle_chat(self, user_input: str) -> None:
        """Send a chat message via WebSocket and display the response."""
        with self.console.status("[dim]Thinking...[/]"):
            response = await self._ws_client.send({
                "type": "chat",
                "payload": {"text": user_input},
            })
        self.console.print()
        self._display_response(response)

    async def _handle_command(self, cmd: str) -> None:
        """Handle slash commands."""
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
                await self._cmd_diary(arg)
            case "/memories":
                await self._cmd_memories()
            case "/mind":
                await self._cmd_mind(arg)
            case "/stats":
                await self._cmd_stats()
            case "/sessions" | "/new" | "/switch" | "/forget" | "/remember" | "/recover":
                self.console.print(
                    "[yellow]Session management is handled by the Gateway daemon. "
                    "Not yet available via WebSocket.[/]"
                )
            case _:
                self.console.print(
                    f"[yellow]Unknown command: {command}[/]. "
                    "Type /help for available commands."
                )

    async def _cmd_diary(self, arg: str) -> None:
        """Handle /diary."""
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
        self._display_response(response)

    async def _cmd_memories(self) -> None:
        """Handle /memories — lists all memories."""
        with self.console.status("[dim]Querying memories...[/]"):
            response = await self._ws_client.send({
                "type": "memory",
                "payload": {"action": "list"},
            })
        self._display_response(response)

    async def _cmd_mind(self, arg: str) -> None:
        """Handle /mind — query mind state."""
        query = arg if arg in ("state", "tone") else "state"
        with self.console.status("[dim]Querying mind state...[/]"):
            response = await self._ws_client.send({
                "type": "mind",
                "payload": {"query": query},
            })
        self._display_response(response)

    async def _cmd_stats(self) -> None:
        """Handle /stats."""
        with self.console.status("[dim]Querying stats...[/]"):
            response = await self._ws_client.send({
                "type": "stats",
                "payload": {},
            })
        self._display_response(response)

    def _display_response(self, response: dict) -> None:
        """Display a WebSocket response."""
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
                tone = payload.get("tone", {})
                self.console.print()
                self.console.print("[bold]Current Tone[/]")
                self.console.print(f"  Warmth:    {tone.get('warmth', '?')}")
                self.console.print(f"  Formality: {tone.get('formality', '?')}")
                self.console.print(f"  Humor:     {tone.get('humor', '?')}")
            else:
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

        elif resp_type == "stats_response":
            payload = response.get("payload", {})
            engine_stats = payload.get("engine", {})
            route_stats = payload.get("route_dispatch", {})
            self.console.print()
            self.console.print("[bold]Pipeline Stats (last 200 turns)[/]")
            self.console.print("─" * 60)
            self.console.print("[bold]Engine[/]")
            if engine_stats:
                for metric, data in engine_stats.items():
                    self.console.print(
                        f"  {metric}: "
                        f"p50={data['p50']:.1f}ms  "
                        f"p95={data['p95']:.1f}ms  "
                        f"avg={data['avg']:.1f}ms  "
                        f"min={data['min']:.1f}ms  "
                        f"max={data['max']:.1f}ms  "
                        f"[dim](n={data['count']})[/]"
                    )
            else:
                self.console.print("  [dim]No engine stats yet.[/]")
            if route_stats:
                self.console.print("[bold]Route Dispatch[/]")
                self.console.print(
                    f"  p50={route_stats['p50']:.1f}ms  "
                    f"p95={route_stats['p95']:.1f}ms  "
                    f"avg={route_stats['avg']:.1f}ms  "
                    f"min={route_stats['min']:.1f}ms  "
                    f"max={route_stats['max']:.1f}ms  "
                    f"[dim](n={route_stats['count']})[/]"
                )
            self.console.print("─" * 60)

        else:
            resp_text = response.get("payload", {}).get("text", json.dumps(response, ensure_ascii=False, default=str))
            self.console.print(Panel.fit(
                Markdown(resp_text),
                border_style="green",
                title="LingYa",
                title_align="left",
            ))
            meta = response.get("payload", {}).get("meta", {})
            if meta:
                parts = [f"{k}={v}ms" for k, v in meta.items()]
                self.console.print(f"[dim]{'  '.join(parts)}[/]")

    def _print_welcome(self) -> None:
        self.console.print()

    def _show_help(self) -> None:
        help_text = """
**Available Commands:**

| Command | Description |
|---------|-------------|
| `/diary` | Read or list LingYa's diary |
| `/memories` | List all stored memories |
| `/mind` | Show current mind state (PAD, OCEAN, tone) |
| `/stats` | Show pipeline performance stats |
| `/help` | Show this help |
| `/exit`, `/quit` | Exit LingYa |

Anything else is treated as a conversation message.
"""
        self.console.print(Markdown(help_text))

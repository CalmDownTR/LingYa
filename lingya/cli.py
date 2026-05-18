from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from lingya.agent import LingYaAgent


class LingYaCLI:
    def __init__(self, agent: LingYaAgent) -> None:
        self.agent = agent
        self.console = Console()

    async def run(self) -> None:
        self._print_welcome()
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/]")
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Goodbye.[/]")
                return

            user_input = user_input.strip()
            if not user_input:
                continue

            # Check for commands
            if user_input.startswith("/"):
                await self._handle_command(user_input)
            else:
                # Show thinking indicator
                with self.console.status("[dim]Thinking...[/]"):
                    response = await self.agent.handle_input(user_input)
                self.console.print()
                self.console.print(Panel.fit(
                    Markdown(response),
                    border_style="green",
                    title="LingYa",
                    title_align="left",
                ))

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
                await self._cmd_personality(arg)

            case "/fetch":
                await self._cmd_fetch(arg)

            case "/reflect":
                await self._cmd_reflect()

            case "/history":
                await self._cmd_history()

            case "/clear":
                await self._cmd_clear()

            case _:
                self.console.print(f"[yellow]Unknown command: {command}[/]. Type /help for available commands.")

    async def _cmd_personality(self, arg: str) -> None:
        p = self.agent.personality.personality
        prompt = p.to_system_prompt()
        self.console.print(Panel(prompt, title="Current Personality", border_style="blue"))

    async def _cmd_fetch(self, url: str) -> None:
        if not url:
            self.console.print("[yellow]Usage: /fetch <url>[/]")
            return

        self.console.print(f"[dim]Fetching {url}...[/]")
        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "LingYa/0.1"})
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                # Remove excessive blank lines
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                text = "\n".join(lines)

            result = await self.agent.ingest_and_learn(text, url, "web_page")
            self.console.print(f"[green]{result}[/]")
        except Exception as e:
            self.console.print(f"[red]Failed to fetch: {e}[/]")

    async def _cmd_reflect(self) -> None:
        with self.console.status("[dim]Reflecting...[/]"):
            result = await self.agent.reflect()
        self.console.print()
        self.console.print(Panel(Markdown(result), title="Reflection", border_style="magenta"))

    async def _cmd_history(self) -> None:
        messages = self.agent.memory.short_term.get_messages()
        if not messages:
            self.console.print("[dim]No conversation history.[/]")
            return
        for m in messages:
            role_color = "cyan" if m.role == "user" else "green"
            self.console.print(f"[{role_color}]{m.role}[/]: {m.content[:200]}{'...' if len(m.content) > 200 else ''}")

    async def _cmd_clear(self) -> None:
        self.agent.memory.short_term.clear()
        self.console.print("[dim]Short-term memory cleared.[/]")

    def _print_welcome(self) -> None:
        p = self.agent.personality.personality
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
| `/fetch <url>` | Ingest content from a web page |
| `/personality` | View current personality |
| `/reflect` | Analyze the current conversation |
| `/history` | Show conversation history |
| `/clear` | Clear short-term memory |
| `/help` | Show this help |
| `/exit`, `/quit` | Exit LingYa |

Anything else is treated as a conversation message.
"""
        self.console.print(Markdown(help_text))

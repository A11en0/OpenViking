"""
Query commands for Memex - ask, chat with RAG.
"""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from client import MemexClient
from rag.recipe import MemexRecipe


class QueryCommands:
    """Query commands for RAG-based Q&A."""

    def __init__(self, client: MemexClient, console: Console):
        """Initialize query commands.

        Args:
            client: Memex client instance.
            console: Rich console for output.
        """
        self.client = client
        self.console = console
        self._recipe: Optional[MemexRecipe] = None
        self._chat_mode = False

    @property
    def recipe(self) -> MemexRecipe:
        """Get or create RAG recipe."""
        if self._recipe is None:
            self._recipe = MemexRecipe(self.client)
        return self._recipe

    def ask(self, query: str, target_uri: Optional[str] = None) -> None:
        """Ask a question using RAG (single turn).

        Args:
            query: User's question.
            target_uri: URI to search in.
        """
        if not query:
            self.console.print("[red]Usage: /ask <question>[/red]")
            return

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Searching knowledge base...", total=None)

                response = self.recipe.query(
                    user_query=query,
                    target_uri=target_uri,
                    use_chat_history=False,
                )

                progress.update(task, description="Done!")

            # Display response
            self.console.print()
            self.console.print(
                Panel(
                    Markdown(response),
                    title="[bold cyan]Memex[/bold cyan]",
                    border_style="cyan",
                )
            )

        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")

    def chat(self, query: str, target_uri: Optional[str] = None) -> None:
        """Chat with RAG (multi-turn conversation).

        Args:
            query: User's message.
            target_uri: URI to search in.
        """
        if not query:
            # Toggle chat mode
            self._chat_mode = not self._chat_mode
            if self._chat_mode:
                self.console.print(
                    "[green]Chat mode enabled. Type your questions directly.[/green]"
                )
                self.console.print(
                    "[dim]Use /clear to clear history, /exit to exit chat mode.[/dim]"
                )
            else:
                self.console.print("[yellow]Chat mode disabled.[/yellow]")
            return

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Thinking...", total=None)

                response = self.recipe.query(
                    user_query=query,
                    target_uri=target_uri,
                    use_chat_history=True,
                )

                progress.update(task, description="Done!")

            # Display response
            self.console.print()
            self.console.print(
                Panel(
                    Markdown(response),
                    title="[bold cyan]Memex[/bold cyan]",
                    border_style="cyan",
                )
            )

        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")

    def clear_history(self) -> None:
        """Clear chat history."""
        self.recipe.clear_history()
        self.console.print("[green]Chat history cleared.[/green]")

    @property
    def chat_mode(self) -> bool:
        """Check if chat mode is enabled."""
        return self._chat_mode

    @chat_mode.setter
    def chat_mode(self, value: bool) -> None:
        """Set chat mode."""
        self._chat_mode = value

    def process_input(self, user_input: str, target_uri: Optional[str] = None) -> None:
        """Process user input in chat mode or as direct question.

        Args:
            user_input: User's input.
            target_uri: URI to search in.
        """
        if self._chat_mode:
            self.chat(user_input, target_uri)
        else:
            self.ask(user_input, target_uri)

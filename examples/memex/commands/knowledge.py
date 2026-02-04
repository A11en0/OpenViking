"""
Knowledge management commands for Memex - add, rm, import.
"""

import os
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from client import MemexClient


class KnowledgeCommands:
    """Knowledge management commands for adding and removing resources."""

    def __init__(self, client: MemexClient, console: Console):
        """Initialize knowledge commands.

        Args:
            client: Memex client instance.
            console: Rich console for output.
        """
        self.client = client
        self.console = console

    def add(
        self,
        path: str,
        target: Optional[str] = None,
        reason: Optional[str] = None,
        instruction: Optional[str] = None,
        wait: bool = True,
    ) -> None:
        """Add a resource to the knowledge base.

        Args:
            path: File path, directory path, or URL.
            target: Target URI in viking://.
            reason: Reason for adding this resource.
            instruction: Special instructions for processing.
            wait: Whether to wait for processing to complete.
        """
        if not path:
            self.console.print("[red]Usage: /add <path> [target] [reason][/red]")
            return

        # Expand user path
        if path.startswith("~"):
            path = os.path.expanduser(path)

        # Check if local path exists
        if not path.startswith(("http://", "https://")) and not os.path.exists(path):
            self.console.print(f"[red]Path not found: {path}[/red]")
            return

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"Adding {path}...", total=None)

                result = self.client.add_resource(
                    path=path,
                    target=target,
                    reason=reason,
                    instruction=instruction,
                )

                root_uri = result.get("root_uri", "unknown")
                progress.update(task, description=f"Added to {root_uri}")

                if wait:
                    progress.update(task, description="Processing...")
                    self.client.wait_processed(timeout=120)
                    progress.update(task, description="Done!")

            self.console.print(
                Panel(
                    f"[green]✓[/green] Added: {path}\n[cyan]URI:[/cyan] {root_uri}",
                    title="Resource Added",
                    border_style="green",
                )
            )

        except Exception as e:
            self.console.print(f"[red]Error adding resource: {e}[/red]")

    def rm(self, uri: str, recursive: bool = False) -> None:
        """Remove a resource from the knowledge base.

        Args:
            uri: URI of the resource to remove.
            recursive: Whether to remove recursively.
        """
        if not uri:
            self.console.print("[red]Usage: /rm <uri> [-r][/red]")
            return

        try:
            self.client.remove(uri=uri, recursive=recursive)
            self.console.print(f"[green]✓[/green] Removed: {uri}")

        except Exception as e:
            self.console.print(f"[red]Error removing {uri}: {e}[/red]")

    def import_dir(
        self,
        directory: str,
        target: Optional[str] = None,
        reason: Optional[str] = None,
        wait: bool = True,
    ) -> None:
        """Import all files from a directory.

        Args:
            directory: Directory path to import.
            target: Target URI in viking://.
            reason: Reason for importing.
            wait: Whether to wait for processing to complete.
        """
        if not directory:
            self.console.print("[red]Usage: /import <directory> [target][/red]")
            return

        # Expand user path
        if directory.startswith("~"):
            directory = os.path.expanduser(directory)

        if not os.path.isdir(directory):
            self.console.print(f"[red]Not a directory: {directory}[/red]")
            return

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"Importing {directory}...", total=None)

                # Add the entire directory as a resource
                result = self.client.add_resource(
                    path=directory,
                    target=target,
                    reason=reason or f"Imported from {directory}",
                )

                root_uri = result.get("root_uri", "unknown")
                progress.update(task, description=f"Imported to {root_uri}")

                if wait:
                    progress.update(task, description="Processing...")
                    self.client.wait_processed(timeout=300)
                    progress.update(task, description="Done!")

            self.console.print(
                Panel(
                    f"[green]✓[/green] Imported: {directory}\n[cyan]URI:[/cyan] {root_uri}",
                    title="Directory Imported",
                    border_style="green",
                )
            )

        except Exception as e:
            self.console.print(f"[red]Error importing directory: {e}[/red]")

    def add_url(
        self,
        url: str,
        target: Optional[str] = None,
        reason: Optional[str] = None,
        wait: bool = True,
    ) -> None:
        """Add a URL resource to the knowledge base.

        Args:
            url: URL to add.
            target: Target URI in viking://.
            reason: Reason for adding.
            wait: Whether to wait for processing.
        """
        if not url:
            self.console.print("[red]Usage: /url <url> [target][/red]")
            return

        if not url.startswith(("http://", "https://")):
            self.console.print("[red]Invalid URL. Must start with http:// or https://[/red]")
            return

        self.add(path=url, target=target, reason=reason, wait=wait)

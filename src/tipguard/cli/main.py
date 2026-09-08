"""Command-line entry point for TIP-Guard."""

import typer

from tipguard import __version__

app = typer.Typer(help="TIP-Guard command-line interface.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """TIP-Guard command-line interface."""


@app.command()
def version() -> None:
    """Print the installed tipguard version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()

"""Entry point: ``uv run python -m src <command> [options]``."""

import fire

from src.cli import RagCLI


def main() -> None:
    """Expose RagCLI's methods as CLI subcommands."""
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()

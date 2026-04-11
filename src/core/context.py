"""Context engine — loads agent identity files into a system prompt."""

from pathlib import Path

# Default identity files in priority order
IDENTITY_FILES = ("AGENTS.md", "SOUL.md", "TOOLS.md")


class ContextEngine:
    """Loads markdown identity files and assembles a system prompt."""

    def __init__(self, project_root: str | Path):
        self.root = Path(project_root)

    def _load_file(self, filename: str) -> str | None:
        """Read a single markdown file, returning None if missing."""
        path = self.root / filename
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None

    def build_system_prompt(self, extra_files: list[str] | None = None) -> str:
        """Assemble identity files into a delimited system prompt.

        Args:
            extra_files: Additional markdown filenames to append.

        Returns:
            Combined system prompt string.
        """
        files = list(IDENTITY_FILES) + (extra_files or [])
        sections: list[str] = []

        for filename in files:
            content = self._load_file(filename)
            if content:
                sections.append(
                    f"{'=' * 40}\n"
                    f"SOURCE: {filename}\n"
                    f"{'=' * 40}\n\n"
                    f"{content.strip()}"
                )

        return "\n\n".join(sections)

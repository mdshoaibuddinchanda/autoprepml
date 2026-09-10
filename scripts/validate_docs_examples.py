"""Validate the syntax of Python examples embedded in project Markdown."""

from __future__ import annotations

import ast
import re
from pathlib import Path


FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def validate(root: Path) -> int:
    """Compile every fenced Python block and return the number checked."""
    files = [root / "README.md", root / "creator_examples" / "README.md"]
    files.extend(sorted((root / "docs").rglob("*.md")))
    checked = 0
    failures = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for index, source in enumerate(FENCE_RE.findall(text), start=1):
            checked += 1
            try:
                ast.parse(source, filename=f"{path}:{index}")
            except SyntaxError as exc:
                failures.append(f"{path.relative_to(root)} block {index}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {checked} Python documentation snippet(s)")
    return checked


if __name__ == "__main__":
    validate(Path(__file__).resolve().parents[1])

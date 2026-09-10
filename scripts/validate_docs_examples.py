"""Validate and execute selected Python examples embedded in Markdown."""

from __future__ import annotations

import ast
import contextlib
import io
import os
import re
from pathlib import Path
import tempfile


FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
EXECUTABLE_FENCE_RE = re.compile(
    r"<!--\s*executable(?:-example)?\s*-->\s*" r"```(?:python|py)\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def validate(root: Path) -> int:
    """Compile every fenced Python block and return the number checked."""
    files = [root / "README.md", root / "creator_examples" / "README.md"]
    files.extend(sorted((root / "docs").rglob("*.md")))
    checked = 0
    executed = 0
    failures = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for index, source in enumerate(FENCE_RE.findall(text), start=1):
            checked += 1
            try:
                ast.parse(source, filename=f"{path}:{index}")
            except SyntaxError as exc:
                failures.append(f"{path.relative_to(root)} block {index}: {exc}")
        for index, source in enumerate(EXECUTABLE_FENCE_RE.findall(text), start=1):
            executed += 1
            try:
                with tempfile.TemporaryDirectory(prefix="autoprepml-doc-") as workdir:
                    previous = Path.cwd()
                    try:
                        os.chdir(workdir)
                        with (
                            contextlib.redirect_stdout(io.StringIO()),
                            contextlib.redirect_stderr(io.StringIO()),
                        ):
                            exec(
                                compile(source, f"{path}:{index}", "exec"), {"__name__": "__docs__"}
                            )
                    finally:
                        os.chdir(previous)
            except Exception as exc:  # pragma: no cover - exercised by CI failures
                failures.append(f"{path.relative_to(root)} executable block {index}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {checked} Python documentation snippet(s)")
    print(f"Executed {executed} canonical documentation example(s)")
    return checked


if __name__ == "__main__":
    validate(Path(__file__).resolve().parents[1])

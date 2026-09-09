"""Execute the maintained AutoPrepML notebooks from a clean kernel."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory


def _notebook_paths(root: Path, include_creator: bool) -> list[Path]:
    paths = sorted((root / "examples" / "notebooks").glob("*.ipynb"))
    if include_creator:
        paths.extend(sorted((root / "creator_examples").glob("*.ipynb")))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="write executed outputs back into each notebook",
    )
    parser.add_argument(
        "--include-creator",
        action="store_true",
        help="also execute network-backed creator examples",
    )
    args = parser.parse_args()

    try:
        import nbformat
        from nbformat import validator
        from nbclient import NotebookClient
    except ImportError as error:  # pragma: no cover - exercised in minimal installs
        raise SystemExit(
            "Notebook validation requires the optional dependencies. "
            'Install with: python -m pip install -e ".[notebooks]"'
        ) from error

    root = Path(__file__).resolve().parents[1]
    paths = _notebook_paths(root, args.include_creator)
    if not paths:
        raise SystemExit("No notebooks were found")

    with TemporaryDirectory(prefix="autoprepml-notebooks-") as working_dir:
        for path in paths:
            print(f"Executing {path.relative_to(root)}")
            notebook = nbformat.read(path, as_version=4)
            _, notebook = validator.normalize(notebook, version=4)
            notebook_path = root if args.include_creator else working_dir
            client = NotebookClient(
                notebook,
                timeout=600,
                kernel_name="python3",
                resources={"metadata": {"path": str(notebook_path)}},
            )
            client.execute()
            if args.inplace:
                nbformat.write(notebook, path)
            print(f"Passed {path.name}")

    print(f"Validated {len(paths)} notebook(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

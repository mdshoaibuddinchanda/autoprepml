"""Backward-compatible setuptools entry point.

Project metadata and dependencies live in ``pyproject.toml``. Keeping this
small shim allows older tooling that still invokes ``python setup.py`` to use
the same single source of truth without duplicating package configuration.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()

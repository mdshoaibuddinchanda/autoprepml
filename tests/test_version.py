"""Package version consistency checks."""

from importlib.metadata import version as distribution_version

import autoprepml


def test_package_version_matches_distribution_metadata():
    """The public version must come from installed package metadata."""
    assert autoprepml.__version__ == distribution_version("autoprepml")

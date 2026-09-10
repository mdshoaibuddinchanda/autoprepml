# Release checklist

Use this checklist for every public release. A release is not complete until
the source, wheel, documentation, and repository state agree.

## Before tagging

1. Confirm the version in `pyproject.toml`, `CHANGELOG.md`, and release notes.
2. Run the complete supported Python matrix and confirm the required workflow
   is green.
3. Run tests with line and branch coverage, security audits, strict docs, and
   notebook validation.
4. Build a source distribution and wheel from a clean checkout.
5. Run `twine check` and install the wheel into a fresh environment.
6. Confirm the wheel contains no tests, notebooks, private data, credentials,
   coverage files, or generated reports.
7. Run the documented synthetic and open-data smoke workflows in temporary
   directories and remove their outputs.
8. Review the public README, API reference, migration notes, and repository
   metadata for unsupported claims.

## Publish and verify

1. Create and push the annotated version tag.
2. Confirm the trusted-publishing workflow uses the `pypi` environment and
   publishes through OIDC without a long-lived token.
3. Confirm the GitHub Release contains the exact source and wheel artifacts.
4. Verify the PyPI files and installed version in a clean environment.
5. Record workflow URLs, package URLs, benchmark results, and known
   limitations in the local production-readiness plan.

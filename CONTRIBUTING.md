# Contributing to this project

## Quick start
1. Install dev deps:
   ```bash
   pip install -e .[tester,linter,mypy,formatter]
   pre-commit install
   ```
2. Run checks:
   ```bash
   pre-commit run --all-files && pytest -q
   ```

The repository pins tool versions in both `pre-commit` and optional dependencies so local runs and CI use the same formatter, linter, and type checker versions.

## Commit style
- Keep pull requests focused and add tests for bug fixes.
- Update `CHANGELOG.md` for user-visible changes.

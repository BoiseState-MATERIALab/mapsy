# Contributing to this project

## Quick start
1. Install dev deps:
   ```bash
   pip install -e .[tester,linter,mypy,formatter]
   pre-commit install
   ```
2. Run checks:
   ```bash
   ruff . && black . && mypy . && pytest -q
   ```

## Commit style
- Keep pull requests focused and add tests for bug fixes.
- Update `CHANGELOG.md` for user-visible changes.

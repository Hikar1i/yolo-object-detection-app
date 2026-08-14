# Contributing

Thanks for considering a contribution to **YOLOv8 Object Detection Web Application**!

## Getting started

1. Fork the repo and clone your fork.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Create a feature branch: `git checkout -b feature/my-improvement`.

## Development workflow

- Run the app locally to verify your change: `streamlit run src/app.py`.
- Run the test suite before opening a PR:
  ```bash
  pytest -m "not integration"   # fast, offline
  pytest                        # full suite, downloads model weights
  ```
- Keep functions typed and documented — this project uses type hints and
  docstrings throughout (`src/`).
- Follow [PEP 8](https://peps.python.org/pep-0008/); a formatter like
  `black`/`ruff format` is recommended but not enforced by CI.

## Commit & PR guidelines

- Write clear, present-tense commit messages (`Add batch export option`,
  not `Added stuff`).
- One logical change per PR where practical.
- Describe **what** changed and **why** in the PR description; link any
  related issue.
- Update `CHANGELOG.md` under "Unreleased" for user-facing changes.

## Reporting bugs / requesting features

Open a GitHub issue with:
- Steps to reproduce (for bugs) or the use case (for features).
- Expected vs. actual behavior.
- Environment details (OS, Python version, GPU/CPU).

## Code of conduct

Be respectful and constructive. This is a portfolio/educational project —
questions and beginner-friendly PRs are welcome.

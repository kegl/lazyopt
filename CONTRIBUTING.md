# Contributing to devai-hyperopt

We welcome contributions to devai-hyperopt! By participating, you agree to abide by our guidelines.

## Getting Started

1. Fork the repository
2. Clone your fork and install in development mode:
   ```bash
   git clone git@github.com:<your-username>/devai-hyperopt.git
   cd devai-hyperopt
   pip install -e ".[examples]"
   pip install pytest ruff
   ```
3. Create a feature branch:
   ```bash
   git checkout -b my-feature
   ```

## Development Workflow

### Running Tests

```bash
pytest tests/ -v
```

### Linting and Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check devai_hyperopt/ tests/
ruff format devai_hyperopt/ tests/
```

All code must pass `ruff check` and `ruff format --check` before merging.

### Adding a New Feature

1. Write tests in `tests/` covering your change.
2. Ensure all existing tests still pass.
3. Run `ruff check --fix` and `ruff format` before committing.

## Pull Requests

- Keep PRs focused on a single change.
- Include a clear description of what the PR does and why.
- Reference any related issues.
- Ensure CI passes (pytest + ruff on Python 3.10/3.11/3.12).

## Reporting Issues

Open a GitHub issue with:
- A clear description of the bug or feature request.
- Steps to reproduce (for bugs).
- Expected vs. actual behavior.

## License

By contributing, you agree that your contributions will be licensed under the [BSD 3-Clause](LICENSE) license.

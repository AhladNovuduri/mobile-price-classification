# Continuous integration

The GitHub Actions workflow in `.github/workflows/tests.yml` runs the test suite on Python 3.10, 3.11, and 3.12 for pushes to the main/improvement branches and pull requests targeting main.

Run the same checks locally with:

```bash
pip install -e ".[dev]"
pytest
```

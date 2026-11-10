# Contributing

Contributions should preserve reproducibility and the supplied source assets.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Pull request checklist

- [ ] Keep raw input files unchanged.
- [ ] Add or update automated tests for behavioral changes.
- [ ] Keep preprocessing inside the fitted pipeline.
- [ ] Use deterministic random seeds where applicable.
- [ ] Document new assumptions, metrics, and limitations.
- [ ] Run `pytest` before submitting.
- [ ] Do not commit local environments, credentials, or generated model artifacts.

## Data changes

Do not silently replace or relabel the supplied datasets. Record provenance, license, schema changes, and validation results when adding new data.

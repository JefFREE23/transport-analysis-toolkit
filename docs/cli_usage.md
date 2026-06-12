# CLI Usage

Install the package in editable mode:

```bash
python -m pip install -e .
```

Then run:

```bash
transport analyze sample.csv
transport hall sample.csv
transport mr sample.csv
transport tensor sample.csv
transport fit-one-carrier sample.csv
transport fit-two-carrier sample.csv
transport run-msa sample.csv
```

Use `--json` before the subcommand for machine-readable output:

```bash
transport --json hall sample.csv
```

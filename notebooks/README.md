# notebooks/

Interactive **presentation and exploration layer** — *not* the reproducible
core of the project.

## Division of roles (deliberate)

- **Computation & the reproducible pipeline** live in `data/`, `estimators/`,
  `eval/` and `experiments/` (plain `.py` modules — tested, diffable,
  deterministically executable). That is the runnable core and the forensically
  clean path — notebooks are unsuitable for it because cells can be run in an
  arbitrary order.
- **Notebooks only import these modules and visualize** the finished results
  from `eval/out/`. **No** estimation or evaluation logic is duplicated here;
  anything that belongs in the pipeline goes into `eval/report.py` or
  `experiments/`.

## Reproduction

Run the pipeline first (creates the CSVs in `eval/out/`; see
`docs/experiments.md` for the full order):

```bash
python experiments/exp_accuracy.py
python experiments/exp_contamination.py
python experiments/exp_samplesize.py
```

Then open `explore.ipynb` (kernel = project `.venv`). "Restart & Run All"
must run through at any time without manual intermediate steps.

# Repository demo recording guide

## Before recording

From the repository root, install the package and Jupyter if they are not already available:

```powershell
python -m pip install -e ".[dev]" jupyter
```

Start Jupyter from the repository root, open `notebooks/repo_demo.ipynb`, then use **Restart Kernel and Run All Cells**. For a non-interactive repeatability check, run:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/repo_demo.ipynb
```

The canonical demo has no runtime credentials, private datasets, external APIs, or network calls.

## Opening hook

“In under three minutes, this is the point where prepared SEC financial-statement evidence becomes an evaluated, auditable model signal—without pretending the model is the decision-maker.”

## Cell-by-cell narration

| Time | Cell / action | Narration |
|---|---|---|
| 0:00–0:20 | Show the title cell. | “`secfsn_plus_models` is Layer 04 of the SECFSN stack. It accepts a prepared company-quarter feature panel, produces evaluated predictions, and packages a clean handoff for the downstream decision system. It does not parse raw filings or make investment decisions.” |
| 0:20–0:35 | Run the setup cell. | “The notebook locates the repository root portably, imports only the production APIs used in the demo, and suppresses irrelevant warnings. There are no paths, secrets, or live services baked in.” |
| 0:35–0:55 | Read the representative-input cell, then run it. | “In a production stack, Layers 02 and 03 have already translated SEC financial statements into canonical features. This repository receives that company-period panel. For a stable recording, this preview is the project’s seeded synthetic fixture—not a real company filing.” |
| 0:55–1:20 | Point to the index, features, and input-source lines. | “Every row is a `(cik, period)` observation. The feature columns stand in for upstream financial-statement-derived evidence; price and sector are metadata. The deterministic seed makes the demonstration rerunnable without credentials or network access.” |
| 1:20–1:35 | Read the workflow cell. | “Now we run the actual Layer 04 orchestration rather than duplicating any logic in the notebook.” |
| 1:35–2:30 | Run the workflow-and-handoff cell. Point to each compact table in order. | “The production pipeline validates the panel, builds next-period-return labels, removes the final unlabelled period, separates metadata, and makes a chronological train/test split. It then trains the registered deterministic linear model and evaluates held-out cross-sectional predictions. The IC and RankIC are research diagnostics, not claims of alpha. The decile metric is honestly unavailable here because the synthetic test panel has only six issuers per period, fewer than the ten needed for period-level deciles. Finally, the adapter preserves the exact index and produces `model_prediction`, `model_expected_return`, and ranked `model_risk_probability` for Layer 05.” |
| 2:30–2:50 | Show the takeaway cell. | “The engineering strengths are reproducibility, explicit alignment contracts, and clear separation of evidence, model output, and final decision rules. Raw SEC ingest and decision-system policies live in other layers.” |

## Closing sound bite

“Layer 04 makes the modeling step reproducible and auditable, then hands a clearly bounded signal—not a final decision—to the decision system.”

## Honest limitation to mention

The repository contains no real SEC-derived feature matrix, raw SEC filing data, schemas, or saved downstream decision outputs. The notebook uses the repository’s official seeded synthetic fallback, so its metrics and values illustrate workflow behavior only. In particular, the small six-issuer test cross-section cannot calculate a period-level decile spread.

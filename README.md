# MPLADS Anomaly Detection — Build Status

Picking up from the last session: the two silent bugs are fixed, and the
full pipeline now runs end to end with a validated model.

## What was broken, and what fixed it

| Bug (from last session) | Root cause | Fix |
|---|---|---|
| "Duplicate work" flooding | Descriptions were too generic (`"Road Repair at HINGOLI"`), so unrelated legitimate works collided by chance | Added a plot/stretch number to every description; duplicate detection now keys on that number matching, not raw text similarity |
| "Stuck ongoing" flooding | Sanction dates were uniformly random over 2 years, so normal ongoing works crossed the 365-day threshold by chance, not because anything was actually wrong | Status now follows causally from a category-typical duration + noise; "stuck" is injected as a real anomaly (very old sanction date, still open), not a side effect of date spread |

## Current validated performance

Run `03_features_and_model.py` to reproduce. Latest run:

- **Recall: 100%** — every injected anomaly (cost outlier, duplicate work,
  impossible completion speed, stuck-ongoing) is caught.
- **Precision: 71%** (F1 0.83) — the remaining false positives are two
  explainable categories, not bugs:
  1. When a real duplicate is flagged, its legitimate sibling gets flagged
     too (a reviewer needs both sides of the pair — this is intentional).
  2. Natural statistical tail: with ~3,600 records, some legitimate costs
     land past 2.5 std-dev by chance. Expected behavior of any z-score
     threshold, and a good "here's the honest tradeoff" line for judges.

This precision/recall pair, plus the reasoning above, is your strongest
slide — it shows you stress-tested the model rather than just running it
once and reporting a number.

## Model artifact

`mplads_model.joblib` is a proper **sklearn Pipeline** (`StandardScaler`
→ `IsolationForest`), not a loose model object — one artifact, one source
of truth for preprocessing, loaded by both the training script and the
API's live-scoring endpoint. It also bundles the reference stats
(per-work-type cost mean/std, typical duration) the API needs for
`/score`. Load it with `joblib.load("mplads_model.joblib")`.

## Pipeline (run in order)

```
01_generate_mp_base.py     -> mp_base.csv (543 MPs, real state/seat split)
02_generate_works.py       -> mplads_synthetic_works.csv (~3,600 works, 79 labeled anomalies)
03_features_and_model.py   -> mplads_scored_works.csv, mplads_flagged_works.csv
04_api.py                  -> FastAPI service for the dashboard team (run separately)
```

If you get real per-work data from the MPLADS portal (the API endpoints
you reverse-engineered — `getTilesReportData` etc. — are worth one more
attempt with a dev checking the exact query params in the Request URL,
per the last session), drop it in as `real_allocated_limit.csv` before
step 1 and the schema downstream is compatible either way.

## Problem-statement coverage

**Solidly covered:** cost overruns, duplicate works, delayed/stuck
projects, risk-based alerts, unusual-pattern detection, MP/state
decision-support views.

**Closed in this pass:**
- District Authority view — `/districts/summary` (was missing entirely;
  added a `DISTRICT` field to the schema).
- "Trend analysis" — `/trends` gives month-over-month flagged rate,
  overall and by state. **This is retrospective trend monitoring, not
  forecasting** — say that explicitly to judges. A real forecast (e.g.
  "expected flags next month") needs a time-series model trained on more
  history than a hackathon dataset can honestly provide. Claiming
  "predictive insights" without that would be the kind of overreach a
  domain-savvy judge catches immediately.

**Still open:**
- Payment-tranche and asset-creation data (PS mentions both; current
  schema stops at sanction → completion).
- Actual alert *delivery* (email/notification) — still Dev 5's task.
- Role-based auth — data endpoints exist, access control doesn't yet.

## Framing for judges (say this, don't dodge it)

> "We validated our anomaly detection engine on a synthetic dataset with
> known ground truth, seeded with real MPLADS MP/state/constituency data
> pulled directly from mospi.gov.in. The same detection logic — hybrid
> rule engine + Isolation Forest, with explainable reason codes — is
> designed to run directly against live eSAKSHI work-level data once
> granular access is available. We chose measurable validation over an
> unverifiable claim of catching real fraud."

This pre-empts the "wait, is this real data?" question instead of getting
caught by it, and the hybrid-model design (ML for continuous features,
rules for binary/structural ones) is a real answer to "why should I trust
this score" — every flag comes with a plain-English reason, not just a
number.

## Next steps for the team

- **You**: tune the risk-score weights (`compute_risk` in
  `03_features_and_model.py`) against a couple more anomaly scenarios if
  time allows — e.g. MP-level aggregate flagging (an MP whose *rate* of
  flagged works is itself unusual), which is currently not modeled.
- **Dev 1 (Backend)**: point your API/auth layer at `04_api.py` — it's a
  working FastAPI service, not a stub. `/works`, `/mp/{id}/summary`,
  `/states/summary`, `/validation` are all live.
- **Dev 2**: if a teammate wants to take one more shot at the real
  portal API, the `getTilesReportData` request URL (Headers tab, not
  Response) is the thing to inspect for a query parameter you can edit
  directly in the browser.
- **Devs 3 & 4 (Frontend)**: `mplads_flagged_works.csv` is your seed data
  for building dashboard views before the API is wired up.
- **Dev 5**: `/validation` endpoint gives you the precision/recall numbers
  live, for the demo pitch.

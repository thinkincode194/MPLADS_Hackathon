"""
Phase 5: API layer -- hand this to your dev team. They build the
dashboard against these endpoints; they never need to touch the model.

Run locally with:
    pip install fastapi uvicorn pandas
    uvicorn 04_api:app --reload --port 8000

Then browse http://localhost:8000/docs for interactive API docs
(FastAPI auto-generates this -- show it live in the demo, judges like it).
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import date
from typing import Optional

app = FastAPI(title="MPLADS Anomaly Risk API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

DF = pd.read_csv("mplads_scored_works.csv")
MODEL_BUNDLE = joblib.load("mplads_model.joblib")  # trained Isolation Forest + reference stats


class NewWork(BaseModel):
    work_type: str
    sanctioned_cost: float
    sanction_date: str        # "YYYY-MM-DD"
    status: str                # "Sanctioned" | "Ongoing" | "Completed"
    completion_date: Optional[str] = None


@app.post("/score")
def score_new_work(work: NewWork):
    """
    Live scoring for a brand-new work submission -- this is the part
    that makes it a monitoring platform rather than a one-time batch
    report. A new sanction can be scored the moment it's entered,
    before funds are even released, not just discovered after the fact
    in a nightly re-run of the notebook.
    """
    stats = MODEL_BUNDLE["type_stats"].get(work.work_type)
    if stats is None:
        raise HTTPException(status_code=400, detail=f"Unknown work_type: {work.work_type}")

    mean_cost, std_cost = stats["TYPE_MEAN_COST"], stats["TYPE_STD_COST"] or 1
    cost_zscore = (work.sanctioned_cost - mean_cost) / std_cost

    speed_ratio = None
    if work.status == "Completed" and work.completion_date:
        sd = date.fromisoformat(work.sanction_date)
        cd = date.fromisoformat(work.completion_date)
        median_days = MODEL_BUNDLE["type_duration_median"].get(work.work_type, 150)
        speed_ratio = (cd - sd).days / median_days if median_days else None

    days_since_sanction = (date.today() - date.fromisoformat(work.sanction_date)).days
    typical = MODEL_BUNDLE["type_duration_median"].get(work.work_type, 150)
    stuck = work.status == "Ongoing" and days_since_sanction > 2.5 * typical

    feat = pd.DataFrame([{
        "cost_zscore": cost_zscore,
        "speed_ratio_filled": speed_ratio if speed_ratio is not None else 1.0,
    }])
    iso_pred = int(MODEL_BUNDLE["pipeline"].predict(feat)[0])

    score, reasons = 0, []
    if abs(cost_zscore) > 2.5:
        score += 35
        reasons.append(f"cost is {cost_zscore:.1f} std-dev from {work.work_type} median")
    if speed_ratio is not None and speed_ratio < 0.15:
        score += 30
        reasons.append(f"completed in {speed_ratio*100:.0f}% of typical duration")
    if stuck:
        score += 20
        reasons.append(f"ongoing {days_since_sanction}d, >2.5x typical duration")
    if iso_pred == -1:
        score += 10
        reasons.append("flagged as statistical outlier (Isolation Forest)")
    # NOTE: duplicate-site detection needs the full works table to compare
    # against, so it isn't checked in this single-record endpoint --
    # your devs should run new submissions through a periodic batch pass
    # of 03_features_and_model.py for that signal, or query /works?state=
    # for same-constituency works to compare client-side.

    return {
        "risk_score": min(score, 100),
        "flagged": score >= 20,
        "reasons": reasons if reasons else ["no anomaly signals"],
    }


@app.get("/")
def root():
    return {
        "status": "ok",
        "total_works": len(DF),
        "flagged_works": int(DF["FLAGGED"].sum()),
        "note": "Scores computed on a synthetic dataset seeded with real "
                "MPLADS MP/state/constituency data. See /validation for "
                "ground-truth precision/recall.",
    }


@app.get("/validation")
def validation():
    """Precision/recall against known injected anomalies -- for the slide."""
    tp = int(((DF["FLAGGED"]) & (DF["is_synthetic_anomaly"])).sum())
    fp = int(((DF["FLAGGED"]) & (~DF["is_synthetic_anomaly"])).sum())
    fn = int(((~DF["FLAGGED"]) & (DF["is_synthetic_anomaly"])).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0,
    }


@app.get("/works")
def list_works(
    state: Optional[str] = None,
    flagged_only: bool = False,
    min_risk: int = Query(0, ge=0, le=100),
    limit: int = Query(100, le=1000),
    offset: int = 0,
):
    df = DF
    if state:
        df = df[df["STATE"].str.lower() == state.lower()]
    if flagged_only:
        df = df[df["FLAGGED"]]
    df = df[df["RISK_SCORE"] >= min_risk]
    df = df.sort_values("RISK_SCORE", ascending=False)
    total = len(df)
    page = df.iloc[offset: offset + limit]
    return {
        "total": total,
        "returned": len(page),
        "results": page[[
            "WORK_ID", "MP_NAME", "STATE", "CONSTITUENCY", "WORK_TYPE",
            "WORK_DESCRIPTION", "SANCTIONED_COST", "STATUS", "SANCTION_DATE",
            "RISK_SCORE", "RISK_REASONS", "FLAGGED",
        ]].to_dict(orient="records"),
    }


@app.get("/works/{work_id}")
def get_work(work_id: str):
    row = DF[DF["WORK_ID"] == work_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Work not found")
    return row.iloc[0].to_dict()


@app.get("/mp/{mp_id}/summary")
def mp_summary(mp_id: str):
    df = DF[DF["MP_ID"] == mp_id]
    if df.empty:
        raise HTTPException(status_code=404, detail="MP not found")
    return {
        "mp_id": mp_id,
        "mp_name": df.iloc[0]["MP_NAME"],
        "state": df.iloc[0]["STATE"],
        "constituency": df.iloc[0]["CONSTITUENCY"],
        "total_works": len(df),
        "flagged_works": int(df["FLAGGED"].sum()),
        "total_sanctioned": float(df["SANCTIONED_COST"].sum()),
        "avg_risk_score": round(float(df["RISK_SCORE"].mean()), 1),
        "works": df.sort_values("RISK_SCORE", ascending=False)[
            ["WORK_ID", "WORK_TYPE", "RISK_SCORE", "FLAGGED", "STATUS"]
        ].to_dict(orient="records"),
    }


@app.get("/states/summary")
def states_summary():
    g = DF.groupby("STATE").agg(
        total_works=("WORK_ID", "count"),
        flagged_works=("FLAGGED", "sum"),
        total_sanctioned=("SANCTIONED_COST", "sum"),
        avg_risk=("RISK_SCORE", "mean"),
    ).reset_index().sort_values("flagged_works", ascending=False)
    g["avg_risk"] = g["avg_risk"].round(1)
    return g.to_dict(orient="records")


@app.get("/districts/summary")
def districts_summary(state: Optional[str] = None):
    """District Authority view -- the PS names District Authorities
    explicitly as a stakeholder; this rollup was missing entirely
    before, only MP- and state-level views existed."""
    df = DF
    if state:
        df = df[df["STATE"].str.lower() == state.lower()]
    g = df.groupby(["STATE", "DISTRICT"]).agg(
        total_works=("WORK_ID", "count"),
        flagged_works=("FLAGGED", "sum"),
        total_sanctioned=("SANCTIONED_COST", "sum"),
        avg_risk=("RISK_SCORE", "mean"),
    ).reset_index().sort_values("flagged_works", ascending=False)
    g["avg_risk"] = g["avg_risk"].round(1)
    return g.to_dict(orient="records")


@app.get("/trends")
def trends(state: Optional[str] = None, group_by_state: bool = False):
    """
    Month-over-month flagged-rate trend -- addresses the PS's
    'predictive insights / trend analysis' language, which nothing else
    in this API covers. NOTE: this is a *retrospective* trend (how the
    flagged rate has moved so far), not a forecast -- be upfront about
    that distinction with judges. A true forecast (e.g. next month's
    expected flag volume) would need a time-series model (Prophet/ARIMA)
    trained on more history than a hackathon-length dataset can offer
    honestly. Framing it as "early-warning trend monitoring" rather than
    "prediction" is the accurate claim.
    """
    df = DF.copy()
    df["SANCTION_DATE"] = pd.to_datetime(df["SANCTION_DATE"])
    df["month"] = df["SANCTION_DATE"].dt.to_period("M").astype(str)
    if state:
        df = df[df["STATE"].str.lower() == state.lower()]

    group_cols = ["month", "STATE"] if group_by_state else ["month"]
    g = df.groupby(group_cols).agg(
        total_works=("WORK_ID", "count"),
        flagged_works=("FLAGGED", "sum"),
    ).reset_index()
    g["flagged_rate"] = (g["flagged_works"] / g["total_works"]).round(3)
    g = g.sort_values(group_cols)
    return g.to_dict(orient="records")

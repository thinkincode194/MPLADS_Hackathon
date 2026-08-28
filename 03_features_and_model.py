"""
Phase 3 + 4: Feature engineering and hybrid anomaly detection.

Design rationale (this is the part to say out loud to judges):
Isolation Forest is good at continuous-feature outliers (cost, velocity)
but bad at binary/categorical patterns (near-duplicate description,
stuck status). So we split the problem:
  - ML (Isolation Forest) on continuous features -> cost_risk, speed_risk
  - Rule engine on binary/structural features -> duplicate_risk, stuck_risk
  - Combine into one weighted, explainable RISK_SCORE with reason codes
This hybrid is also what you'd actually defend against a domain-savvy
judge asking "why should I trust a black-box score" -- every flag comes
with a plain-English reason.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from difflib import SequenceMatcher
from datetime import date

TODAY = date(2026, 8, 28)

df = pd.read_csv("mplads_synthetic_works.csv", parse_dates=["SANCTION_DATE", "COMPLETION_DATE"])

# ---------------------------------------------------------------------
# Feature 1: Cost z-score within work type (are you paying 3x for the
# same category of work elsewhere in the country?)
# ---------------------------------------------------------------------
type_stats = df.groupby("WORK_TYPE")["SANCTIONED_COST"].agg(["mean", "std"]).reset_index()
type_stats.columns = ["WORK_TYPE", "TYPE_MEAN_COST", "TYPE_STD_COST"]
df = df.merge(type_stats, on="WORK_TYPE", how="left")
df["cost_zscore"] = (df["SANCTIONED_COST"] - df["TYPE_MEAN_COST"]) / df["TYPE_STD_COST"].replace(0, 1)

# ---------------------------------------------------------------------
# Feature 2: Completion speed vs category norm (days elapsed if completed;
# implausibly fast = red flag)
# ---------------------------------------------------------------------
df["days_to_complete"] = (df["COMPLETION_DATE"] - df["SANCTION_DATE"]).dt.days
type_duration = df[df["STATUS"] == "Completed"].groupby("WORK_TYPE")["days_to_complete"].median()
df["TYPE_MEDIAN_DAYS"] = df["WORK_TYPE"].map(type_duration)
df["speed_ratio"] = df["days_to_complete"] / df["TYPE_MEDIAN_DAYS"]
# only meaningful for completed works; NaN otherwise (handled below)

# ---------------------------------------------------------------------
# Feature 3 (rule-based): "stuck ongoing" -- sanctioned long ago, still
# ongoing, well past 2x the category's typical duration
# ---------------------------------------------------------------------
df["days_since_sanction"] = (pd.Timestamp(TODAY) - df["SANCTION_DATE"]).dt.days
type_typical = df.groupby("WORK_TYPE")["TYPE_MEDIAN_DAYS"].first()  # fallback lookup
df["stuck_flag"] = (
    (df["STATUS"] == "Ongoing") &
    (df["days_since_sanction"] > 2.5 * df["WORK_TYPE"].map(type_duration).fillna(150))
)

# ---------------------------------------------------------------------
# Feature 4 (rule-based): near-duplicate work filed within a short
# window of each other in the same constituency.
#
# Two signals, combined:
#   (a) plot/stretch number match -- same site referenced twice is the
#       strong, low-false-positive signal (two independent works only
#       share a plot number by ~1/999 chance)
#   (b) raw fuzzy description similarity -- kept as a *secondary*
#       corroborating signal and for the human-readable reason, but not
#       trusted alone: templated descriptions (same work type + same
#       landmark phrase, different site) can look 90%+ similar by
#       coincidence even when they're genuinely different works. That
#       was the actual bug that cost time in the original build -- the
#       fix is requiring the site identifier to match, not just prose.
# ---------------------------------------------------------------------
import re

def extract_plot(desc):
    m = re.search(r"#(\d+)", desc)
    return m.group(1) if m else None

def fuzzy_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

df["plot_no"] = df["WORK_DESCRIPTION"].apply(extract_plot)

dup_flags = pd.Series(False, index=df.index)
dup_reason = pd.Series("", index=df.index)

for (constituency, wtype), group in df.groupby(["CONSTITUENCY", "WORK_TYPE"]):
    if len(group) < 2:
        continue
    idxs = group.index.tolist()
    plots = group["plot_no"].tolist()
    descs = group["WORK_DESCRIPTION"].tolist()
    dates = group["SANCTION_DATE"].tolist()
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            date_gap = abs((dates[i] - dates[j]).days)
            if date_gap >= 90:
                continue
            same_plot = plots[i] is not None and plots[i] == plots[j]
            sim = fuzzy_ratio(descs[i], descs[j])
            if same_plot:
                dup_flags[idxs[i]] = True
                dup_flags[idxs[j]] = True
                dup_reason[idxs[i]] = f"same site as {df.loc[idxs[j], 'WORK_ID']} (plot #{plots[i]}, {sim:.0%} text match, {date_gap}d apart)"
                dup_reason[idxs[j]] = f"same site as {df.loc[idxs[i], 'WORK_ID']} (plot #{plots[j]}, {sim:.0%} text match, {date_gap}d apart)"

df["duplicate_flag"] = dup_flags
df["duplicate_reason"] = dup_reason

# ---------------------------------------------------------------------
# ML layer: proper sklearn Pipeline (StandardScaler -> IsolationForest),
# not a loose model + hand-rolled scaling. This is the artifact that
# gets saved and reused by the API -- one object, one source of truth
# for preprocessing, instead of duplicating scaling logic by hand in
# both the training script and the API's live-scoring endpoint.
# ---------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ml_features = df[["cost_zscore"]].copy()
ml_features["speed_ratio_filled"] = df["speed_ratio"].fillna(1.0)  # neutral for non-completed
ml_features = ml_features.fillna(0)

anomaly_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("iso_forest", IsolationForest(n_estimators=200, contamination=0.03, random_state=42)),
])
df["iso_pred"] = anomaly_pipeline.fit_predict(ml_features)        # -1 = outlier, 1 = normal
df["iso_score"] = -anomaly_pipeline.decision_function(ml_features)  # higher = more anomalous

# normalize iso_score to 0-1 for blending
df["iso_score_norm"] = (df["iso_score"] - df["iso_score"].min()) / (df["iso_score"].max() - df["iso_score"].min())

# ---------------------------------------------------------------------
# Combine into one explainable RISK_SCORE (0-100) with reason codes
# ---------------------------------------------------------------------
def compute_risk(row):
    score = 0
    reasons = []
    if abs(row["cost_zscore"]) > 2.5:
        score += 35
        reasons.append(f"cost is {row['cost_zscore']:.1f} std-dev from {row['WORK_TYPE']} median")
    if row["STATUS"] == "Completed" and pd.notna(row["speed_ratio"]) and row["speed_ratio"] < 0.15:
        score += 30
        reasons.append(f"completed in {row['speed_ratio']*100:.0f}% of typical duration")
    if row["stuck_flag"]:
        score += 20
        reasons.append(f"ongoing {row['days_since_sanction']}d, >2.5x typical duration")
    if row["duplicate_flag"]:
        score += 25
        reasons.append(f"near-duplicate work ({row['duplicate_reason']})")
    if row["iso_pred"] == -1:
        score += 10
        reasons.append("flagged as statistical outlier (Isolation Forest)")
    return min(score, 100), "; ".join(reasons) if reasons else "no anomaly signals"

results = df.apply(compute_risk, axis=1, result_type="expand")
df["RISK_SCORE"] = results[0]
df["RISK_REASONS"] = results[1]
df["FLAGGED"] = df["RISK_SCORE"] >= 20

# ---------------------------------------------------------------------
# Validate against known ground truth (only possible because it's
# synthetic -- this is the number that goes in your slide deck)
# ---------------------------------------------------------------------
tp = ((df["FLAGGED"]) & (df["is_synthetic_anomaly"])).sum()
fp = ((df["FLAGGED"]) & (~df["is_synthetic_anomaly"])).sum()
fn = ((~df["FLAGGED"]) & (df["is_synthetic_anomaly"])).sum()
tn = ((~df["FLAGGED"]) & (~df["is_synthetic_anomaly"])).sum()

precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print("=" * 60)
print("VALIDATION AGAINST KNOWN INJECTED ANOMALIES (ground truth)")
print("=" * 60)
print(f"Total works:            {len(df)}")
print(f"Known injected anomalies: {df['is_synthetic_anomaly'].sum()}")
print(f"Flagged by model:       {df['FLAGGED'].sum()}")
print(f"True Positives:  {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision:.1%}")
print(f"Recall:    {recall:.1%}")
print(f"F1 score:  {f1:.1%}")
print()
print("Recall by anomaly type:")
for atype in ["cost_outlier", "duplicate_work", "impossible_speed", "stuck_ongoing"]:
    sub = df[df["anomaly_type"] == atype]
    if len(sub):
        caught = sub["FLAGGED"].sum()
        print(f"  {atype:20s}: {caught}/{len(sub)} caught ({caught/len(sub):.0%})")

# Save outputs
df.to_csv("mplads_scored_works.csv", index=False)
flagged_df = df[df["FLAGGED"]].sort_values("RISK_SCORE", ascending=False)
flagged_df[["WORK_ID", "MP_NAME", "STATE", "CONSTITUENCY", "WORK_TYPE",
            "SANCTIONED_COST", "STATUS", "RISK_SCORE", "RISK_REASONS",
            "is_synthetic_anomaly"]].to_csv("mplads_flagged_works.csv", index=False)

print(f"\nSaved: mplads_scored_works.csv (all {len(df)} works, full features)")
print(f"Saved: mplads_flagged_works.csv ({len(flagged_df)} flagged works, demo-ready)")

# ---------------------------------------------------------------------
# Persist the trained model + the reference stats it needs (type means/
# std, typical durations) so the API can score a BRAND NEW work
# submission live, not just serve pre-computed rows. This is what makes
# it a "platform" rather than a one-time batch script.
# ---------------------------------------------------------------------
import joblib

joblib.dump({
    "pipeline": anomaly_pipeline,   # StandardScaler + IsolationForest, one object
    "type_stats": type_stats.set_index("WORK_TYPE").to_dict(orient="index"),
    "type_duration_median": type_duration.to_dict(),
    "iso_score_min": df["iso_score"].min(),
    "iso_score_max": df["iso_score"].max(),
}, "mplads_model.joblib")

print("Saved: mplads_model.joblib (sklearn Pipeline: StandardScaler + IsolationForest, plus reference stats for live scoring)")

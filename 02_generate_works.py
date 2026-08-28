"""
Phase 2: Generate per-work records for each MP, with realistic dates and
descriptions, then deliberately inject 4 anomaly types with ground-truth
labels for validating the detection model.

FIXES vs the earlier version:
1. Descriptions now include a specific location/landmark token so two
   *legitimate* works of the same type in the same constituency don't
   collide as false duplicates. Injected duplicates instead clone an
   EXISTING description near-verbatim (with a tiny edit) -- that's what
   a real duplicate-billing pattern looks like.
2. Dates are no longer uniformly random. Sanction dates are weighted
   toward being spread across the tenure so far, but a work's status
   determines its date logic causally (not the other way around):
   - Completed works: sanctioned in the past, completed after a
     category-typical duration (+ noise).
   - Ongoing works: sanctioned recently enough that still-being-ongoing
     is unremarkable under normal scheme dynamics.
   - "Stuck" anomalies are injected separately by taking a normal
     ongoing work and pushing its sanction date far into the past
     without a completion -- that's the actual anomaly signal, not a
     side effect of random spread.
"""
import pandas as pd
import numpy as np
import random
from datetime import date, timedelta

random.seed(7)
np.random.seed(7)

mp_df = pd.read_csv("mp_base.csv")

WORK_TYPES = {
    "Road Construction/Repair": {"median_cost": 1200000, "typical_days": 150},
    "Drinking Water Supply": {"median_cost": 800000, "typical_days": 90},
    "School Building/Classroom": {"median_cost": 1500000, "typical_days": 180},
    "Community Hall": {"median_cost": 2000000, "typical_days": 200},
    "Drainage/Sewage": {"median_cost": 700000, "typical_days": 100},
    "Streetlight Installation": {"median_cost": 300000, "typical_days": 45},
    "Health Sub-Centre Upgrade": {"median_cost": 1000000, "typical_days": 160},
    "Anganwadi Centre": {"median_cost": 600000, "typical_days": 100},
    "Irrigation Canal": {"median_cost": 1800000, "typical_days": 220},
    "Public Toilet Complex": {"median_cost": 400000, "typical_days": 60},
}

AGENCIES = ["PWD Division", "Zilla Panchayat", "Municipal Corporation",
    "Rural Engineering Services", "State Water Board", "District Rural Dev. Agency",
    "Block Development Office", "State Electricity Board"]

LANDMARKS = ["near Govt School", "at Ward 3", "near Bus Stand", "at Main Chowk",
    "near Primary Health Centre", "in Ward 7", "near Panchayat Bhavan",
    "at Village Entry Point", "near Community Centre", "in Sector 4",
    "near Railway Crossing", "at Market Road", "in Ward 12", "near Temple Road",
    "at Colony Gate", "near Bus Depot"]

def district_for_constituency(constituency_str):
    """
    Real Lok Sabha constituencies don't map 1:1 to districts (some
    districts span multiple PCs, some PCs span multiple districts) --
    for demo purposes, group every ~4 constituencies in a state into one
    district. Good enough to exercise a District Authority view, which
    the problem statement names explicitly and the earlier build didn't
    have at all.
    """
    # constituency_str like "Uttar Pradesh PC-40"
    state, pc = constituency_str.rsplit(" PC-", 1)
    district_num = (int(pc) - 1) // 4 + 1
    return f"{state} District {district_num}"

TENURE_START = date(2024, 6, 1)
TODAY = date(2026, 8, 28)  # "as on date" reference for the demo

rows = []
work_id = 1
anomaly_log = []

def add_work(mp, work_type, sanction_date, cost, status, completion_date,
             agency, landmark, is_anom=False, anom_type=None, dup_of=None):
    global work_id
    # Plot/stretch number adds real entropy so two legitimate same-type
    # works in the same constituency don't read as near-duplicates by
    # chance -- this is what a real work-order description looks like
    # (specific enough to be checked on the ground).
    plot_no = random.randint(1, 999)
    desc = f"{work_type} {landmark} #{plot_no}, {mp['CONSTITUENCY']}"
    rows.append({
        "WORK_ID": f"W{work_id:06d}",
        "MP_ID": mp["MP_ID"],
        "MP_NAME": mp["MP_NAME"],
        "STATE": mp["STATE"],
        "CONSTITUENCY": mp["CONSTITUENCY"],
        "DISTRICT": district_for_constituency(mp["CONSTITUENCY"]),
        "WORK_TYPE": work_type,
        "WORK_DESCRIPTION": desc,
        "IMPLEMENTING_AGENCY": agency,
        "SANCTIONED_COST": round(cost),
        "SANCTION_DATE": sanction_date.isoformat(),
        "STATUS": status,
        "COMPLETION_DATE": completion_date.isoformat() if completion_date else "",
        "is_synthetic_anomaly": is_anom,
        "anomaly_type": anom_type if is_anom else "",
    })
    if is_anom:
        anomaly_log.append((work_id, anom_type))
    work_id += 1

for _, mp in mp_df.iterrows():
    n_works = random.randint(4, 9)
    mp_work_descs = []  # track this MP's real descriptions for dup injection

    for _ in range(n_works):
        wtype = random.choice(list(WORK_TYPES.keys()))
        meta = WORK_TYPES[wtype]
        agency = random.choice(AGENCIES)
        landmark = random.choice(LANDMARKS)

        # Sanction date: spread across tenure-so-far, weighted toward earlier
        # (more works get sanctioned/finished the longer the tenure runs)
        max_days_elapsed = (TODAY - TENURE_START).days
        days_in = int(np.random.beta(2, 2) * max_days_elapsed)
        sanction_date = TENURE_START + timedelta(days=days_in)

        # Normal cost with mild variance
        cost = np.random.normal(meta["median_cost"], meta["median_cost"] * 0.15)
        cost = max(cost, meta["median_cost"] * 0.5)

        days_since_sanction = (TODAY - sanction_date).days
        typical_duration = int(np.random.normal(meta["typical_days"], meta["typical_days"] * 0.2))
        typical_duration = max(typical_duration, 20)

        # Status follows causally from whether typical duration has elapsed
        if days_since_sanction >= typical_duration and random.random() < 0.85:
            status = "Completed"
            completion_date = sanction_date + timedelta(days=typical_duration)
            if completion_date > TODAY:
                completion_date = TODAY
        elif days_since_sanction >= typical_duration * 1.5:
            status = "Completed"
            completion_date = sanction_date + timedelta(days=typical_duration)
        else:
            status = "Ongoing"
            completion_date = None

        add_work(mp, wtype, sanction_date, cost, status, completion_date, agency, landmark)
        mp_work_descs.append((wtype, agency, sanction_date, rows[-1]["WORK_DESCRIPTION"]))

    # ---- Inject anomalies (~15% of MPs get one injected anomaly each) ----
    if random.random() < 0.15 and mp_work_descs:

        anom_choice = random.choice(["cost_outlier", "duplicate_work",
                                      "impossible_speed", "stuck_ongoing"])

        if anom_choice == "cost_outlier":
            wtype = random.choice(list(WORK_TYPES.keys()))
            meta = WORK_TYPES[wtype]
            inflated_cost = meta["median_cost"] * random.uniform(3.2, 5.5)
            sd = TENURE_START + timedelta(days=random.randint(30, max_days_elapsed))
            add_work(mp, wtype, sd, inflated_cost, "Sanctioned", None,
                      random.choice(AGENCIES), random.choice(LANDMARKS),
                      True, "cost_outlier")

        elif anom_choice == "duplicate_work" and len(mp_work_descs) >= 1:
            src_type, src_agency, src_date, src_desc = random.choice(mp_work_descs)
            meta = WORK_TYPES[src_type]
            # near-duplicate: same work, filed again ~2-6 weeks later,
            # same or near-identical description, cost within normal range
            dup_date = src_date + timedelta(days=random.randint(14, 42))
            dup_cost = np.random.normal(meta["median_cost"], meta["median_cost"] * 0.1)
            work_id_before = work_id
            add_work(mp, src_type, dup_date, dup_cost, "Sanctioned", None,
                      src_agency, "near same location", True, "duplicate_work")
            # force near-identical description (this is the actual signal)
            rows[-1]["WORK_DESCRIPTION"] = src_desc + " (Phase II)"

        elif anom_choice == "impossible_speed":
            wtype = random.choice(list(WORK_TYPES.keys()))
            meta = WORK_TYPES[wtype]
            sd = TENURE_START + timedelta(days=random.randint(60, max_days_elapsed - 10))
            fake_duration = random.randint(2, 6)  # days -- physically implausible
            cd = sd + timedelta(days=fake_duration)
            add_work(mp, wtype, sd, meta["median_cost"], "Completed", cd,
                      random.choice(AGENCIES), random.choice(LANDMARKS),
                      True, "impossible_speed")

        elif anom_choice == "stuck_ongoing":
            wtype = random.choice(list(WORK_TYPES.keys()))
            meta = WORK_TYPES[wtype]
            # sanctioned very early in tenure, still "Ongoing" today --
            # far past typical_days * 3, the actual anomaly signal
            sd = TENURE_START + timedelta(days=random.randint(0, 30))
            add_work(mp, wtype, sd, meta["median_cost"], "Ongoing", None,
                      random.choice(AGENCIES), random.choice(LANDMARKS),
                      True, "stuck_ongoing")

df = pd.DataFrame(rows)
df.to_csv("mplads_synthetic_works.csv", index=False)

print(f"Generated {len(df)} work records across {mp_df.shape[0]} MPs")
print(f"Injected anomalies: {df['is_synthetic_anomaly'].sum()} "
      f"({df['is_synthetic_anomaly'].mean()*100:.1f}%)")
print(df['anomaly_type'].value_counts())
print()
print("Status distribution:")
print(df['STATUS'].value_counts())

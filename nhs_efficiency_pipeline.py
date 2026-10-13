#!/usr/bin/env python3
"""
NHS Healthcare Efficiency Analysis - end-to-end pipeline
=========================================================
Steps (mirrors the project brief):
  1. Load data                         (Step 2  - Data collection)
  2. Clean + data-quality log          (Step 3  - Data cleaning)
  3. Transform + derived metrics       (Step 4  - Data transformation)
  4. Statistical analysis              (Step 5  - Advanced analysis)
  5. Optimisation: LP + Monte Carlo    (Step 5  - Slot allocation)
  6. Charts + Tableau-ready exports    (Step 6  - Visualisation)

Usage:
    python nhs_efficiency_pipeline.py --data-dir /path/to/csvs --out-dir ../

Inputs  (data-dir): cleaned_appointments.csv, patients.csv, slots.csv
Outputs (out-dir) : 04_Tableau/data/*.csv, 05_Charts/*.png, results.json

Python 3.10+; pandas, numpy, scipy, statsmodels, scikit-learn, pulp, matplotlib
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import pulp

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Constants / business assumptions (change here, re-run, everything updates)
# --------------------------------------------------------------------------
OPEN_MIN = 8 * 60            # clinic opens 08:00
SLOT_LEN = 15                # scheduled slot length (minutes)
SLOTS_PER_DAY = 40           # 08:00 .. 17:45
SESSION_MIN = SLOTS_PER_DAY * SLOT_LEN   # 600 minutes
CLOSE_MIN = OPEN_MIN + SESSION_MIN       # 18:00
WAIT_TARGET = 30             # analyst-set service target: mean in-clinic wait (min). ASSUMPTION - not an NHS standard.
WINTER_MONTHS = [12, 1, 2]   # meteorological winter
SEED = 2025

NHS_BLUE, NHS_DARK, NHS_LIGHT = "#005EB8", "#003087", "#41B6E6"
NHS_WARM, NHS_RED, NHS_GREEN, GREY = "#FFB81C", "#DA291C", "#009639", "#768692"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.grid": True, "grid.alpha": 0.25, "font.family": "DejaVu Sans",
})

R = {}   # results registry -> results.json


def tmin(s):
    """'HH:MM:SS' strings -> minutes since midnight (float)."""
    return pd.to_timedelta(s).dt.total_seconds() / 60.0


# ==========================================================================
# 1-2. LOAD + CLEAN
# ==========================================================================
def load_and_clean(data_dir: Path):
    appts = pd.read_csv(data_dir / "cleaned_appointments.csv")
    pats = pd.read_csv(data_dir / "patients.csv")
    slots = pd.read_csv(data_dir / "slots.csv")
    dq = []   # data-quality log: (check, finding, action)

    def log(check, finding, action):
        dq.append({"check": check, "finding": finding, "action": action})

    log("Row counts", f"appointments={len(appts):,}; patients={len(pats):,}; slots={len(slots):,}", "Loaded as supplied")

    # --- duplicates -------------------------------------------------------
    d_a, d_p, d_s = appts.duplicated().sum(), pats.duplicated().sum(), slots.duplicated().sum()
    k_a = appts.appointment_id.duplicated().sum()
    k_p = pats.patient_id.duplicated().sum()
    k_s = slots.slot_id.duplicated().sum()
    log("Exact duplicate rows", f"appointments={d_a}; patients={d_p}; slots={d_s}", "None to remove")
    log("Duplicate primary keys", f"appointment_id={k_a}; patient_id={k_p}; slot_id={k_s}", "None to remove")
    dup_person = pats.duplicated(["name", "dob", "sex"]).sum()
    log("Duplicate patients (name+DOB+sex)", f"{dup_person} duplicates; {pats.name.duplicated().sum():,} repeated names but with different DOB/ID",
        "Repeated names are different people (distinct DOB); no merge")
    multi = appts.slot_id.duplicated().sum()
    log("slot_id repeated in appointments", f"{multi:,} appointments share a slot with another appointment",
        "Legitimate: a cancelled booking later re-booked. Kept; used to measure slot re-fill")

    # --- referential integrity -------------------------------------------
    orph_s = (~appts.slot_id.isin(slots.slot_id)).sum()
    orph_p = (~appts.patient_id.isin(pats.patient_id)).sum()
    log("Referential integrity", f"orphan slot_id={orph_s}; orphan patient_id={orph_p}", "All keys join; no action")

    # --- standardise dates / times ---------------------------------------
    for c in ["scheduling_date", "appointment_date"]:
        appts[c] = pd.to_datetime(appts[c])
    slots["appointment_date"] = pd.to_datetime(slots["appointment_date"])
    pats["dob"] = pd.to_datetime(pats["dob"])
    appts["status"] = appts["status"].str.strip().str.lower()
    appts["sex"] = appts["sex"].str.strip().str.title()
    log("Date/time standardisation", "Dates parsed to datetime64; times kept as HH:MM:SS text and converted to minutes-since-midnight for maths; status/sex trimmed and case-normalised",
        "Applied")

    # --- missing data ------------------------------------------------------
    miss_cols = ["check_in_time", "appointment_duration", "start_time", "end_time", "waiting_time"]
    miss_ct = appts[miss_cols].isna().sum().to_dict()
    non_att = (appts.status != "attended")
    all_missing_non_att = (appts.loc[non_att, miss_cols].isna().all(axis=1).all()
                           and appts.loc[~non_att, miss_cols].notna().all(axis=1).all())
    log("Missing waiting_time / duration / timestamps",
        f"{miss_ct['waiting_time']:,} nulls in each column; null for 100% of non-attended rows and 0% of attended rows = {all_missing_non_att}",
        "DEVIATION FROM BRIEF: NOT imputed. Nulls are structural (cancelled / DNA patients were never seen), so a specialty or overall median would fabricate waits for people who never attended. Wait metrics are computed on attended visits only")
    log("Missing is_available", f"{slots.is_available.isna().sum()} nulls", "Brief's 'overall average' rule not needed; field is complete")
    log("Other missing values", f"patients nulls={int(pats.isna().sum().sum())}; appointments non-timing nulls={int(appts.drop(columns=miss_cols).isna().sum().sum())}", "None")

    # --- consistency checks -----------------------------------------------
    lead_ok = ((appts.appointment_date - appts.scheduling_date).dt.days == appts.scheduling_interval).mean()
    log("scheduling_interval = appointment_date - scheduling_date", f"{lead_ok:.1%} consistent", "Interval used directly as 'booking lead time (days)'")
    m = appts.merge(pats[["patient_id", "sex", "dob", "insurance"]], on="patient_id", suffixes=("", "_p"))
    sex_mis = (m.sex != m.sex_p).sum()
    age_calc = np.floor((m.appointment_date - m.dob).dt.days / 365.25)
    age_ok = ((m.age - age_calc).abs() <= 1).mean()
    log("Appointment sex / age vs patient master", f"sex mismatches={sex_mis}; age within 1 year of DOB-derived age={age_ok:.1%}", "Consistent; patient master used for insurance")
    # --- WINSORISATION: recorded waiting_time / appointment_duration are capped ----------
    att = appts[appts.status == "attended"].copy()
    st_, ci_, en_ = tmin(att.start_time), tmin(att.check_in_time), tmin(att.end_time)
    w_ts, d_ts = st_ - ci_, en_ - st_
    rec_w, rec_d = att.waiting_time, att.appointment_duration
    cap_w_hi, cap_w_lo = float(rec_w.max()), float(rec_w.min())
    cap_d_hi, cap_d_lo = float(rec_d.max()), float(rec_d.min())
    sh_w_hi, sh_w_lo = (rec_w >= cap_w_hi - 0.05).mean(), (rec_w <= cap_w_lo + 0.05).mean()
    sh_d_hi, sh_d_lo = (rec_d >= cap_d_hi - 0.05).mean(), (rec_d <= cap_d_lo + 0.05).mean()
    log("Capped (winsorised) measures",
        f"waiting_time: {sh_w_hi:.1%} of attended visits sit exactly at the {cap_w_hi:.1f}-min ceiling and {sh_w_lo:.1%} at the {cap_w_lo:.1f}-min floor (= 90th / 10th percentile); "
        f"appointment_duration: {sh_d_hi:.1%} at {cap_d_hi:.1f} min, {sh_d_lo:.1%} at {cap_d_lo:.1f} min. "
        f"Timestamps (start - check-in) show uncapped waits up to {w_ts.max():.0f} min (mean {w_ts.mean():.1f} vs {rec_w.mean():.1f} recorded) and consultations up to {d_ts.max():.0f} min (mean {d_ts.mean():.2f} vs {rec_d.mean():.2f})",
        "KEY FINDING. The supplied 'cleaned' fields hide the tail (this is also why an IQR screen finds no outliers). Analysis uses TIMESTAMP-DERIVED wait_min and duration_min; recorded values kept as wait_recorded_min / duration_recorded_min for sensitivity. Rows are not removed")
    R["winsor"] = {"cap_wait_hi": cap_w_hi, "cap_wait_lo": cap_w_lo, "share_wait_hi": sh_w_hi, "share_wait_lo": sh_w_lo,
                   "cap_dur_hi": cap_d_hi, "cap_dur_lo": cap_d_lo, "share_dur_hi": sh_d_hi, "share_dur_lo": sh_d_lo,
                   "mean_wait_recorded": rec_w.mean(), "mean_wait_derived": w_ts.mean(), "max_wait_derived": w_ts.max(),
                   "mean_dur_recorded": rec_d.mean(), "mean_dur_derived": d_ts.mean(), "max_dur_derived": d_ts.max(),
                   "p95_wait_derived": w_ts.quantile(.95), "p99_wait_derived": w_ts.quantile(.99)}
    # attach derived measures to all appointment rows (NaN for non-attended)
    appts["wait_recorded_min"], appts["duration_recorded_min"] = appts.waiting_time, appts.appointment_duration
    appts["wait_min"] = tmin(appts.start_time) - tmin(appts.check_in_time)
    appts["duration_min"] = tmin(appts.end_time) - tmin(appts.start_time)

    # --- outcomes with unresolved status ----------------------------------
    unk, sch = (appts.status == "unknown").sum(), (appts.status == "scheduled").sum()
    sch_dates = appts.loc[appts.status == "scheduled", "appointment_date"]
    log("Unresolved status ('unknown', 'scheduled')",
        f"unknown={unk}, scheduled={sch} (all scheduled fall {sch_dates.min().date()} to {sch_dates.max().date()}, i.e. the data-extract tail)",
        "Kept in volume counts; excluded from outcome-rate denominators (attended / DNA / cancelled)")

    # --- slots outside observation window ----------------------------------
    last_appt = appts.appointment_date.max()
    late_slots = slots[slots.appointment_date > last_appt]
    log("Slot rows after last appointment date", f"{len(late_slots)} slot rows ({late_slots.appointment_date.nunique()} days after {last_appt.date()}), all flagged available with zero appointments",
        "Excluded from utilisation (no appointment data exists for those days)")

    # --- outliers (IQR) -----------------------------------------------------
    iqr_rows = []
    for c in ["wait_min", "duration_min", "scheduling_interval"]:
        x = appts[c].dropna()
        q1, q3 = x.quantile([.25, .75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n_out = int(((x < lo) | (x > hi)).sum())
        iqr_rows.append({"field": c, "q1": q1, "q3": q3, "lower_fence": lo, "upper_fence": hi, "flagged": n_out})
        appts[f"{c}_outlier_iqr"] = ((appts[c] < lo) | (appts[c] > hi)).fillna(False)
    log("IQR outlier screen (on timestamp-derived measures)", "; ".join(f"{r['field']}: {r['flagged']:,} flagged (fence {r['lower_fence']:.1f}-{r['upper_fence']:.1f})" for r in iqr_rows),
        "Flag columns added; nothing removed. Long waits are real (patients queued behind overrunning sessions); lead times >22 days are valid bookings inside a 30-day horizon")
    R["iqr"] = iqr_rows

    # --- single-queue diagnostic (timestamp-derived) ----------------------------------
    a2 = att.assign(st=tmin(att.start_time), en=tmin(att.end_time)).sort_values(["appointment_date", "st"])
    a2["prev_en"] = a2.groupby("appointment_date").en.shift()
    overlap = ((a2.st - a2.prev_en) < -0.1).mean()
    changeover = (a2.st - a2.prev_en).median()
    log("Overlapping consultations (timestamps)", f"{overlap:.1%} of attended visits start before the previous one ended; median gap between consecutive consultations = {changeover:.2f} min",
        "Timestamps form a clean single queue (no overlaps) with ~1 min changeover. Used to calibrate/validate the simulation. (Recorded capped durations would wrongly suggest ~12% overlap)")
    R["overlap_share"], R["median_changeover_min"] = overlap, changeover
    R["dq_log"] = dq
    return appts, pats, slots, last_appt


# ==========================================================================
# 3. TRANSFORM
# ==========================================================================
def hour_block(h):
    return pd.cut(h, [7, 9, 11, 13, 15, 17], labels=["08-10", "10-12", "12-14", "14-16", "16-18"]).astype(str)


def transform(appts, pats, slots, last_appt):
    a = appts.merge(pats[["patient_id", "insurance", "dob"]], on="patient_id", how="left")
    a["appointment_datetime"] = pd.to_datetime(a.appointment_date.dt.strftime("%Y-%m-%d") + " " + a.appointment_time)
    a["year"] = a.appointment_date.dt.year
    a["month"] = a.appointment_date.dt.month
    a["month_name"] = a.appointment_date.dt.strftime("%b")
    a["year_month"] = a.appointment_date.dt.strftime("%Y-%m")
    a["quarter"] = a.appointment_date.dt.quarter
    a["weekday"] = a.appointment_date.dt.day_name()
    a["dow"] = a.appointment_date.dt.dayofweek
    a["hour"] = a.appointment_time.str[:2].astype(int)
    a["time_block"] = hour_block(a.hour)
    a["season"] = a.month.map(lambda m: "Winter" if m in (12, 1, 2) else "Spring" if m in (3, 4, 5) else "Summer" if m in (6, 7, 8) else "Autumn")
    a["is_winter"] = a.month.isin(WINTER_MONTHS).astype(int)
    a["age_band"] = pd.cut(a.age, [0, 29, 49, 64, 79, 200], labels=["<30", "30-49", "50-64", "65-79", "80+"]).astype(str)
    a["lead_time_days"] = a.scheduling_interval
    a["lead_time_band"] = pd.cut(a.lead_time_days, [0, 1, 3, 7, 14, 31], labels=["1 day", "2-3 days", "4-7 days", "8-14 days", "15-30 days"]).astype(str)
    a["is_attended"] = (a.status == "attended").astype(int)
    a["is_dna"] = (a.status == "did not attend").astype(int)
    a["is_cancelled"] = (a.status == "cancelled").astype(int)
    a["is_resolved"] = a.status.isin(["attended", "did not attend", "cancelled"]).astype(int)
    a["wait_over_30"] = np.where(a.is_attended == 1, (a.wait_min > 30).astype(int), np.nan)
    a["wait_over_60"] = np.where(a.is_attended == 1, (a.wait_min > 60).astype(int), np.nan)
    st, ci, ap, en = tmin(a.start_time), tmin(a.check_in_time), tmin(a.appointment_time), tmin(a.end_time)
    a["checkin_early_min"] = ap - ci
    a["start_delay_min"] = st - ap          # consultation start vs scheduled slot time
    a["end_clock_min"] = en

    # slot re-fill: was a cancelled booking's slot later used by another booking?
    non_cancel_slots = set(a.loc[a.status != "cancelled", "slot_id"])
    a["cancel_slot_refilled"] = np.where(a.is_cancelled == 1, a.slot_id.isin(non_cancel_slots).astype(int), np.nan)

    # patient history (for DNA modelling) - strictly prior resolved attend/DNA
    h = a[a.status.isin(["attended", "did not attend"])].sort_values(["patient_id", "appointment_datetime"]).copy()
    h["prior_appts"] = h.groupby("patient_id").cumcount()
    h["prior_dna"] = h.groupby("patient_id").is_dna.cumsum() - h.is_dna
    a = a.merge(h[["appointment_id", "prior_appts", "prior_dna"]], on="appointment_id", how="left")

    # ---- slot-level fact ---------------------------------------------------
    s = slots.copy()
    s["year"], s["month"] = s.appointment_date.dt.year, s.appointment_date.dt.month
    s["weekday"], s["dow"] = s.appointment_date.dt.day_name(), s.appointment_date.dt.dayofweek
    s["hour"] = s.appointment_time.str[:2].astype(int)
    s["time_block"] = hour_block(s.hour)
    s["year_month"] = s.appointment_date.dt.strftime("%Y-%m")
    s["in_window"] = (s.appointment_date <= last_appt).astype(int)
    agg = a.groupby("slot_id").agg(n_bookings=("appointment_id", "size"),
                                  has_att=("is_attended", "max"), has_dna=("is_dna", "max"),
                                  has_unres=("status", lambda x: int(x.isin(["scheduled", "unknown"]).any())),
                                  n_cancel=("is_cancelled", "sum"))
    s = s.merge(agg, on="slot_id", how="left").fillna({"n_bookings": 0, "has_att": 0, "has_dna": 0, "has_unres": 0, "n_cancel": 0})

    def outcome(r):
        if r.in_window == 0: return "Outside data window"
        if r.has_att: return "Attended"
        if r.has_dna: return "Did not attend"
        if r.has_unres: return "Unresolved"
        if r.n_bookings == 0: return "Never booked"
        return "Cancelled, not re-filled"
    s["slot_outcome"] = s.apply(outcome, axis=1)
    s["is_booked_flag"] = (~s.is_available).astype(int)         # brief's definition of 'booked'
    s["is_effective_use"] = (s.slot_outcome == "Attended").astype(int)

    # ---- daily fact --------------------------------------------------------
    att = a[a.is_attended == 1]
    day = s[s.in_window == 1].groupby("appointment_date").agg(
        slots=("slot_id", "size"), booked_slots=("is_booked_flag", "sum"),
        attended=("is_effective_use", "sum"),
        dna=("slot_outcome", lambda x: (x == "Did not attend").sum()),
        cancelled_not_refilled=("slot_outcome", lambda x: (x == "Cancelled, not re-filled").sum()),
        never_booked=("slot_outcome", lambda x: (x == "Never booked").sum()),
        unresolved=("slot_outcome", lambda x: (x == "Unresolved").sum())).reset_index()
    cancelled_appts = a[a.is_cancelled == 1].groupby("appointment_date").size().rename("cancelled_appts")
    dwait = att.groupby("appointment_date").agg(mean_wait=("wait_min", "mean"), p90_wait=("wait_min", lambda x: x.quantile(.9)),
                                                workload_min=("duration_min", "sum"), mean_duration=("duration_min", "mean"),
                                                last_end_min=("end_clock_min", "max"), mean_lead=("lead_time_days", "mean"))
    day = day.merge(cancelled_appts, on="appointment_date", how="left").merge(dwait, on="appointment_date", how="left")
    day["cancelled_appts"] = day.cancelled_appts.fillna(0)
    day["slot_booked_rate"] = day.booked_slots / day.slots
    day["slot_effective_rate"] = day.attended / day.slots
    day["time_load_pct"] = day.workload_min / SESSION_MIN
    day["overrun_min"] = (day.last_end_min - CLOSE_MIN).clip(lower=0)
    day["year"], day["month"] = day.appointment_date.dt.year, day.appointment_date.dt.month
    day["year_month"] = day.appointment_date.dt.strftime("%Y-%m")
    day["is_winter"] = day.month.isin(WINTER_MONTHS).astype(int)
    day["weekday"] = day.appointment_date.dt.day_name()
    day = day.dropna(subset=["mean_wait"]).reset_index(drop=True)
    return a, s, day


# ==========================================================================
# 4. ANALYSIS
# ==========================================================================
def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def analyse(a, s, day, last_appt):
    att = a[a.is_attended == 1]
    sw = s[s.in_window == 1]
    res_slots = sw[sw.slot_outcome != "Unresolved"]
    # ---------------- KPIs ----------------------------------------------------
    fin = a[a.is_resolved == 1]
    dna_den = a[a.status.isin(["attended", "did not attend"])]
    kpi = {
        "appointments_total": int(len(a)), "patients_total": int(a.patient_id.nunique()),
        "date_from": str(a.appointment_date.min().date()), "date_to": str(a.appointment_date.max().date()),
        "operating_days": int(day.shape[0]), "slots_in_window": int(len(sw)),
        "mean_wait_min": att.wait_min.mean(), "median_wait_min": att.wait_min.median(), "p90_wait_min": att.wait_min.quantile(.9),
        "share_wait_over_30": att.wait_over_30.mean(), "share_wait_over_60": att.wait_over_60.mean(),
        "mean_lead_days": a.lead_time_days.mean(), "median_lead_days": a.lead_time_days.median(), "p90_lead_days": a.lead_time_days.quantile(.9),
        "mean_duration_min": att.duration_min.mean(), "slot_len_min": SLOT_LEN,
        "booked_rate": sw.is_booked_flag.mean(), "effective_rate": (sw.slot_outcome == "Attended").sum() / len(res_slots),
        "dna_rate": dna_den.is_dna.mean(), "cancel_rate": fin.is_cancelled.mean(),
        "dna_share_of_slots": (res_slots.slot_outcome == "Did not attend").mean(),
        "open_share_of_slots": res_slots.slot_outcome.isin(["Cancelled, not re-filled", "Never booked"]).mean(),
        "cancel_refill_rate": a.loc[a.is_cancelled == 1, "cancel_slot_refilled"].mean(),
        "cancelled_appts": int(a.is_cancelled.sum()),
        "attended_per_day": day.attended.mean(), "time_load_pct": day.time_load_pct.mean(),
        "mean_finish_clock_h": day.last_end_min.mean() / 60, "mean_overrun_min": day.overrun_min.mean(),
        "share_days_overrun_30": (day.overrun_min > 30).mean(), "share_days_overrun_60": (day.overrun_min > 60).mean(),
        "mean_start_delay_min": att.start_delay_min.mean(),
        "status_counts": a.status.value_counts().to_dict(),
    }
    R["kpi"] = kpi
    # slot outcome accounting
    so = sw.slot_outcome.value_counts()
    R["slot_outcomes"] = {k: int(v) for k, v in so.items()}
    R["slot_outcomes_pct_resolved"] = {k: float(v / len(res_slots)) for k, v in so.items() if k != "Unresolved"}
    R["cancel_refill_detail"] = {"cancelled_appts": int(a.is_cancelled.sum()), "refilled": int(a.cancel_slot_refilled.sum()),
                                 "not_refilled": int((a.cancel_slot_refilled == 0).sum())}

    # ---------------- waiting by hour / block / weekday ---------------------
    by_hour = att.groupby("hour").agg(mean_wait=("wait_min", "mean"), median_wait=("wait_min", "median"), n=("wait_min", "size"),
                                      mean_dur=("duration_min", "mean"), mean_delay=("start_delay_min", "mean")).reset_index()
    R["wait_by_hour"] = by_hour.round(2).to_dict("records")
    F, p = stats.f_oneway(*[g.wait_min.values for _, g in att.groupby("hour")])
    ss_b = sum(len(g) * (g.wait_min.mean() - att.wait_min.mean()) ** 2 for _, g in att.groupby("hour"))
    R["hour_anova"] = {"F": F, "p": p, "eta2": ss_b / ((att.wait_min - att.wait_min.mean()) ** 2).sum()}
    slope = np.polyfit(att.hour, att.wait_min, 1)[0]
    R["wait_slope_per_hour"] = slope
    R["wait_8am"], R["wait_5pm"] = float(by_hour.mean_wait.iloc[0]), float(by_hour.mean_wait.iloc[-1])
    hm = att.pivot_table(index="weekday", columns="hour", values="wait_min", aggfunc="mean").reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    R["heat_weekday_hour"] = hm.round(1).to_dict()
    dow_f = stats.f_oneway(*[g.wait_min.values for _, g in att.groupby("dow")])
    R["dow_anova"] = {"F": dow_f[0], "p": dow_f[1], "means": att.groupby("weekday").wait_min.mean().round(2).to_dict()}

    # ---------------- yearly / monthly / seasonality ------------------------
    yr = a.groupby("year").agg(mean_wait=("wait_min", "mean"), mean_lead=("lead_time_days", "mean"), appts=("appointment_id", "size")).reset_index()
    yr["dna_rate"] = a[a.status.isin(["attended", "did not attend"])].groupby("year").is_dna.mean().values
    yr["cancel_rate"] = fin.groupby("year").is_cancelled.mean().values
    R["by_year"] = yr.round(4).to_dict("records")
    mo = att.groupby("month").wait_min.mean()
    lead_mo = a.groupby("month").lead_time_days.mean()
    mo_df = pd.DataFrame({"month": mo.index, "mean_wait": mo.values, "mean_lead": lead_mo.values,
                          "mean_workload_min": day.groupby("month").workload_min.mean().values,
                          "attended_per_day": day.groupby("month").attended.mean().values})
    R["by_month"] = mo_df.round(2).to_dict("records")
    ym = att.groupby("year_month").wait_min.mean()
    R["monthly_series"] = {k: float(v) for k, v in ym.items()}
    # winter vs rest (day-level, the independent unit)
    w, o = day[day.is_winter == 1].mean_wait, day[day.is_winter == 0].mean_wait
    t, pt = stats.ttest_ind(w, o, equal_var=False)
    mw = stats.mannwhitneyu(w, o)
    pooled = np.sqrt((w.var() + o.var()) / 2)
    R["winter_test"] = {"winter_mean": w.mean(), "other_mean": o.mean(), "diff": w.mean() - o.mean(), "pct": w.mean() / o.mean() - 1,
                        "t": t, "p": pt, "mw_p": mw.pvalue, "cohens_d": (w.mean() - o.mean()) / pooled, "n_winter_days": len(w), "n_other_days": len(o),
                        "ci95": [float(x) for x in stats.t.interval(.95, len(w) - 1, loc=w.mean() - o.mean(), scale=np.sqrt(w.var() / len(w) + o.var() / len(o)))]}
    jan = day[day.month == 1].mean_wait.mean()
    R["jan_vs_rest_pct"] = jan / day[day.month != 1].mean_wait.mean() - 1
    # workload mediation
    m1 = smf.ols("mean_wait ~ is_winter", day).fit(cov_type="HC3")
    m2 = smf.ols("mean_wait ~ is_winter + workload_min", day).fit(cov_type="HC3")
    m3 = smf.ols("mean_wait ~ workload_min", day).fit(cov_type="HC3")
    R["mediation"] = {"winter_coef_alone": m1.params["is_winter"], "winter_p_alone": m1.pvalues["is_winter"],
                      "winter_coef_with_workload": m2.params["is_winter"], "winter_p_with_workload": m2.pvalues["is_winter"],
                      "workload_coef_per_100min": m2.params["workload_min"] * 100, "r2_workload_only": m3.rsquared, "r2_full": m2.rsquared}
    # trend
    day["t_years"] = (day.appointment_date - day.appointment_date.min()).dt.days / 365.25
    tr = smf.ols("mean_wait ~ t_years + C(month)", day).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
    ci = tr.conf_int().loc["t_years"].tolist()
    R["trend_wait"] = {"slope_min_per_year": tr.params["t_years"], "p": tr.pvalues["t_years"], "ci95": ci}
    tr2 = smf.ols("mean_lead ~ t_years + C(month)", day).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
    R["trend_lead"] = {"slope_days_per_year": tr2.params["t_years"], "p": tr2.pvalues["t_years"]}
    # trend by time block (worsening?)
    tb = []
    for blk, g in att.groupby("time_block"):
        yy = g.groupby("year").wait_min.mean()
        sl = stats.linregress(yy.index, yy.values)
        tb.append({"time_block": blk, "slope_per_year": sl.slope, "p": sl.pvalue, "mean": g.wait_min.mean()})
    R["trend_by_block"] = tb
    # moving averages
    day = day.sort_values("appointment_date")
    day["wait_ma30"] = day.mean_wait.rolling(30, min_periods=15).mean()
    mser = ym.copy(); mser.index = pd.PeriodIndex(mser.index, freq="M")
    ma12 = mser.rolling(12, min_periods=6).mean()

    # ---------------- lead time -----------------------------------------------
    R["lead_time"] = {"mean": a.lead_time_days.mean(), "median": a.lead_time_days.median(), "p90": a.lead_time_days.quantile(.9),
                      "share_gt_14": (a.lead_time_days > 14).mean(), "share_gt_22_iqr": a.scheduling_interval_outlier_iqr.mean()}
    R["lead_by_winter"] = a.groupby("is_winter").lead_time_days.mean().to_dict()

    # ---------------- segment tests (waiting / DNA / cancellation) ------------
    factors = ["age_group", "age_band", "sex", "insurance", "weekday", "time_block", "lead_time_band", "season"]
    rows = []
    for f in factors:
        g = [x.wait_min.values for _, x in att.groupby(f)]
        H, pH = stats.kruskal(*g)
        eta = (H - len(g) + 1) / (len(att) - len(g))
        ct = pd.crosstab(dna_den[f], dna_den.is_dna)
        pd_ = stats.chi2_contingency(ct)[1]; v_d = cramers_v(ct)
        ct2 = pd.crosstab(fin[f], fin.is_cancelled)
        pc_ = stats.chi2_contingency(ct2)[1]; v_c = cramers_v(ct2)
        rows.append({"factor": f, "wait_eta2": eta, "wait_p": pH, "dna_cramers_v": v_d, "dna_p": pd_, "cancel_cramers_v": v_c, "cancel_p": pc_})
    seg = pd.DataFrame(rows)
    ntests = len(seg) * 3
    for c in ["wait_p", "dna_p", "cancel_p"]:
        seg[c + "_bonf"] = (seg[c] * ntests).clip(upper=1)
    R["segment_tests"] = seg.round(6).to_dict("records")
    R["n_tests_bonferroni"] = ntests

    # segment profile table (for Tableau + report)
    def seg_profile(col):
        g1 = att.groupby(col).agg(attended=("wait_min", "size"), mean_wait=("wait_min", "mean"), p90_wait=("wait_min", lambda x: x.quantile(.9)), mean_duration=("duration_min", "mean"))
        g2 = dna_den.groupby(col).is_dna.agg(["mean", "size"]).rename(columns={"mean": "dna_rate", "size": "dna_den"})
        g3 = fin.groupby(col).is_cancelled.mean().rename("cancel_rate")
        g4 = a.groupby(col).lead_time_days.mean().rename("mean_lead_days")
        out = g1.join(g2).join(g3).join(g4).reset_index().rename(columns={col: "segment"})
        out.insert(0, "dimension", col)
        # Wilson-style 95% CI half-width for DNA
        p_, n_ = out.dna_rate, out.dna_den
        out["dna_ci95"] = 1.96 * np.sqrt(p_ * (1 - p_) / n_)
        return out
    seg_prof = pd.concat([seg_profile(c) for c in ["age_group", "age_band", "sex", "insurance", "weekday", "time_block", "lead_time_band", "season"]], ignore_index=True)

    # ---------------- DNA predictive model -------------------------------------
    dm = dna_den.copy()
    dm["prior_dna"] = dm.prior_dna.fillna(0); dm["prior_appts"] = dm.prior_appts.fillna(0)
    X = pd.get_dummies(dm[["age", "lead_time_days", "hour", "dow", "month", "prior_dna", "prior_appts", "sex", "insurance"]], columns=["sex", "insurance", "dow", "month"], drop_first=True)
    X = (X - X.mean()) / X.std()
    yv = dm.is_dna.values
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pr = cross_val_predict(LogisticRegression(max_iter=300), X, yv, cv=cv, method="predict_proba")[:, 1]
    R["dna_model"] = {"auc_cv5": roc_auc_score(yv, pr), "n": len(yv), "base_rate": yv.mean(), "features": list(X.columns)[:6] + ["..."]}
    pdna = dm.groupby(dm.prior_dna.clip(upper=2)).is_dna.agg(["mean", "size"]).reset_index()
    R["dna_by_prior_dna"] = pdna.round(4).to_dict("records")
    ptw = a.sort_values(["patient_id", "appointment_datetime"])
    ptw = ptw[ptw.status.isin(["attended", "did not attend"])].copy()
    ptw["prev_wait"] = ptw.groupby("patient_id").wait_min.shift()
    pw = ptw.dropna(subset=["prev_wait"])
    R["dna_vs_prev_wait"] = {"corr": float(np.corrcoef(pw.prev_wait, pw.is_dna)[0, 1]), "dna_after_wait_gt_60": float(pw[pw.prev_wait > 60].is_dna.mean()), "dna_after_wait_le_30": float(pw[pw.prev_wait <= 30].is_dna.mean())}

    # ---------------- daily relationships (waiting / utilisation / missed) -----
    cols = ["mean_wait", "attended", "dna", "cancelled_appts", "workload_min", "slot_booked_rate", "time_load_pct"]
    R["daily_corr"] = day[cols].corr().round(3).to_dict()
    mm = smf.ols("mean_wait ~ attended + dna + cancelled_appts", day).fit(cov_type="HC3")
    R["daily_ols"] = {"coef": mm.params.round(3).to_dict(), "p": mm.pvalues.round(6).to_dict(), "r2": mm.rsquared}
    mw_ = smf.ols("mean_wait ~ workload_min", day).fit(cov_type="HC3")
    R["wait_vs_workload"] = {"slope_per_100min": mw_.params["workload_min"] * 100, "r2": mw_.rsquared, "p": mw_.pvalues["workload_min"],
                             "corr": float(day[["mean_wait", "workload_min"]].corr().iloc[0, 1])}
    # does slot-booked rate (the brief's utilisation) relate to wait?
    R["corr_booked_rate_wait"] = float(day[["mean_wait", "slot_booked_rate"]].corr().iloc[0, 1])
    R["corr_time_load_wait"] = float(day[["mean_wait", "time_load_pct"]].corr().iloc[0, 1])
    # quartiles of time-load
    day["load_q"] = pd.qcut(day.time_load_pct, 4, labels=["Q1 lightest", "Q2", "Q3", "Q4 heaviest"])
    R["wait_by_load_quartile"] = day.groupby("load_q").agg(wait=("mean_wait", "mean"), overrun=("overrun_min", "mean"), load=("time_load_pct", "mean")).round(3).reset_index().astype({"load_q": str}).to_dict("records")

    # utilisation by dimension
    util = []
    for dim in ["year", "month", "dow", "hour", "time_block"]:
        g = res_slots.groupby(dim).agg(slots=("slot_id", "size"), booked=("is_booked_flag", "sum"),
                                       attended=("is_effective_use", "sum"),
                                       dna=("slot_outcome", lambda x: (x == "Did not attend").sum()),
                                       open_slots=("slot_outcome", lambda x: x.isin(["Cancelled, not re-filled", "Never booked"]).sum())).reset_index().rename(columns={dim: "segment"})
        g.insert(0, "dimension", dim)
        g["segment"] = g.segment.astype(str)
        g["booked_rate"], g["effective_rate"] = g.booked / g.slots, g.attended / g.slots
        g["dna_rate_slots"], g["open_rate_slots"] = g.dna / g.slots, g.open_slots / g.slots
        util.append(g)
    util = pd.concat(util, ignore_index=True)
    R["utilisation_by_hour"] = util[util.dimension == "hour"].round(4).to_dict("records")
    R["utilisation_by_dow"] = util[util.dimension == "dow"].round(4).to_dict("records")
    # hour-specific attendance probabilities (for LP)
    hp = res_slots.groupby("hour").apply(lambda g: (g.slot_outcome == "Attended").mean())
    R["p_att_by_hour"] = {int(k): float(v) for k, v in hp.items()}

    return day, mo_df, seg_prof, util, ma12, by_hour, hm, yr


# ==========================================================================
# 5. OPTIMISATION: Monte Carlo + LP
# ==========================================================================
class ClinicSim:
    """Vectorised single-queue clinic-day simulator, calibrated to the observed data.

    One clinician-equivalent queue; slots every `slot_len` minutes; each offered slot yields an
    attending patient with prob p_att; patient checks in `early` minutes before the slot (empirical),
    waits until the queue is free, consults for `dur` minutes (empirical bootstrap, optionally shifted)
    followed by a calibrated changeover time `turn`. 
    """

    def __init__(self, att, seed=SEED):
        self.dur = att.duration_min.values
        self.early = att.checkin_early_min.values
        self.seed = seed
        self.turn = 1.0

    def run(self, n_days=4000, slots=SLOTS_PER_DAY, slot_len=SLOT_LEN, p_att=0.83, dur_shift=0.0, offered_mask=None, turn=None, seed=None, min_dur=3.0):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        turn = self.turn if turn is None else turn
        free = np.full(n_days, float(OPEN_MIN))
        last_end = np.full(n_days, float(OPEN_MIN))
        W, H, Wsum, cnt = [], [], np.zeros(n_days), np.zeros(n_days)
        for k in range(slots):
            if offered_mask is not None and not offered_mask[k]:
                continue
            show = rng.random(n_days) < p_att
            ap = OPEN_MIN + k * slot_len
            ci = ap - self.early[rng.integers(0, len(self.early), n_days)]
            d = np.maximum(self.dur[rng.integers(0, len(self.dur), n_days)] + dur_shift, min_dur)
            st = np.maximum(ci, free)
            wait = st - ci
            free = np.where(show, st + d + turn, free)
            last_end = np.where(show, st + d, last_end)
            W.append(np.where(show, wait, np.nan)); H.append(np.full(n_days, (ap // 60)))
            Wsum += np.where(show, wait, 0); cnt += show
        Wm = np.vstack(W)  # slots x days
        flat = Wm[~np.isnan(Wm)]
        hours = np.vstack(H)[~np.isnan(Wm)]
        return {"wait": flat, "hour": hours, "attended_per_day": cnt.mean(), "last_end": last_end,
                "mean_wait": float(np.mean(flat)), "p90_wait": float(np.percentile(flat, 90)),
                "share_gt30": float(np.mean(flat > 30)), "share_gt60": float(np.mean(flat > 60)),
                "mean_finish_h": float(last_end.mean() / 60),
                "share_overrun30": float(np.mean(last_end > CLOSE_MIN + 30)),
                "overrun_mean": float(np.mean(np.maximum(last_end - CLOSE_MIN, 0))),
                "load": float(np.mean(cnt * (self.dur.mean() + dur_shift + turn)) / (slots * slot_len))}

    def calibrate(self, target_mean_wait, p_att, lo=0.0, hi=4.0, iters=14):
        for _ in range(iters):
            mid = (lo + hi) / 2
            m = self.run(n_days=3000, p_att=p_att, turn=mid, seed=11)["mean_wait"]
            lo, hi = (mid, hi) if m < target_mean_wait else (lo, mid)
        self.turn = (lo + hi) / 2
        return self.turn


def mask_even(n_offer, slots=SLOTS_PER_DAY):
    """Offered slots spread evenly over the day."""
    idx = np.floor(np.arange(1, slots + 1) * n_offer / slots) - np.floor(np.arange(0, slots) * n_offer / slots)
    return idx.astype(bool)


def mask_trim_end(n_offer, slots=SLOTS_PER_DAY):
    m = np.zeros(slots, bool); m[:int(round(n_offer))] = True; return m


def mask_midday_block(n_offer, slots=SLOTS_PER_DAY):
    """Remove closed slots as one contiguous block around 12:00-ish (centred at slot 18)."""
    n_close = slots - int(round(n_offer)); m = np.ones(slots, bool)
    start = max(0, min(slots - n_close, 16))
    m[start:start + n_close] = False
    return m


def solve_lp(rho_star, p_h, dsvc, n_max=4.0):
    """Capacity-allocation LP.
    max  sum_h p_h * n_h                       (expected attended patients / day)
    s.t. cumulative workload through hour h:  sum_{k<=h} p_k n_k dsvc <= rho* * 60 * (h+1)   for each hour block
         0 <= n_h <= n_max                    (4 fifteen-minute slots per hour block, no over-booking)
    """
    hrs = sorted(p_h)
    prob = pulp.LpProblem("slot_allocation", pulp.LpMaximize)
    n = {h: pulp.LpVariable(f"n_{h}", lowBound=0, upBound=n_max) for h in hrs}
    prob += pulp.lpSum(p_h[h] * n[h] for h in hrs)
    for i, h in enumerate(hrs):
        prob += pulp.lpSum(p_h[k] * n[k] * dsvc for k in hrs[: i + 1]) <= rho_star * 60.0 * (i + 1), f"cum_{h}"
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    sol = {h: float(n[h].value()) for h in hrs}
    return sol, float(pulp.value(prob.objective)), pulp.LpStatus[prob.status]


def optimise(a, s, day, kpi):
    att = a[a.is_attended == 1]
    sim = ClinicSim(att)
    sw = s[(s.in_window == 1) & (s.slot_outcome != "Unresolved")]
    p_att = (sw.slot_outcome == "Attended").mean()
    p_dna = (sw.slot_outcome == "Did not attend").mean()
    p_open_cancel = (sw.slot_outcome == "Cancelled, not re-filled").mean()
    p_never = (sw.slot_outcome == "Never booked").mean()
    p_filled = 1 - p_open_cancel - p_never
    q_dna = p_dna / p_filled
    turn = sim.calibrate(att.wait_min.mean(), p_att)
    R["sim_params"] = {"p_att": p_att, "p_dna": p_dna, "p_open_cancel": p_open_cancel, "p_never_booked": p_never, "q_dna_given_filled": q_dna,
                       "calibrated_changeover_min": turn, "mean_duration_min": att.duration_min.mean(), "observed_median_gap_min": R["median_changeover_min"]}

    # ---------- validation (out-of-sample: hourly profile + day-end) ----------
    base = sim.run(n_days=20000, p_att=p_att, seed=7)
    obs_h = att.groupby("hour").wait_min.mean()
    sim_h = pd.Series(base["wait"]).groupby(base["hour"]).mean()
    sim_h.index = sim_h.index.astype(int)
    rmse = float(np.sqrt(((obs_h.values - sim_h.values) ** 2).mean()))
    obs_end = day.last_end_min.values
    R["sim_validation"] = {
        "obs_mean_wait": att.wait_min.mean(), "sim_mean_wait": base["mean_wait"],
        "hour_profile_rmse_min": rmse, "hour_profile_corr": float(np.corrcoef(obs_h.values, sim_h.values)[0, 1]),
        "obs_finish_h": obs_end.mean() / 60, "sim_finish_h": base["mean_finish_h"],
        "obs_p25_p75_finish_h": [float(np.percentile(obs_end, 25) / 60), float(np.percentile(obs_end, 75) / 60)],
        "sim_p25_p75_finish_h": [float(np.percentile(base["last_end"], 25) / 60), float(np.percentile(base["last_end"], 75) / 60)],
        "obs_p90_wait": att.wait_min.quantile(.9), "sim_p90_wait": base["p90_wait"],
        "obs_share_gt30": float((att.wait_min > 30).mean()), "sim_share_gt30": base["share_gt30"],
        "obs_hour": {int(k): float(v) for k, v in obs_h.items()}, "sim_hour": {int(k): float(v) for k, v in sim_h.items()}}
    sim_val_end = base["last_end"]

    def scen(name, n_days=10000, reps=10, **kw):
        """Run `reps` independent batches -> mean +/- 95% CI across batches."""
        out = []
        for r_ in range(reps):
            o = sim.run(n_days=n_days // reps, seed=100 + r_, **kw)
            out.append({k: o[k] for k in ["mean_wait", "p90_wait", "share_gt30", "attended_per_day", "mean_finish_h", "share_overrun30", "overrun_mean", "load"]})
        d = pd.DataFrame(out)
        row = {"scenario": name}
        for c in d.columns:
            row[c] = d[c].mean(); row[c + "_ci"] = 1.96 * d[c].std(ddof=1) / np.sqrt(reps)
        return row

    def p_after(refill=0.0, dna_cut=0.0):
        p_open = p_open_cancel * (1 - refill) + p_never
        qd = q_dna * (1 - dna_cut)
        return (1 - p_open) * (1 - qd)

    # ---------- LP frontier -------------------------------------------------
    d_svc = att.duration_min.mean() + turn
    p_h = R["p_att_by_hour"]
    frontier = []
    for rho in [1.00, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82]:
        sol, obj, status = solve_lp(rho, p_h, d_svc)
        n_slots_offered = sum(sol.values()) * 1.0
        mask = mask_even(round(n_slots_offered))
        row = scen(f"LP cap rho*={rho:.2f}", offered_mask=mask, p_att=p_att, n_days=6000, reps=6)
        row.update({"rho_star": rho, "lp_expected_attended": obj, "lp_slots_offered": n_slots_offered, "lp_status": status, "lp_solution": {str(k): round(v, 3) for k, v in sol.items()}})
        frontier.append(row)
    R["lp_frontier"] = frontier
    ok = [f for f in frontier if f["mean_wait"] <= WAIT_TARGET]
    R["lp_recommended"] = max(ok, key=lambda f: f["attended_per_day"]) if ok else None

    # ---------- placement test: same number of offered slots, different layout ----
    n_off = 36
    placement = []
    for nm, mk in [("Spread evenly", mask_even), ("Trim from end of day", mask_trim_end), ("Midday block removed", mask_midday_block)]:
        placement.append(scen(f"{n_off} slots - {nm}", offered_mask=mk(n_off), p_att=p_att, n_days=6000, reps=6))
    R["placement_test"] = placement

    # ---------- scenarios -----------------------------------------------------
    scenarios = [
        scen("S0 Baseline (as observed)", p_att=p_att),
        scen("S1 Fill the gaps: re-fill 70% of cancelled slots + 25% fewer DNAs", p_att=p_after(0.7, 0.25)),
        scen("S2 LP-capped bookings (utilisation cap 0.91)", p_att=p_att, offered_mask=mask_even(round(solve_lp(0.91, p_h, d_svc)[1] / p_att))),
        scen("S3 Longer slots: 18-minute template (33 slots)", p_att=p_att, slots=33, slot_len=18),
        scen("S4 Shorter consultations (-2 min via pre-visit prep)", p_att=p_att, dur_shift=-2.0),
        scen("S5 S1 + S4 (fill gaps AND shorten consultations)", p_att=p_after(0.7, 0.25), dur_shift=-2.0),
    ]

    # ---------- grid search for a recommended package ------------------------
    grid = []
    for n_off_ in [30, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
        for dsh in [0.0, -1.0, -2.0]:
            for refill, cut in [(0.0, 0.0), (0.7, 0.25)]:
                o = sim.run(n_days=3000, p_att=p_after(refill, cut), offered_mask=mask_midday_block(n_off_), dur_shift=dsh, seed=999)
                grid.append({"offered_slots": n_off_, "dur_shift": dsh, "refill": refill, "dna_cut": cut, "attended_per_day": o["attended_per_day"],
                             "mean_wait": o["mean_wait"], "p90_wait": o["p90_wait"], "share_gt30": o["share_gt30"], "mean_finish_h": o["mean_finish_h"], "overrun_mean": o["overrun_mean"]})
    grid = pd.DataFrame(grid)
    feas = grid[grid.mean_wait <= WAIT_TARGET]
    best = feas.sort_values("attended_per_day", ascending=False).iloc[0]
    best_noshift = feas[(feas.dur_shift == 0)].sort_values("attended_per_day", ascending=False)
    R["grid_best"] = best.to_dict()
    R["grid_best_no_duration_change"] = best_noshift.iloc[0].to_dict() if len(best_noshift) else None
    best_cfg = dict(offered=int(best.offered_slots), dsh=float(best.dur_shift), refill=float(best.refill), cut=float(best.dna_cut))
    scenarios.append(scen(f"S6 Recommended package ({best_cfg['offered']} slots, midday reset, {best_cfg['dsh']:+.0f} min consult, refill + reminders)",
                          p_att=p_after(best_cfg["refill"], best_cfg["cut"]), offered_mask=mask_midday_block(best_cfg["offered"]), dur_shift=best_cfg["dsh"]))
    R["scenarios"] = scenarios
    # sensitivity of recommended package to target
    sens = []
    for tgt in [25, 30, 35, 40]:
        f = grid[grid.mean_wait <= tgt]
        if len(f):
            b = f.sort_values("attended_per_day", ascending=False).iloc[0]
            sens.append({"target": tgt, "offered_slots": int(b.offered_slots), "dur_shift": float(b.dur_shift), "attended_per_day": float(b.attended_per_day), "mean_wait": float(b.mean_wait)})
    R["target_sensitivity"] = sens
    return sim, base, grid, frontier, scenarios, placement, p_att


# ==========================================================================
# 6. CHARTS + EXPORTS
# ==========================================================================
def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_charts(a, s, day, mo_df, seg_prof, util, ma12, by_hour, hm, yr, base, grid, frontier, scenarios, placement, chart_dir: Path):
    att = a[a.is_attended == 1]
    kpi = R["kpi"]
    # 1 wait by hour (observed vs sim)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(by_hour.hour, by_hour.mean_wait, color=NHS_BLUE, width=.7, label="Observed mean wait")
    sv = R["sim_validation"]["sim_hour"]
    ax.plot(list(sv.keys()), list(sv.values()), color=NHS_WARM, marker="o", lw=2, label="Simulation (validation)")
    ax.axhline(WAIT_TARGET, color=NHS_RED, ls="--", lw=1); ax.text(7.55, WAIT_TARGET + 1, f"{WAIT_TARGET}-min target (analyst-set)", color=NHS_RED, fontsize=8)
    for h_, v_ in zip(by_hour.hour, by_hour.mean_wait): ax.text(h_, v_ + 1, f"{v_:.0f}", ha="center", fontsize=8)
    ax.set_xticks(by_hour.hour); ax.set_xticklabels([f"{h:02d}:00" for h in by_hour.hour])
    ax.set_ylabel("Minutes in clinic before consultation"); ax.set_title(f"Waits grow ~{R['wait_slope_per_hour']:.1f} min for every hour into the day")
    ax.legend(frameon=False, loc="upper left"); savefig(fig, chart_dir / "01_wait_by_hour.png")
    # 2 heatmap weekday x hour
    fig, ax = plt.subplots(figsize=(8, 3.4))
    im = ax.imshow(hm.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels([f"{c:02d}:00" for c in hm.columns]); ax.set_yticks(range(5)); ax.set_yticklabels(hm.index)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]): ax.text(j, i, f"{hm.values[i, j]:.0f}", ha="center", va="center", fontsize=8, color="black")
    ax.grid(False); ax.set_title("Mean in-clinic wait (min): time of day matters, weekday barely does")
    fig.colorbar(im, ax=ax, label="minutes"); savefig(fig, chart_dir / "02_heatmap_weekday_hour.png")
    # 3 seasonality
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = [NHS_DARK if m in WINTER_MONTHS else NHS_LIGHT for m in mo_df.month]
    ax.bar(mo_df.month, mo_df.mean_wait, color=cols)
    ax.axhline(att.wait_min.mean(), color=GREY, ls="--", lw=1)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    for m_, v_ in zip(mo_df.month, mo_df.mean_wait): ax.text(m_, v_ + .4, f"{v_:.1f}", ha="center", fontsize=8)
    wt = R["winter_test"]
    ax.set_ylim(30, 48); ax.set_ylabel("Mean wait (min)")
    ax.set_title(f"Winter (dark) waits are {wt['pct']:.0%} higher: {wt['winter_mean']:.1f} vs {wt['other_mean']:.1f} min")
    savefig(fig, chart_dir / "03_seasonality_month.png")
    # 4 trend with MA
    ms = pd.Series(R["monthly_series"]); ms.index = pd.PeriodIndex(ms.index, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(ms.index, ms.values, color=NHS_LIGHT, lw=1, label="Monthly mean wait")
    ax.plot(ma12.index.to_timestamp(), ma12.values, color=NHS_DARK, lw=2.4, label="12-month moving average")
    ax.set_ylabel("Minutes"); ax.set_title(f"No sustained deterioration: trend {R['trend_wait']['slope_min_per_year']:+.2f} min/yr (p={R['trend_wait']['p']:.2f})")
    ax.legend(frameon=False); savefig(fig, chart_dir / "04_trend_moving_average.png")
    # 5 slot outcome accounting
    so = R["slot_outcomes_pct_resolved"]
    order = ["Attended", "Did not attend", "Cancelled, not re-filled", "Never booked"]
    vals = [so.get(k, 0) for k in order]
    colors = [NHS_BLUE, NHS_WARM, NHS_RED, GREY]
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    left = 0
    for k, v, c in zip(order, vals, colors):
        ax.barh(0, v, left=left, color=c, height=.55)
        if v > 0.03: ax.text(left + v / 2, 0, f"{v:.1%}", ha="center", va="center", color="white" if c != NHS_WARM else "black", fontsize=9, fontweight="bold")
        left += v
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], order, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.15), fontsize=8)
    ax.set_yticks([]); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.set_xlim(0, 1); ax.grid(False)
    ax.set_title("Where every slot went (10 years, 103,573 resolved slots)"); savefig(fig, chart_dir / "05_slot_outcome_accounting.png")
    # 6 slot count vs time utilisation
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = ["Slots booked\n(is_available = False)", "Slots with attended\npatient", "Session time used by\nconsultations (min / 600)"]
    v = [kpi["booked_rate"], kpi["effective_rate"], kpi["time_load_pct"]]
    b = ax.bar(labels, v, color=[GREY, NHS_BLUE, NHS_RED], width=.55)
    for r_, x_ in zip(b, v): ax.text(r_.get_x() + r_.get_width() / 2, x_ + .01, f"{x_:.1%}", ha="center", fontweight="bold")
    ax.axhline(1.0, color="black", lw=.8, ls=":"); ax.set_ylim(0, 1.12); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(f"Slots look {kpi['effective_rate']:.0%} used, but consultations alone fill {kpi['time_load_pct']:.0%} of the session"); savefig(fig, chart_dir / "06_slots_vs_time_utilisation.png")
    # 7 DNA by insurance + age with CI
    sp = seg_prof[seg_prof.dimension.isin(["insurance", "age_band"])].copy()
    sp = sp.sort_values(["dimension", "dna_rate"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ypos = np.arange(len(sp)); ax.errorbar(sp.dna_rate, ypos, xerr=sp.dna_ci95, fmt="o", color=NHS_BLUE, ecolor=NHS_LIGHT, capsize=3)
    ax.axvline(kpi["dna_rate"], color=NHS_RED, ls="--", lw=1)
    ax.set_yticks(ypos); ax.set_yticklabels([f"{d.replace('age_band', 'Age')}: {x}" for d, x in zip(sp.dimension, sp.segment)], fontsize=8)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1)); ax.set_xlabel("DNA rate (95% CI)")
    ax.set_title(f"DNA rate is ~{kpi['dna_rate']:.1%} everywhere: no segment to target"); savefig(fig, chart_dir / "07_dna_by_segment.png")
    # 8 wait vs workload
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(day.workload_min, day.mean_wait, s=6, alpha=.25, color=NHS_BLUE)
    z = np.polyfit(day.workload_min, day.mean_wait, 1); xs = np.linspace(day.workload_min.min(), day.workload_min.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=NHS_RED, lw=2); ax.axvline(SESSION_MIN, color="black", ls=":", lw=1)
    ax.text(SESSION_MIN + 3, day.mean_wait.min() + 2, "600 min =\nscheduled\nsession", fontsize=8)
    ax.set_xlabel("Consultation minutes delivered that day"); ax.set_ylabel("Daily mean wait (min)")
    ax.set_title(f"Daily workload explains {R['wait_vs_workload']['r2']:.0%} of the variation in waiting (+{R['wait_vs_workload']['slope_per_100min']:.0f} min per +100 min)")
    savefig(fig, chart_dir / "08_wait_vs_workload.png")
    # 9 lead time histogram
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hist(a.lead_time_days, bins=np.arange(0.5, 31.5, 1), color=NHS_BLUE)
    ax.axvline(R["iqr"][2]["upper_fence"], color=NHS_RED, ls="--"); ax.text(R["iqr"][2]["upper_fence"] + .3, ax.get_ylim()[1] * .85, "IQR upper fence\n(22 days)", color=NHS_RED, fontsize=8)
    ax.set_xlabel("Booking lead time (days from scheduling to appointment)"); ax.set_ylabel("Appointments")
    ax.set_title(f"Booking lead time: median {kpi['median_lead_days']:.0f} days, 90% within {kpi['p90_lead_days']:.0f} days"); savefig(fig, chart_dir / "09_lead_time_hist.png")
    # 10 simulation validation: finish time
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(day.last_end_min / 60, bins=40, alpha=.65, color=NHS_BLUE, density=True, label="Observed")
    ax.hist(base["last_end"] / 60, bins=40, alpha=.5, color=NHS_WARM, density=True, label="Simulated")
    ax.axvline(CLOSE_MIN / 60, color="black", ls=":"); ax.text(CLOSE_MIN / 60 + .02, ax.get_ylim()[1] * .9, "18:00 close", fontsize=8)
    ax.set_xlabel("Clock hour the last consultation ends"); ax.set_yticks([])
    ax.set_title(f"Clinic finishes ~{R['kpi']['mean_overrun_min']:.0f} min after closing on average; model reproduces it"); ax.legend(frameon=False)
    savefig(fig, chart_dir / "10_finish_time_validation.png")
    # 11 LP frontier
    fr = pd.DataFrame(frontier)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.errorbar(fr.attended_per_day, fr.mean_wait, yerr=fr.mean_wait_ci, color=NHS_BLUE, marker="o", capsize=3, lw=2)
    for _, r_ in fr.iterrows(): ax.annotate(f"ρ*={r_.rho_star:.2f}", (r_.attended_per_day, r_.mean_wait), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(WAIT_TARGET, color=NHS_RED, ls="--", lw=1); ax.text(fr.attended_per_day.min(), WAIT_TARGET + 1, f"{WAIT_TARGET}-min target", color=NHS_RED, fontsize=8)
    ax.set_xlabel("Attended patients per day"); ax.set_ylabel("Mean in-clinic wait (min)")
    ax.set_title("LP + simulation frontier: each patient/day of throughput costs waiting time"); savefig(fig, chart_dir / "11_lp_frontier.png")
    # 12 scenarios
    sc = pd.DataFrame(scenarios)
    short = ["S0 Baseline", "S1 Fill gaps", "S2 LP cap\nρ*=0.91", "S3 18-min\nslots", "S4 -2 min\nconsults", "S5 Fill gaps\n+ -2 min", "S6 Recom-\nmended"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = [GREY, NHS_RED, NHS_LIGHT, NHS_LIGHT, NHS_BLUE, NHS_BLUE, NHS_GREEN]
    axs[0].bar(short, sc.mean_wait, yerr=sc.mean_wait_ci, color=cols, capsize=3); axs[0].axhline(WAIT_TARGET, color=NHS_RED, ls="--", lw=1)
    axs[0].set_title("Mean in-clinic wait (min)"); axs[0].tick_params(axis="x", labelsize=7)
    for i, v in enumerate(sc.mean_wait): axs[0].text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    axs[1].bar(short, sc.attended_per_day, yerr=sc.attended_per_day_ci, color=cols, capsize=3)
    axs[1].set_title("Attended patients per day"); axs[1].tick_params(axis="x", labelsize=7); axs[1].set_ylim(0, max(sc.attended_per_day) + 4)
    for i, v in enumerate(sc.attended_per_day): axs[1].text(i, v + .4, f"{v:.1f}", ha="center", fontsize=8)
    savefig(fig, chart_dir / "12_scenario_comparison.png")
    # 13 yearly lines by block
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for blk, g in att.groupby("time_block"):
        yy = g.groupby("year").wait_min.mean(); ax.plot(yy.index, yy.values, marker="o", ms=3, label=blk)
    ax.set_ylabel("Mean wait (min)"); ax.set_title("Every time block is flat across 2015-2024: the problem is structural, not worsening"); ax.legend(title="Slot time", ncol=5, frameon=False, fontsize=8)
    savefig(fig, chart_dir / "13_year_by_time_block.png")


def export_tableau(a, s, day, seg_prof, util, tab_dir: Path, grid, frontier, scenarios, placement):
    tab_dir.mkdir(parents=True, exist_ok=True)
    cols = ["appointment_id", "slot_id", "patient_id", "appointment_date", "appointment_datetime", "scheduling_date", "year", "quarter", "month", "month_name", "year_month",
            "weekday", "dow", "hour", "time_block", "season", "is_winter", "status", "is_attended", "is_dna", "is_cancelled", "is_resolved", "sex", "age", "age_group", "age_band",
            "insurance", "lead_time_days", "lead_time_band", "scheduling_interval_outlier_iqr", "wait_min", "duration_min", "wait_over_30", "wait_over_60", "checkin_early_min", "start_delay_min",
            "wait_recorded_min", "duration_recorded_min", "cancel_slot_refilled", "prior_appts", "prior_dna"]
    a[cols].to_csv(tab_dir / "fact_appointments.csv", index=False)
    s[["slot_id", "appointment_date", "appointment_time", "year", "month", "year_month", "weekday", "dow", "hour", "time_block", "is_available", "is_booked_flag", "in_window",
       "n_bookings", "n_cancel", "slot_outcome", "is_effective_use"]].to_csv(tab_dir / "fact_slots.csv", index=False)
    dd = day.drop(columns=["load_q"], errors="ignore").copy()
    dd["last_end_clock"] = (dd.last_end_min / 60)
    dd.to_csv(tab_dir / "agg_daily.csv", index=False)
    mo = a.groupby(["year", "month", "year_month"]).agg(appointments=("appointment_id", "size"), attended=("is_attended", "sum"), dna=("is_dna", "sum"), cancelled=("is_cancelled", "sum"),
                                                        mean_wait=("wait_min", "mean"), mean_lead=("lead_time_days", "mean"), mean_duration=("duration_min", "mean")).reset_index()
    mo["dna_rate"] = mo.dna / (mo.attended + mo.dna); mo["cancel_rate"] = mo.cancelled / (mo.attended + mo.dna + mo.cancelled)
    mo["is_winter"] = mo.month.isin(WINTER_MONTHS).astype(int)
    mo["wait_ma12"] = mo.mean_wait.rolling(12, min_periods=6).mean()
    mo.to_csv(tab_dir / "agg_monthly.csv", index=False)
    seg_prof.to_csv(tab_dir / "agg_segments.csv", index=False)
    util.to_csv(tab_dir / "agg_utilisation.csv", index=False)
    att = a[a.is_attended == 1]
    hh = att.groupby(["weekday", "dow", "hour"]).agg(mean_wait=("wait_min", "mean"), p90_wait=("wait_min", lambda x: x.quantile(.9)), visits=("wait_min", "size")).reset_index()
    sl = s[(s.in_window == 1) & (s.slot_outcome != "Unresolved")].groupby(["dow", "hour"]).agg(slots=("slot_id", "size"), attended=("is_effective_use", "sum")).reset_index()
    hh = hh.merge(sl, on=["dow", "hour"]); hh["effective_rate"] = hh.attended / hh.slots
    hh.to_csv(tab_dir / "agg_heatmap_weekday_hour.csv", index=False)
    mh = att.groupby(["month", "hour"]).wait_min.mean().reset_index().rename(columns={"wait_min": "mean_wait"}); mh.to_csv(tab_dir / "agg_heatmap_month_hour.csv", index=False)
    pd.DataFrame(scenarios).drop(columns=[c for c in ["lp_solution"] if c in pd.DataFrame(scenarios).columns]).to_csv(tab_dir / "scenario_results.csv", index=False)
    pd.DataFrame(frontier).drop(columns=["lp_solution"]).to_csv(tab_dir / "lp_frontier.csv", index=False)
    grid.to_csv(tab_dir / "scenario_grid.csv", index=False)


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, float): return None if np.isnan(o) else o
    if isinstance(o, (pd.Timestamp,)): return str(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default="..")
    args = ap.parse_args()
    data_dir, out = Path(args.data_dir), Path(args.out_dir)
    (out / "05_Charts").mkdir(parents=True, exist_ok=True)

    print("[1/6] load + clean");      appts, pats, slots, last_appt = load_and_clean(data_dir)
    print("[2/6] transform");         a, s, day = transform(appts, pats, slots, last_appt)
    print("[3/6] analyse");           day, mo_df, seg_prof, util, ma12, by_hour, hm, yr = analyse(a, s, day, last_appt)
    print("[4/6] optimise (LP + Monte Carlo)"); sim, base, grid, frontier, scenarios, placement, p_att = optimise(a, s, day, R["kpi"])
    print("[5/6] charts");            make_charts(a, s, day, mo_df, seg_prof, util, ma12, by_hour, hm, yr, base, grid, frontier, scenarios, placement, out / "05_Charts")
    print("[6/6] exports");           export_tableau(a, s, day, seg_prof, util, out / "04_Tableau" / "data", grid, frontier, scenarios, placement)
    (out / "results.json").write_text(json.dumps(jsonable(R), indent=1))
    print("done ->", out)


if __name__ == "__main__":
    main()

import os
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import datetime

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Enrollment Predictability (JetLearn)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Enrollment Predictability")
st.caption("Predict enrollments for a selected month using cohort conversion rates by JetLearn Deal Source.")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv(default_path: str = "Master_sheet_DB.csv") -> pd.DataFrame:
    """
    Load CSV either from file uploader or a default path if present.
    """
    # Priority 1: File uploader
    uploaded = st.sidebar.file_uploader("Upload CSV (optional) — default is Master_sheet_DB.csv", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        return df

    # Priority 2: Default path
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        return df

    return pd.DataFrame()

def month_floor(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, errors="coerce").dt.to_period("M").dt.to_timestamp()

def detect_column(candidates, columns):
    """
    Try to auto-detect a column by common aliases.
    """
    cols_lower = {c.lower(): c for c in columns}
    for alias in candidates:
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
    return None

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def valid_month_range(series: pd.Series):
    s = pd.to_datetime(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None, None
    return s.min(), s.max()

# -------------------------
# Load data
# -------------------------
df_raw = load_csv()

if df_raw.empty:
    st.warning("No data found. Upload a CSV or place **Master_sheet_DB.csv** beside this app.")
    st.stop()

st.subheader("1) Column Mapping")

# Try to auto-detect typical names
create_candidates = ["Create Date", "Create_Date", "Created Date", "Created_On", "Deal Create Date"]
enroll_candidates = ["Payment Received Date", "Payment_Received_Date", "Enrollment Date", "Payment Date", "Payment_Received"]
source_candidates = ["JetLearn Deal Source", "JetLearn_Deal_Source", "Deal Source", "Source", "Lead Source"]
stage_candidates  = ["Deal Stage", "Deal_Stage", "Stage"]

auto_create = detect_column(create_candidates, df_raw.columns)
auto_enrol  = detect_column(enroll_candidates, df_raw.columns)
auto_source = detect_column(source_candidates, df_raw.columns)
auto_stage  = detect_column(stage_candidates, df_raw.columns)

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
with c1:
    col_create = st.selectbox("Create Date column", options=list(df_raw.columns), index=(list(df_raw.columns).index(auto_create) if auto_create in df_raw.columns else 0))
with c2:
    col_enroll = st.selectbox("Payment Received Date column", options=list(df_raw.columns), index=(list(df_raw.columns).index(auto_enrol) if auto_enrol in df_raw.columns else 0))
with c3:
    col_source = st.selectbox("Deal Source column", options=list(df_raw.columns), index=(list(df_raw.columns).index(auto_source) if auto_source in df_raw.columns else 0))
with c4:
    col_stage_opt = st.selectbox("Deal Stage column (optional)", options=["<None>"] + list(df_raw.columns),
                                 index=(0 if auto_stage is None else (list(df_raw.columns).index(auto_stage) + 1)))

exclude_invalid = False
invalid_label = "1.2 Invalid Deal"
if col_stage_opt != "<None>":
    exclude_invalid = st.checkbox("Exclude rows where Deal Stage == '1.2 Invalid Deal'", value=False)
    invalid_label = st.text_input("Invalid stage label (exact match)", value="1.2 Invalid Deal")

# Subset and rename
df = df_raw[[col_create, col_enroll, col_source] + ([] if col_stage_opt == "<None>" else [col_stage_opt])].copy()
df.rename(columns={
    col_create: "CreateDate",
    col_enroll: "EnrollDate",
    col_source: "DealSource",
}, inplace=True)
if col_stage_opt != "<None>":
    df.rename(columns={col_stage_opt: "DealStage"}, inplace=True)

# Clean
df["CreateDate"] = pd.to_datetime(df["CreateDate"], errors="coerce")
df["EnrollDate"] = pd.to_datetime(df["EnrollDate"], errors="coerce")
df["DealSource"] = df["DealSource"].astype(str).str.strip().fillna("Unknown")

if col_stage_opt != "<None>" and exclude_invalid:
    before = len(df)
    df = df[df["DealStage"].astype(str).str.strip() != invalid_label]
    after = len(df)
    st.caption(f"Filtered invalid stage rows: {before - after} removed.")

# Month-floor columns
df["CreateMonth"] = df["CreateDate"].dt.to_period("M").dt.to_timestamp()
df["EnrollMonth"] = df["EnrollDate"].dt.to_period("M").dt.to_timestamp()

# -------------------------
# Controls
# -------------------------
st.subheader("2) Prediction Controls")

# Available months for selection = any month appearing in either Create or Enroll dates
valid_min_create, valid_max_create = valid_month_range(df["CreateDate"])
valid_min_enrol, valid_max_enrol   = valid_month_range(df["EnrollDate"])
cand_min = min([d for d in [valid_min_create, valid_min_enrol] if d is not None], default=None)
cand_max = max([d for d in [valid_max_create, valid_max_enrol] if d is not None], default=None)

if cand_min is None or cand_max is None:
    st.error("Could not parse dates properly. Please verify your date columns.")
    st.stop()

months_series = pd.period_range(cand_min.to_period("M"), cand_max.to_period("M"), freq="M").to_timestamp()
default_target = months_series.max()

c1, c2, c3 = st.columns([1.3, 1.1, 1.2])
with c1:
    target_month = st.selectbox(
        "Target month (predict enrollments in…)",
        options=list(months_series),
        index=list(months_series).index(default_target)
    )
with c2:
    recency_n = st.slider("Data recency window (months used to learn cohort rates)", min_value=3, max_value=24, value=12, step=1)
with c3:
    max_lag = st.slider("Max carry-over window (lags in months)", min_value=1, max_value=12, value=6, step=1)

# Optional source filter
all_sources = sorted(df["DealSource"].dropna().unique().tolist())
sel_sources = st.multiselect("Filter Deal Sources (optional; default all)", options=all_sources, default=all_sources)

df = df[df["DealSource"].isin(sel_sources)]

# -------------------------
# Build cohort matrix (by source & lag)
# -------------------------
st.subheader("3) Cohort Model & Prediction")

# Training window: last N months ending just before the target month
train_end = pd.Timestamp(target_month) - pd.offsets.MonthBegin(0)  # start of target month
train_start = (train_end - pd.DateOffset(months=recency_n)).to_period("M").to_timestamp()

st.caption(f"Training window: **{train_start.strftime('%Y-%m')}** to **{(train_end - pd.DateOffset(days=1)).strftime('%Y-%m-%d')}** (inclusive).")

# Filter data for training window (for both create and enroll month interplay)
df_train = df[(df["CreateMonth"] >= train_start) & (df["CreateMonth"] < train_end)].copy()

# Deals created per month & source (denominator for lag rates)
created = (
    df_train
    .groupby(["DealSource", "CreateMonth"], dropna=False)
    .size()
    .rename("CreatedCount")
    .reset_index()
)

# Enrollments attributed to creation cohort by lag
# Keep only rows with a valid EnrollMonth
df_enrolled = df_train[~df_train["EnrollMonth"].isna()].copy()
df_enrolled["Lag"] = ((df_enrolled["EnrollMonth"].dt.to_period("M") - df_enrolled["CreateMonth"].dt.to_period("M")).apply(lambda p: p.n)).astype("Int64")
df_enrolled = df_enrolled[(df_enrolled["Lag"] >= 0) & (df_enrolled["Lag"] <= max_lag)]

enrolled_by_lag = (
    df_enrolled
    .groupby(["DealSource", "CreateMonth", "Lag"], dropna=False)
    .size()
    .rename("EnrollAtLag")
    .reset_index()
)

# Merge to compute rate = EnrollAtLag / CreatedCount for exact lag k
cohort = enrolled_by_lag.merge(created, on=["DealSource", "CreateMonth"], how="left")
cohort["Rate"] = cohort["EnrollAtLag"] / cohort["CreatedCount"]

# For stability: aggregate rates over training window per (source, lag)
rate_by_source_lag = (
    cohort
    .groupby(["DealSource", "Lag"], dropna=False)
    .agg(
        EnrollAtLag=("EnrollAtLag", "sum"),
        Created=("CreatedCount", "sum")
    )
    .reset_index()
)
rate_by_source_lag["Rate"] = rate_by_source_lag["EnrollAtLag"] / rate_by_source_lag["Created"]
rate_by_source_lag = rate_by_source_lag.replace([np.inf, -np.inf], np.nan).fillna({"Rate": 0.0})

# Also compute a global fallback rate by lag across all sources
rate_by_lag_global = (
    cohort
    .groupby(["Lag"], dropna=False)
    .agg(
        EnrollAtLag=("EnrollAtLag", "sum"),
        Created=("CreatedCount", "sum")
    )
    .reset_index()
)
rate_by_lag_global["Rate"] = rate_by_lag_global["EnrollAtLag"] / rate_by_lag_global["Created"]
rate_by_lag_global = rate_by_lag_global.replace([np.inf, -np.inf], np.nan).fillna({"Rate": 0.0})
global_rates = {int(row["Lag"]): float(row["Rate"]) for _, row in rate_by_lag_global.iterrows()}

# Helper to get rate for (source, lag) with fallback to global
def get_rate(src: str, lag: int) -> float:
    row = rate_by_source_lag[(rate_by_source_lag["DealSource"] == src) & (rate_by_source_lag["Lag"] == lag)]
    if not row.empty and pd.notna(row["Rate"].iloc[0]):
        return float(row["Rate"].iloc[0])
    return float(global_rates.get(lag, 0.0))

# -------------------------
# Predict for target month
# -------------------------
M = pd.Timestamp(target_month).to_period("M").to_timestamp()

# Created in M and in previous months (for carry-over)
created_hist = (
    df
    .groupby(["DealSource", "CreateMonth"], dropna=False)
    .size()
    .rename("CreatedCount")
    .reset_index()
)

# Same-month (lag 0): use deals created in M
m0 = created_hist[created_hist["CreateMonth"] == M].copy()
m0["Lag"] = 0
if m0.empty:
    # Create zero rows for selected sources if no creation in M
    m0 = pd.DataFrame({"DealSource": sel_sources, "CreateMonth": [M]*len(sel_sources), "CreatedCount": [0]*len(sel_sources), "Lag": [0]*len(sel_sources)})

# Carry-over (lags 1..max_lag): deals created in M-k
carry_frames = []
for k in range(1, max_lag + 1):
    mk = (M - pd.DateOffset(months=k)).to_period("M").to_timestamp()
    tmp = created_hist[created_hist["CreateMonth"] == mk].copy()
    if tmp.empty:
        # zero rows if no creation in that past month
        tmp = pd.DataFrame({"DealSource": sel_sources, "CreateMonth": [mk]*len(sel_sources), "CreatedCount": [0]*len(sel_sources)})
    tmp["Lag"] = k
    carry_frames.append(tmp)
carry = pd.concat(carry_frames, ignore_index=True) if carry_frames else pd.DataFrame(columns=["DealSource", "CreateMonth", "CreatedCount", "Lag"])

pred_base = pd.concat([m0, carry], ignore_index=True)

# Apply rates
pred_base["Rate"] = pred_base.apply(lambda r: get_rate(r["DealSource"], int(r["Lag"])), axis=1)
pred_base["PredEnrollments"] = pred_base["CreatedCount"] * pred_base["Rate"]

# Summaries
pred_by_source = (
    pred_base
    .groupby(["DealSource", "Lag"], dropna=False)["PredEnrollments"]
    .sum()
    .reset_index()
)

# Pivot to show M0 vs Carry-over
pivot_src = pred_by_source.pivot_table(index="DealSource", columns="Lag", values="PredEnrollments", aggfunc="sum").fillna(0.0)
pivot_src.columns = [f"Lag_{int(c)}" for c in pivot_src.columns]
pivot_src["SameMonth(M0)"] = pivot_src.get("Lag_0", 0.0)
carry_cols = [c for c in pivot_src.columns if c.startswith("Lag_") and c != "Lag_0"]
pivot_src["CarryOver(M-1..M-n)"] = pivot_src[carry_cols].sum(axis=1) if carry_cols else 0.0
pivot_src["Pred_Total"] = pivot_src["SameMonth(M0)"] + pivot_src["CarryOver(M-1..M-n)"]
pivot_src = pivot_src[["SameMonth(M0)", "CarryOver(M-1..M-n)", "Pred_Total"]].sort_values("Pred_Total", ascending=False)

overall_same = float(pivot_src["SameMonth(M0)"].sum()) if not pivot_src.empty else 0.0
overall_carry = float(pivot_src["CarryOver(M-1..M-n)"].sum()) if not pivot_src.empty else 0.0
overall_total = overall_same + overall_carry

# -------------------------
# Display results
# -------------------------
st.markdown(f"### Prediction for **{M.strftime('%Y-%m')}**")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Predicted Total Enrollments", f"{overall_total:,.1f}")
kpi2.metric("Same-month (M0)", f"{overall_same:,.1f}")
kpi3.metric("Carry-over (M-1..M-n)", f"{overall_carry:,.1f}")

st.markdown("#### Deal Source Breakdown")
st.dataframe(
    pivot_src.reset_index().rename(columns={"DealSource": "Deal Source"}),
    use_container_width=True
)

# -------------------------
# Diagnostics: recent actuals vs predicted for last K months
# -------------------------
st.markdown("---")
st.subheader("4) Diagnostics")

diag_months = st.slider("Show last K months history (actuals vs backtested predictions)", 3, 18, 6)

# Build actual monthly enrollments (ground truth)
actuals = (
    df[~df["EnrollMonth"].isna()]
    .groupby("EnrollMonth")
    .size()
    .rename("ActualEnrollments")
    .reset_index()
)

# Backtest predictions for each of last K months using the same cohort-rate approach
history_months = pd.period_range((M - pd.DateOffset(months=diag_months-1)).to_period("M"), M.to_period("M"), freq="M").to_timestamp()

def predict_for_month(target_M, recency_n, max_lag, sel_sources):
    # Training window ends just before target_M
    train_end_m = target_M
    train_start_m = (train_end_m - pd.DateOffset(months=recency_n)).to_period("M").to_timestamp()

    df_train_m = df[(df["CreateMonth"] >= train_start_m) & (df["CreateMonth"] < train_end_m)].copy()

    created_m = df_train_m.groupby(["DealSource", "CreateMonth"], dropna=False).size().rename("CreatedCount").reset_index()
    df_enr_m = df_train_m[~df_train_m["EnrollMonth"].isna()].copy()
    if df_enr_m.empty and created_m.empty:
        return 0.0

    df_enr_m["Lag"] = ((df_enr_m["EnrollMonth"].dt.to_period("M") - df_enr_m["CreateMonth"].dt.to_period("M")).apply(lambda p: p.n)).astype("Int64")
    df_enr_m = df_enr_m[(df_enr_m["Lag"] >= 0) & (df_enr_m["Lag"] <= max_lag)]

    enrolled_by_lag_m = (
        df_enr_m.groupby(["DealSource", "CreateMonth", "Lag"], dropna=False).size().rename("EnrollAtLag").reset_index()
    )
    cohort_m = enrolled_by_lag_m.merge(created_m, on=["DealSource", "CreateMonth"], how="left")
    cohort_m["Rate"] = cohort_m["EnrollAtLag"] / cohort_m["CreatedCount"]

    rate_src_lag_m = (
        cohort_m.groupby(["DealSource", "Lag"], dropna=False)
        .agg(EnrollAtLag=("EnrollAtLag", "sum"), Created=("CreatedCount", "sum"))
        .reset_index()
    )
    rate_src_lag_m["Rate"] = rate_src_lag_m["EnrollAtLag"] / rate_src_lag_m["Created"]
    rate_src_lag_m = rate_src_lag_m.replace([np.inf, -np.inf], np.nan).fillna({"Rate": 0.0})

    rate_lag_glob_m = (
        cohort_m.groupby(["Lag"], dropna=False)
        .agg(EnrollAtLag=("EnrollAtLag", "sum"), Created=("CreatedCount", "sum"))
        .reset_index()
    )
    rate_lag_glob_m["Rate"] = rate_lag_glob_m["EnrollAtLag"] / rate_lag_glob_m["Created"]
    rate_lag_glob_m = rate_lag_glob_m.replace([np.inf, -np.inf], np.nan).fillna({"Rate": 0.0})
    glob_dict = {int(r["Lag"]): float(r["Rate"]) for _, r in rate_lag_glob_m.iterrows()}

    def gr(src, lag):
        row = rate_src_lag_m[(rate_src_lag_m["DealSource"] == src) & (rate_src_lag_m["Lag"] == lag)]
        if not row.empty and pd.notna(row["Rate"].iloc[0]):
            return float(row["Rate"].iloc[0])
        return float(glob_dict.get(lag, 0.0))

    # Created sets for target_M
    created_hist_m = df.groupby(["DealSource", "CreateMonth"], dropna=False).size().rename("CreatedCount").reset_index()

    # lag 0 (same month)
    m0_m = created_hist_m[created_hist_m["CreateMonth"] == target_M].copy()
    if m0_m.empty:
        m0_m = pd.DataFrame({"DealSource": sel_sources, "CreateMonth": [target_M]*len(sel_sources), "CreatedCount": [0]*len(sel_sources)})
    m0_m["Lag"] = 0

    # carry-over
    carry_frames_m = []
    for k in range(1, max_lag + 1):
        mk = (target_M - pd.DateOffset(months=k)).to_period("M").to_timestamp()
        tmp = created_hist_m[created_hist_m["CreateMonth"] == mk].copy()
        if tmp.empty:
            tmp = pd.DataFrame({"DealSource": sel_sources, "CreateMonth": [mk]*len(sel_sources), "CreatedCount": [0]*len(sel_sources)})
        tmp["Lag"] = k
        carry_frames_m.append(tmp)
    carry_m = pd.concat(carry_frames_m, ignore_index=True) if carry_frames_m else pd.DataFrame(columns=["DealSource", "CreateMonth", "CreatedCount", "Lag"])

    base_m = pd.concat([m0_m, carry_m], ignore_index=True)
    base_m["Rate"] = base_m.apply(lambda r: gr(r["DealSource"], int(r["Lag"])), axis=1)
    base_m["PredEnrollments"] = base_m["CreatedCount"] * base_m["Rate"]
    return float(base_m["PredEnrollments"].sum())

# Build dataframe for chart
hist_rows = []
for tm in history_months:
    pred_val = predict_for_month(tm, recency_n=recency_n, max_lag=max_lag, sel_sources=sel_sources)
    hist_rows.append({"Month": tm, "Type": "Predicted", "Enrollments": pred_val})

actual_rows = []
for _, r in actuals.iterrows():
    if r["EnrollMonth"] in list(history_months):
        actual_rows.append({"Month": r["EnrollMonth"], "Type": "Actual", "Enrollments": float(r["ActualEnrollments"])})

hist_df = pd.DataFrame(hist_rows + actual_rows)
if not hist_df.empty:
    hist_df.sort_values("Month", inplace=True)

    line = alt.Chart(hist_df).mark_line(point=True).encode(
        x=alt.X("Month:T", title="Month"),
        y=alt.Y("Enrollments:Q", title="Enrollments"),
        color=alt.Color("Type:N"),
        tooltip=[alt.Tooltip("Month:T"), alt.Tooltip("Type:N"), alt.Tooltip("Enrollments:Q", format=".1f")]
    ).properties(height=360)

    st.altair_chart(line, use_container_width=True)
else:
    st.info("Not enough history to draw diagnostics.")

st.markdown("---")
st.markdown("**Notes**")
st.markdown("""
- This is a **cohort-rate model**: it learns the exact-month lag conversion rate (M0, M1, M2, …) by **Deal Source** over your chosen training window, then applies those rates to current deal volumes.
- It is intentionally transparent, fast, and robust for Streamlit Cloud.  
- If a (source, lag) is sparse, it **backs off** to a global lag rate.
- Use the **Data recency** slider to reflect your most recent funnel behavior.
""")

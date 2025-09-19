
# JetLearn Sales Intelligence — FINAL (repo-ready)
# Main file path: app.py (place at repo root). Auto-loads Master_sheet_DB.csv.
# If scikit-learn is missing, the Predictability tab gracefully falls back
# to a baseline so the app never crashes.

import os, datetime as dt
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# Try to import scikit-learn; degrade gracefully if unavailable
SKLEARN_OK = True
try:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:
    SKLEARN_OK = False

# ------------------------------
# App config
# ------------------------------
st.set_page_config(
    page_title="JetLearn Sales Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
alt.data_transformers.disable_max_rows()
TZ = "Asia/Kolkata"

# ------------------------------
# Utilities
# ------------------------------
DATE_FMT_INFER = ["%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S",
                  "%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%Y %H:%M"]

def parse_dt(s):
    if pd.isna(s): return pd.NaT
    if isinstance(s, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.to_datetime(s)
    for fmt in DATE_FMT_INFER:
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            continue
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def to_month_start(ts):
    if pd.isna(ts): return pd.NaT
    return pd.Timestamp(ts).to_period("M").to_timestamp()

def month_bounds(ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    ts = pd.Timestamp(ts).tz_localize(TZ) if pd.Timestamp(ts).tz is None else pd.Timestamp(ts).tz_convert(TZ)
    start = ts.to_period("M").start_time
    end = ts.to_period("M").end_time
    return start, end

def last_month_of(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts).tz_localize(TZ) if pd.Timestamp(ts).tz is None else pd.Timestamp(ts).tz_convert(TZ)
    return (ts - pd.offsets.MonthBegin(1)).to_period("M").to_timestamp(tz=TZ)

def safe_count(series: pd.Series) -> int:
    return int(series.notna().sum())

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["create date", "created date", "deal create date"]:
            rename_map[c] = "Create Date"
        elif lc in ["payment received date", "payment receive date", "enrolment date", "enrollment date"]:
            rename_map[c] = "Payment Received Date"
        elif lc in ["jetlearn deal source", "jetline deal source", "deal source", "source"]:
            rename_map[c] = "JetLearn Deal Source"
        elif lc == "country":
            rename_map[c] = "Country"
        elif lc == "age":
            rename_map[c] = "Age"
        elif lc in ["academic counselor", "ac", "academy counselor", "academic counsellor",
                    "student/academic counsellor", "student/academic counselor"]:
            rename_map[c] = "Academic Counselor"
        elif lc in ["no. of calls connected", "number of times call connected", "calls connected"]:
            rename_map[c] = "Calls Connected"
        elif lc in ["no. of sales activities", "numbers of sales activity", "sales activities", "activities count"]:
            rename_map[c] = "Sales Activities"
        elif lc in ["last call connected", "last call connected date"]:
            rename_map[c] = "Last Call Connected"
        elif lc in ["last number of activities", "last activity count"]:
            rename_map[c] = "Last Activity Count"
        elif lc == "deal stage":
            rename_map[c] = "Deal Stage"
    return df.rename(columns=rename_map) if rename_map else df

def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Create Date" in df.columns:
        df["Create Date"] = df["Create Date"].apply(parse_dt)
        df["Create Month"] = df["Create Date"].apply(to_month_start)
        df["Create Day"] = df["Create Date"].dt.date
        df["Create Week"] = df["Create Date"].dt.to_period("W").apply(lambda p: p.start_time.date())
    if "Payment Received Date" in df.columns:
        df["Payment Received Date"] = df["Payment Received Date"].apply(parse_dt)
        df["Payment Month"] = df["Payment Received Date"].apply(to_month_start)
        df["Payment Day"] = df["Payment Received Date"].dt.date
        df["Payment Week"] = df["Payment Received Date"].dt.to_period("W").apply(lambda p: p.start_time.date())
    return df

def filter_df(df, countries, sources, counselors, min_age, max_age, date_min, date_max):
    out = df.copy()
    if countries: out = out[out["Country"].isin(countries)]
    if sources: out = out[out["JetLearn Deal Source"].isin(sources)]
    if counselors: out = out[out["Academic Counselor"].isin(counselors)]
    if "Age" in out.columns:
        out = out[(out["Age"].fillna(0) >= min_age) & (out["Age"].fillna(0) <= max_age)]
    if date_min: out = out[out["Create Date"] >= pd.Timestamp(date_min)]
    if date_max: out = out[out["Create Date"] <= pd.Timestamp(date_max) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    return out

def conversions_in_period(df, start_date, end_date):
    if "Payment Received Date" not in df.columns: return 0
    mask = df["Payment Received Date"].notna()
    if start_date is not None: mask &= (df["Payment Received Date"] >= start_date)
    if end_date is not None: mask &= (df["Payment Received Date"] <= end_date)
    return int(mask.sum())

# ------------------------------
# Sidebar (Drawer) — Filters & Data Upload
# ------------------------------
st.markdown("### 📱 JetLearn Sales Intelligence")
st.caption("Mobile-friendly UI with a left drawer for filters.")

with st.sidebar:
    st.header("⚙️ Filters & Data")
    # Auto-load Master_sheet_DB.csv from repo root if present
    df = None
    default_csv = Path("Master_sheet_DB.csv")
    if default_csv.exists():
        try:
            df = pd.read_csv(default_csv)
            st.success("Loaded: Master_sheet_DB.csv")
        except Exception as e:
            st.error(f"Failed loading Master_sheet_DB.csv: {e}")

    upl = st.file_uploader("Upload CSV (overrides default Master_sheet_DB.csv)", type=["csv"])
    if upl is not None:
        df = pd.read_csv(upl)

    if df is None:
        st.info("Place **Master_sheet_DB.csv** at repo root or upload a file. A tiny demo dataset is shown below.")
        rng = np.random.default_rng(42)
        n = 600
        today = pd.Timestamp.now(tz=TZ)
        create_dates = pd.date_range(today - pd.Timedelta(days=150), periods=n, freq="12H")
        pay_delays = rng.integers(-10, 45, size=n)
        payment_dates = [ (pd.NaT if d<0 or rng.random()<0.45 else create_dates[i] + pd.Timedelta(days=int(d)))
                          for i, d in enumerate(pay_delays) ]
        df = pd.DataFrame({
            "Create Date": create_dates[:n],
            "Payment Received Date": payment_dates[:n],
            "Country": rng.choice(["United Kingdom","Netherlands","India","UAE","Switzerland","Ireland"], size=n),
            "JetLearn Deal Source": rng.choice(["Organic","Referrals","Events","Lead Magnet","PM - Search","PM - Social","Others"], size=n),
            "Age": rng.integers(6, 16, size=n),
            "Academic Counselor": rng.choice(["Ali","Fuzail","Unmesh","Ankush","Kamal","Shahbaz","Rajan","Jai","Aniket","Ria"], size=n),
            "Calls Connected": np.maximum(0, rng.normal(3, 2, size=n).round()).astype(int),
            "Sales Activities": np.maximum(0, rng.normal(7, 4, size=n).round()).astype(int),
            "Last Call Connected": [pd.NaT if rng.random()<0.2 else d + pd.Timedelta(days=int(rng.integers(0,10))) for d in create_dates[:n]],
            "Last Activity Count": np.maximum(0, rng.normal(2, 1.5, size=n).round()).astype(int),
            "Deal Stage": rng.choice(["1.1 New","1.2 Invalid Deal","2.0 Calibration Booked","3.0 Proposal","4.0 Won"], size=n,
                                     p=[0.25,0.1,0.3,0.25,0.1])
        })

    # Normalize + time fields
    df = normalize_columns(df)
    df = add_time_columns(df)

    # Filters
    countries = sorted(df["Country"].dropna().unique()) if "Country" in df.columns else []
    sources = sorted(df["JetLearn Deal Source"].dropna().unique()) if "JetLearn Deal Source" in df.columns else []
    counselors = sorted(df["Academic Counselor"].dropna().unique()) if "Academic Counselor" in df.columns else []

    f_countries = st.multiselect("Country", countries)
    f_sources = st.multiselect("JetLearn Deal Source", sources)
    f_counselors = st.multiselect("Academic Counselor", counselors)

    min_age = int(df["Age"].min()) if "Age" in df.columns and df["Age"].notna().any() else 0
    max_age = int(df["Age"].max()) if "Age" in df.columns and df["Age"].notna().any() else 100
    age_min, age_max = st.slider("Age range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    cmin = df["Create Date"].min() if "Create Date" in df.columns else None
    cmax = df["Create Date"].max() if "Create Date" in df.columns else None
    date_min = st.date_input("Create Date from", value=cmin.date() if pd.notna(cmin) else dt.date.today())
    date_max = st.date_input("Create Date to", value=cmax.date() if pd.notna(cmax) else dt.date.today())

    apply_filters = st.button("Apply Filters")

df_f = filter_df(df, f_countries, f_sources, f_counselors, age_min, age_max, date_min, date_max) if apply_filters else df.copy()

st.caption("ⓘ Definitions — Conversion = count of non-null `Payment Received Date`. Enrollment = count of `Create Date`.")

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 MIS", "🧠 Data Analysis (MTD & Cohort)", "🔮 Predictability (M0 + M−N)", "🧑‍💼 AC Performance"])

# ------------------------------
# TAB 1: MIS
# ------------------------------
with tab1:
    st.subheader("📊 MIS Snapshot")
    now = pd.Timestamp.now(tz=TZ)
    yday = (now - pd.Timedelta(days=1)).normalize()
    start_this_m, end_this_m = month_bounds(now)
    last_m = last_month_of(now)
    start_last_m, end_last_m = month_bounds(last_m)

    yday_conv = conversions_in_period(df_f, yday, yday + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    days_elapsed = (now - start_this_m).days + 1
    conv_so_far = conversions_in_period(df_f, start_this_m, now)
    daily_rate = conv_so_far / max(1, days_elapsed)
    days_in_month = end_this_m.day
    today_forecast = int(round(daily_rate * days_elapsed))
    last_month_conv = conversions_in_period(df_f, start_last_m, end_last_m)
    this_month_forecast = int(round(daily_rate * days_in_month))

    if "JetLearn Deal Source" in df_f.columns:
        ref_this_m = df_f[(df_f["Payment Received Date"].between(start_this_m, end_this_m, inclusive="both")) &
                          (df_f["JetLearn Deal Source"] == "Referrals")]
        ref_deals_m = df_f[(df_f["Create Date"].between(start_this_m, end_this_m, inclusive="both")) &
                           (df_f["JetLearn Deal Source"] == "Referrals")]
        referral_conversions = safe_count(ref_this_m["Payment Received Date"]) if len(ref_this_m) else 0
        referral_deals = len(ref_deals_m) if len(ref_deals_m) else 0
    else:
        referral_conversions = 0
        referral_deals = 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Yesterday's Conversions", yday_conv)
    c2.metric("Today's Forecasted Conversions", today_forecast)
    c3.metric("Last Month Conversions", last_month_conv)
    c4.metric("This Month Forecast (Total)", this_month_forecast)
    c5.metric("Referrals — Deals (MTD)", referral_deals)
    c6.metric("Referrals — Conversions (MTD)", referral_conversions)

    st.divider()
    if "Payment Day" in df_f.columns:
        conv_daily = (df_f[(df_f["Payment Received Date"].between(start_this_m, end_this_m, inclusive="both"))]
                      .groupby("Payment Day").size().reset_index(name="Conversions"))
        if not conv_daily.empty:
            chart = alt.Chart(conv_daily).mark_bar().encode(
                x=alt.X("Payment Day:T", title="Day (This Month)"),
                y=alt.Y("Conversions:Q"),
                tooltip=["Payment Day:T", "Conversions:Q"]
            ).properties(height=240, title="Daily Conversions (This Month)")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No conversions recorded this month in the filtered data.")

# ------------------------------
# TAB 2: Data Analysis (MTD & Cohort)
# ------------------------------
with tab2:
    st.subheader("🧠 MTD vs Cohort Analysis")
    st.caption("• **MTD** = Conversions from deals **created** this month and also **converted** within this month. "
               "• **Cohort** = All conversions that **happened in a chosen month**, irrespective of when the deal was created.")

    granularity = st.multiselect(
        "Granularity (group by) — choose any",
        options=[c for c in ["Academic Counselor", "JetLearn Deal Source", "Country"] if c in df_f.columns],
        default=[c for c in ["Academic Counselor"] if c in df_f.columns]
    )
    time_view = st.selectbox("Time view", ["Day", "Week", "Month"], index=2)

    start_this_m, end_this_m = month_bounds(pd.Timestamp.now(tz=TZ))
    mtd_df = df_f[df_f["Create Date"].between(start_this_m, end_this_m, inclusive="both")].copy()
    mtd_df["Is MTD Conversion"] = mtd_df["Payment Received Date"].between(start_this_m, end_this_m, inclusive="both")

    if time_view == "Day":
        time_col_c = "Create Day"; time_col_p = "Payment Day"
    elif time_view == "Week":
        time_col_c = "Create Week"; time_col_p = "Payment Week"
    else:
        time_col_c = "Create Month"; time_col_p = "Payment Month"

    g_cols = [time_col_c] + granularity if granularity else [time_col_c]
    enroll = mtd_df.groupby(g_cols).size().reset_index(name="Enrollments (MTD created)")
    mtd_conv = mtd_df[mtd_df["Is MTD Conversion"]].groupby(g_cols).size().reset_index(name="MTD Conversions")

    mtd_summary = pd.merge(enroll, mtd_conv, on=g_cols, how="left").fillna(0)
    mtd_summary["MTD Conversions"] = mtd_summary["MTD Conversions"].astype(int)
    if "Enrollments (MTD created)" in mtd_summary:
        mtd_summary["MTD Conv Rate"] = (mtd_summary["MTD Conversions"] / mtd_summary["Enrollments (MTD created)"]).replace([np.inf, np.nan], 0).round(3)

    st.markdown("#### 📅 MTD (This Month)")
    st.dataframe(mtd_summary, use_container_width=True)

    st.markdown("#### 📅 Cohort (Choose a Month)")
    all_months = sorted([m for m in df_f["Payment Month"].dropna().unique()]) if "Payment Month" in df_f.columns else []
    default_cohort = all_months[-1] if all_months else pd.Timestamp.now(tz=TZ).to_period("M").to_timestamp()
    sel_cohort = st.selectbox("Cohort month (by Payment Received Date)", options=all_months if all_months else [default_cohort], index=len(all_months)-1 if all_months else 0)

    cohort_df = df_f[df_f["Payment Month"] == sel_cohort]
    g_cols_c = [time_col_p] + granularity if granularity else [time_col_p]
    cohort_tbl = cohort_df.groupby(g_cols_c).size().reset_index(name="Cohort Conversions")
    st.dataframe(cohort_tbl, use_container_width=True)

    if not cohort_tbl.empty:
        tv = "Payment Day:T" if time_view == "Day" else ("Payment Week:T" if time_view == "Week" else "Payment Month:T")
        chart = alt.Chart(cohort_tbl).mark_line(point=True).encode(
            x=alt.X(tv, title=time_view),
            y=alt.Y("Cohort Conversions:Q"),
            color=alt.Color(granularity[0]+":N") if granularity else alt.value("steelblue"),
            tooltip=list(cohort_tbl.columns)
        ).properties(height=320, title="Cohort Conversions")
        st.altair_chart(chart, use_container_width=True)

# ------------------------------
# TAB 3: Predictability (M0 + M−N)
# ------------------------------
with tab3:
    st.subheader("🔮 Predictability (runs on data **only up to last month**)")
    st.caption("• Trains a classification model on all deals up to **last month** and estimates conversions expected in the **running month**.\n"
               "• Splits into **M0** (created this month & convert this month) and **M−N** (created in past months but convert this month).")

    now = pd.Timestamp.now(tz=TZ)
    running_month = now.to_period("M").to_timestamp(tz=TZ)
    last_m = last_month_of(now)

    train_df = df_f[df_f["Create Date"] <= month_bounds(last_m)[1]].copy()
    train_df["Converted"] = train_df["Payment Received Date"].notna().astype(int)

    num_cols = [c for c in ["Age","Calls Connected","Sales Activities","Last Activity Count"] if c in train_df.columns]
    if "Last Call Connected" in train_df.columns:
        train_df["Days Since Last Call"] = (now - train_df["Last Call Connected"]).dt.days.replace({pd.NaT: np.nan})
        num_cols.append("Days Since Last Call")
    cat_cols = [c for c in ["Country","JetLearn Deal Source","Academic Counselor","Deal Stage"] if c in train_df.columns]
    features = num_cols + cat_cols + ["Create Month"]

    if "Create Month" in train_df.columns:
        # encode Create Month as rough ordinal bucket
        train_df["CreateMonthOrd"] = train_df["Create Month"].view("int64") // 10**9 // (30*24*3600)
        features = [f for f in features if f != "Create Month"] + ["CreateMonthOrd"]

    X = train_df[[c for c in features if c in train_df.columns]].copy()
    y = train_df["Converted"]

    model = None
    if SKLEARN_OK:
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, [c for c in X.columns if c in num_cols or c=="Days Since Last Call" or c=="CreateMonthOrd"]),
                ("cat", categorical_transformer, [c for c in X.columns if c in cat_cols])
            ],
            remainder="drop"
        )
        model = Pipeline(steps=[("prep", preprocessor), ("clf", GradientBoostingClassifier(random_state=42))])
        if X.shape[0] >= 50 and y.sum() > 0:
            model.fit(X, y)
            st.success(f"ML model trained on {X.shape[0]} deals up to {last_m.strftime('%b %Y')}.")
        else:
            st.warning("Not enough data or conversions for ML. Using naive baseline.")
            model = None
    else:
        st.warning("scikit-learn not found. Using baseline conversion-rate estimator for Predictability.\n"
                   "Add scikit-learn==1.3.2 to requirements.txt and redeploy to enable ML.")

    # Candidate pools
    m0_candidates = df_f[(df_f["Create Month"] == running_month)].copy()
    mn_candidates = df_f[(df_f["Create Month"] < running_month)].copy()

    def estimate_expected_conversions(cands: pd.DataFrame, label: str):
        if cands.empty:
            return 0, cands.assign(Score=0.0)
        # exclude already-converted in running month
        already = cands["Payment Month"] == running_month if "Payment Month" in cands.columns else False
        pool = cands[~already].copy()
        if pool.empty:
            return 0, cands.assign(Score=0.0)

        tmp = pool.copy()
        if "Last Call Connected" in tmp.columns:
            tmp["Days Since Last Call"] = (now - tmp["Last Call Connected"]).dt.days.replace({pd.NaT: np.nan})
        if "Create Month" in tmp.columns:
            tmp["CreateMonthOrd"] = tmp["Create Month"].view("int64") // 10**9 // (30*24*3600)
        feats = [c for c in features if c in tmp.columns]

        if model is not None and len(feats)>0:
            proba = model.predict_proba(tmp[feats])[:,1]
            tmp["Score"] = proba
            expected = float(np.sum(proba))
        else:
            hist_rate = y.mean() if len(y)>0 else 0.1
            tmp["Score"] = hist_rate
            expected = hist_rate * len(tmp)

        return expected, tmp

    m0_expected, m0_scored = estimate_expected_conversions(m0_candidates, "M0")
    mn_expected, mn_scored = estimate_expected_conversions(mn_candidates, "M−N")

    actual_running_conversions = safe_count(df_f[df_f["Payment Month"] == running_month]["Payment Received Date"]) if "Payment Month" in df_f.columns else 0
    st.metric("Actual Conversions (Running Month to-date)", actual_running_conversions)
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected M0 Conversions", int(round(m0_expected)))
    c2.metric("Expected M−N Conversions", int(round(mn_expected)))
    c3.metric("Total Expected (M0 + M−N)", int(round(m0_expected + mn_expected)))

    with st.expander("View scored candidates (top 50)"):
        if not m0_scored.empty:
            st.markdown("**M0 candidates**")
            st.dataframe(m0_scored.sort_values("Score", ascending=False).head(50), use_container_width=True)
        if not mn_scored.empty:
            st.markdown("**M−N candidates**")
            st.dataframe(mn_scored.sort_values("Score", ascending=False).head(50), use_container_width=True)

    st.markdown("#### 🔎 Expected Conversions — Breakdowns")
    by_opts = [c for c in ["Academic Counselor","JetLearn Deal Source","Country"] if c in df_f.columns]
    by_sel = st.multiselect("Group by...", options=by_opts, default=[by_opts[0]] if by_opts else [])

    def breakdown(df_scored, label):
        if df_scored.empty or not by_sel:
            return pd.DataFrame()
        return df_scored.groupby(by_sel)["Score"].sum().reset_index(name=f"Expected Conversions ({label})")

    m0_b = breakdown(m0_scored, "M0")
    mn_b = breakdown(mn_scored, "M−N")
    if not m0_b.empty or not mn_b.empty:
        out = m0_b.merge(mn_b, on=by_sel, how="outer").fillna(0.0)
        out["Expected Total"] = out.filter(like="Expected").sum(axis=1)
        st.dataframe(out.sort_values("Expected Total", ascending=False), use_container_width=True)

# ------------------------------
# TAB 4: AC Performance
# ------------------------------
with tab4:
    st.subheader("🧑‍💼 Academic Counselor Performance (Current Month)")
    start_this_m, end_this_m = month_bounds(pd.Timestamp.now(tz=TZ))
    ac_col = "Academic Counselor" if "Academic Counselor" in df_f.columns else None
    if ac_col is None:
        st.info("No 'Academic Counselor' column found. Upload data with this column to enable this view.")
    else:
        cur_m_created = df_f[df_f["Create Date"].between(start_this_m, end_this_m, inclusive="both")]
        cur_m_converted = df_f[(df_f["Payment Received Date"].between(start_this_m, end_this_m, inclusive="both"))]

        ac_enroll = cur_m_created.groupby(ac_col).size().reset_index(name="Enrollments (MTD)")
        ac_mtd_conv = cur_m_converted.groupby(ac_col).size().reset_index(name="MTD Conversions")
        ac_cohort = ac_mtd_conv.rename(columns={"MTD Conversions":"Cohort Conversions"})

        perf = ac_enroll.merge(ac_mtd_conv, on=ac_col, how="left").merge(ac_cohort, on=ac_col, how="left")
        perf = perf.fillna(0)
        perf["MTD Conv Rate"] = (perf["MTD Conversions"] / perf["Enrollments (MTD)"]).replace([np.inf, np.nan], 0).round(3)

        sort_by = st.selectbox("Sort by", ["Enrollments (MTD)","MTD Conversions","Cohort Conversions","MTD Conv Rate"], index=1)
        ascending = st.toggle("Ascending", value=False)

        st.dataframe(perf.sort_values(sort_by, ascending=ascending), use_container_width=True)

        long = perf.melt(id_vars=[ac_col], value_vars=["Enrollments (MTD)","MTD Conversions","Cohort Conversions"], var_name="Metric", value_name="Count")
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X(f"{ac_col}:N", sort="-y", title="Academic Counselor"),
            y=alt.Y("Count:Q"),
            color="Metric:N",
            tooltip=[ac_col, "Metric", "Count"]
        ).properties(height=360, title="AC-wise MTD Enrollments vs Conversions (Cohort included)").interactive()
        st.altair_chart(chart, use_container_width=True)

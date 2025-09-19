# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="JetLearn Analytics App", layout="wide")

# -------------------------------
# Utility functions
# -------------------------------

def to_month_start(ts):
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).to_period("M").to_timestamp()

def _floor_to_week(series: pd.Series, week_start: str = "MON") -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if week_start.upper().startswith("MON"):
        return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.normalize()
    elif week_start.upper().startswith("SUN"):
        return (s - pd.to_timedelta((s.dt.dayofweek + 1) % 7, unit="D")).dt.normalize()
    else:
        return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.normalize()

def add_time_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    if "Create Date" in df.columns:
        df["Create Date"] = pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
        df["Create Month"] = df["Create Date"].apply(to_month_start)
        df["Create Day"] = df["Create Date"].dt.date
        df["Create Week"] = _floor_to_week(df["Create Date"], week_start="MON").dt.date

    if "Payment Received Date" in df.columns:
        df["Payment Received Date"] = pd.to_datetime(df["Payment Received Date"], errors="coerce", dayfirst=True)
        df["Payment Month"] = df["Payment Received Date"].apply(to_month_start)
        df["Payment Day"] = df["Payment Received Date"].dt.date
        df["Payment Week"] = _floor_to_week(df["Payment Received Date"], week_start="MON").dt.date

    return df

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = add_time_columns(df)
    return df

def get_conversion_count(df, date_filter=None):
    if "Payment Received Date" not in df.columns:
        return 0
    if date_filter:
        return df[df["Payment Received Date"].dt.date == date_filter].shape[0]
    return df["Payment Received Date"].notna().sum()

# -------------------------------
# Streamlit App
# -------------------------------

st.title("📊 JetLearn Analytics App")

uploaded_file = st.sidebar.file_uploader("Upload Master_sheet_DB.csv", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    tab1, tab2, tab3, tab4 = st.tabs(["MIS", "Data Analysis", "Predictability", "AC Performance"])

    # -------------------------------
    # Tab 1: MIS
    # -------------------------------
    with tab1:
        st.header("Management Information System (MIS)")

        today = pd.Timestamp.today().date()
        yesterday = today - pd.Timedelta(days=1)

        yesterday_conv = get_conversion_count(df, yesterday)
        today_forecast = int(df["Payment Received Date"].notna().sum() / max(1, (df["Payment Received Date"].dt.day.max() - 1)) * today.day)

        last_month = (pd.Timestamp.today().replace(day=1) - pd.DateOffset(months=1)).month
        this_month = pd.Timestamp.today().month

        last_month_conv = df[df["Payment Received Date"].dt.month == last_month].shape[0]
        this_month_conv = df[df["Payment Received Date"].dt.month == this_month].shape[0]

        referrals = df[df["JetLearn Deal Source"] == "Referrals"]

        st.metric("Yesterday's Conversions", yesterday_conv)
        st.metric("Today's Forecasted Conversions", today_forecast)
        st.metric("Last Month Conversions", last_month_conv)
        st.metric("This Month Conversions (so far)", this_month_conv)
        st.metric("Referral Deals (This Month)", referrals[referrals["Create Month"].dt.month == this_month].shape[0])
        st.metric("Referral Conversions (This Month)", referrals[referrals["Payment Month"].dt.month == this_month].shape[0])

    # -------------------------------
    # Tab 2: Data Analysis
    # -------------------------------
    with tab2:
        st.header("Data Analysis (MTD vs Cohort)")

        st.write("MTD: Conversions from deals created this month")
        st.write("Cohort: Conversions in this month irrespective of create date")

        month_select = st.selectbox("Select Month", df["Create Month"].dropna().unique())

        mtd = df[(df["Create Month"] == month_select) & (df["Payment Month"] == month_select)]
        cohort = df[df["Payment Month"] == month_select]

        st.subheader("MTD Conversions")
        st.write(mtd.groupby("Create Day").size())

        st.subheader("Cohort Conversions")
        st.write(cohort.groupby("Payment Day").size())

    # -------------------------------
    # Tab 3: Predictability
    # -------------------------------
    with tab3:
        st.header("Predictability")

        st.write("Training ML model using historical data until last month.")

        # define target (conversion = Payment Received Date not null)
        df["Converted"] = df["Payment Received Date"].notna().astype(int)

        features = ["Age", "Country", "JetLearn Deal Source", "Number of Times Call Connected",
                    "Number of Sales Activity", "Last Call Connected", "Last Number of Activities"]
        X = df[features].copy()
        y = df["Converted"]

        # encode categoricals
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.metric("Model Accuracy", f"{acc:.2%}")

        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        st.bar_chart(importance_df.set_index("Feature"))

    # -------------------------------
    # Tab 4: AC Performance
    # -------------------------------
    with tab4:
        st.header("Academic Counselor Performance")

        if "Academic Counselor" in df.columns:
            ac_summary = df.groupby("Academic Counselor").agg(
                Leads=("Create Date", "count"),
                Conversions=("Payment Received Date", lambda x: x.notna().sum())
            ).reset_index()

            ac_summary["Conversion Rate"] = (ac_summary["Conversions"] / ac_summary["Leads"] * 100).round(2)

            st.dataframe(ac_summary.sort_values("Conversion Rate", ascending=False))

            chart = alt.Chart(ac_summary).mark_bar().encode(
                x="Academic Counselor",
                y="Conversions",
                tooltip=["Leads", "Conversions", "Conversion Rate"]
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("No 'Academic Counselor' column found in dataset.")

else:
    st.info("Please upload Master_sheet_DB.csv to get started.")

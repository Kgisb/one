
# JetLearn Sales Intelligence — Streamlit App (Master_sheet_DB.csv ready)

A mobile-friendly Streamlit app with a left **drawer** of filters and four tabs:
1) **MIS** — yesterday's conversions, today's forecast, last month actuals, this month forecast, and referral (MTD) metrics.
2) **Data Analysis (MTD & Cohort)** — MTD and Cohort analytics with Day/Week/Month granularity and group-by (AC / Source / Country).
3) **Predictability (M0 + M−N)** — ML forecast for the **running month**, trained **only up to last month**; shows M0 and M−N plus breakdowns.
4) **AC Performance** — current month AC-wise enrollments, MTD conversions, cohort conversions, sortable + chart.

## Use your file name
- The app **auto-loads `Master_sheet_DB.csv`** from the repo root by default.
- You can still upload a different CSV via the sidebar (which overrides the default).

## Data Definitions
- **Enrollment** = count of rows by `Create Date`.
- **Conversion** = count of rows by `Payment Received Date`.
- **MTD** = created in the running month & converted in the same month.
- **Cohort** = conversions that occurred in a chosen month (regardless of create month).

## Quick Start
```bash
pip install -U -r requirements.txt
streamlit run app.py
```
Place `Master_sheet_DB.csv` next to `app.py` in your repo folder for automatic loading.

## Notes
- Timezone: **Asia/Kolkata** for "today", "yesterday", and "running month".
- Auto-normalizes close column-name variants (e.g., "Jetline Deal Source" → `JetLearn Deal Source`).
- If dataset is too small for ML, the app falls back to a baseline historical conversion rate.


# JetLearn Sales Intelligence — FINAL

**Repo layout**
```
one/
 ├─ app.py                 # Main Streamlit app
 ├─ Master_sheet_DB.csv    # Data (auto-loaded at startup if present)
 ├─ requirements.txt
 └─ README.md
```

**Run locally**
```bash
pip install -U -r requirements.txt
streamlit run app.py
```

**Streamlit Cloud**
- Main file path: `app.py`
- Add `requirements.txt` in repo root.
- After pushing, click **Rerun** / **Restart** so the environment rebuilds.
- If `scikit-learn` is missing during build, the app will still run; the Predictability tab falls back to a baseline estimator.

**Definitions**
- **Enrollment** = count of `Create Date`
- **Conversion** = count of non-null `Payment Received Date`
- **MTD** = created & converted within the running month
- **Cohort** = conversions that happened in a chosen month (regardless of create month)
- **Predictability** = model trained on data up to end of last month; forecasts running-month **M0** and **M−N**.

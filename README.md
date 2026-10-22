# Flood Risk Predictor & Simulator — XGBoost version

This version preserves the existing Streamlit dashboard and rule-based flood-risk simulator, and adds an XGBoost model for forecasting monthly rainfall from the supplied historical rainfall CSV.

## Important dataset limitation

`flood_data.csv` contains monthly and seasonal rainfall by year. It does not contain observed flood/no-flood labels or the river-level, soil-saturation, and drainage measurements needed to train a genuine flood-event classifier. Therefore:

- **XGBoost:** forecasts monthly rainfall for the next 12 months.
- **Flood-risk score:** remains the existing transparent weighted simulator based on rainfall, river level, soil saturation, and drainage capacity.
- The project does **not** claim that XGBoost directly predicts whether a flood will occur.

## Run on Windows

Open Command Prompt in this folder and run:

```bash
py -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py` is unavailable, use `python` in place of `py` for the first command.

## Files

- `app.py` — Streamlit dashboard and flood-risk simulator, with the XGBoost forecast section.
- `xgboost_rainfall.py` — data preparation, XGBoost training, chronological holdout evaluation, and 12-month recursive forecast.
- `flood_data.csv` — supplied historical monthly rainfall data.
- `requirements.txt` — Python dependencies.

The model's MAE and RMSE are measured in millimetres against a chronological 12-month holdout. They are rainfall forecast errors, not flood classification accuracy.

"""XGBoost-based monthly rainfall forecasting from the project's historical CSV.

This dataset contains rainfall measurements, not observed flood/no-flood labels.
The model therefore forecasts rainfall; it does not directly classify flood events.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FEATURES = ["year", "month", "lag_1", "lag_2", "lag_3", "lag_12",
            "rolling_3", "rolling_12"]


def load_monthly_rainfall(csv_path):
    """Convert the wide year-by-month CSV into a clean monthly time series."""
    raw = pd.read_csv(csv_path)
    if "Year" not in raw.columns or not set(MONTHS).issubset(raw.columns):
        raise ValueError("CSV must contain Year and Jan-Dec columns.")

    # The provided CSV has a descriptive second row (e.g. Jan-Feb) with no Year.
    raw["Year"] = pd.to_numeric(raw["Year"], errors="coerce")
    raw = raw.dropna(subset=["Year"]).copy()
    raw["Year"] = raw["Year"].astype(int)

    rows = []
    for _, row in raw.iterrows():
        for month_number, month_name in enumerate(MONTHS, start=1):
            value = pd.to_numeric(row[month_name], errors="coerce")
            if pd.notna(value):
                rows.append({
                    "date": pd.Timestamp(year=int(row["Year"]), month=month_number, day=1),
                    "rainfall_mm": max(0.0, float(value)),
                })
    series = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
    if len(series) < 30:
        raise ValueError("Not enough monthly rainfall records to train XGBoost.")
    return series.reset_index(drop=True)


def _make_features(series):
    data = series.copy()
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    values = data["rainfall_mm"]
    data["lag_1"] = values.shift(1)
    data["lag_2"] = values.shift(2)
    data["lag_3"] = values.shift(3)
    data["lag_12"] = values.shift(12)
    data["rolling_3"] = values.shift(1).rolling(3).mean()
    data["rolling_12"] = values.shift(1).rolling(12).mean()
    return data.dropna(subset=FEATURES + ["rainfall_mm"]).reset_index(drop=True)


def _new_model():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=1,
    )


def evaluate_model(csv_path):
    """Chronological holdout: predict the final 12 observed months."""
    series = load_monthly_rainfall(csv_path)
    featured = _make_features(series)
    if len(featured) < 25:
        raise ValueError("Not enough usable rows after creating lag features.")
    split = max(1, len(featured) - 12)
    train, test = featured.iloc[:split], featured.iloc[split:]
    model = _new_model()
    model.fit(train[FEATURES], train["rainfall_mm"])
    predictions = np.maximum(0, model.predict(test[FEATURES]))
    actual = test["rainfall_mm"].to_numpy()
    metrics = {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "r2": float(r2_score(actual, predictions)) if len(actual) > 1 else float("nan"),
        "test_months": int(len(test)),
    }
    comparison = pd.DataFrame({
        "Month": test["date"].dt.strftime("%b %Y").to_numpy(),
        "Actual Rainfall (mm)": actual,
        "Predicted Rainfall (mm)": predictions,
    })
    return metrics, comparison


def forecast_next_12_months(csv_path):
    """Fit on all available observations and recursively forecast 12 months."""
    series = load_monthly_rainfall(csv_path)
    featured = _make_features(series)
    model = _new_model()
    model.fit(featured[FEATURES], featured["rainfall_mm"])

    history = series[["date", "rainfall_mm"]].copy().reset_index(drop=True)
    last_date = history["date"].iloc[-1]
    outputs = []
    for step in range(1, 13):
        date = last_date + pd.DateOffset(months=step)
        vals = history["rainfall_mm"].to_numpy(dtype=float)
        row = {
            "year": date.year,
            "month": date.month,
            "lag_1": vals[-1],
            "lag_2": vals[-2],
            "lag_3": vals[-3],
            "lag_12": vals[-12],
            "rolling_3": float(np.mean(vals[-3:])),
            "rolling_12": float(np.mean(vals[-12:])),
        }
        prediction = max(0.0, float(model.predict(pd.DataFrame([row], columns=FEATURES))[0]))
        outputs.append({"Month": date.strftime("%b %Y"), "Forecast Rainfall (mm)": round(prediction, 1)})
        history.loc[len(history)] = [date, prediction]
    return pd.DataFrame(outputs)

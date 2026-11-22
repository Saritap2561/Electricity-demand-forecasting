
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Loading electricity demand data...")

df = pd.read_csv("AEP_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values("Datetime")
df = df.dropna(subset=["AEP_MW"])
df = df.drop_duplicates(subset=["Datetime"])
df = df.reset_index(drop=True)

# Create time-based features
df["hour"] = df["Datetime"].dt.hour
df["day_of_week"] = df["Datetime"].dt.dayofweek
df["month"] = df["Datetime"].dt.month
df["day_of_year"] = df["Datetime"].dt.dayofyear

# Previous demand values
df["lag_1"] = df["AEP_MW"].shift(1)
df["lag_24"] = df["AEP_MW"].shift(24)
df["lag_168"] = df["AEP_MW"].shift(168)

df = df.dropna().reset_index(drop=True)

features = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "lag_1",
    "lag_24",
    "lag_168",
]

X = df[features]
y = df["AEP_MW"]

# Time-based split: earliest 80% for training
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Random Forest predictions
predictions = model.predict(X_test)

rf_mae = mean_absolute_error(y_test, predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, predictions))


# Evaluate a recursive 24-hour forecast
print("\nEvaluating recursive 24-hour forecast...")

history = df["AEP_MW"].iloc[
    split_index - 168:split_index
].tolist()

recursive_predictions = []

for i in range(24):

    forecast_time = df["Datetime"].iloc[split_index + i]

    row = {
        "hour": forecast_time.hour,
        "day_of_week": forecast_time.dayofweek,
        "month": forecast_time.month,
        "day_of_year": forecast_time.dayofyear,
        "lag_1": history[-1],
        "lag_24": history[-24],
        "lag_168": history[-168],
    }

    input_data = pd.DataFrame(
        [[row[name] for name in features]],
        columns=features
    )

    prediction = float(model.predict(input_data)[0])

    recursive_predictions.append(prediction)
    history.append(prediction)

actual_24_hours = y_test.iloc[:24]

recursive_mae = mean_absolute_error(
    actual_24_hours,
    recursive_predictions
)

recursive_rmse = np.sqrt(
    mean_squared_error(
        actual_24_hours,
        recursive_predictions
    )
)

print("\n--- RECURSIVE 24-HOUR FORECAST RESULTS ---")
print(f"MAE:  {recursive_mae:.2f} MW")
print(f"RMSE: {recursive_rmse:.2f} MW")


# Baseline: predict demand using the previous day's same hour
baseline_predictions = X_test["lag_24"].to_numpy()

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)
# Save evaluation results for the dashboard
evaluation_results = {
    "Random Forest MAE": rf_mae,
    "Random Forest RMSE": rf_rmse,
    "Recursive 24-Hour MAE": recursive_mae,
    "Recursive 24-Hour RMSE": recursive_rmse,
    "Baseline MAE": baseline_mae,
    "Baseline RMSE": baseline_rmse,
    "Training Records": len(X_train),
    "Testing Records": len(X_test),
}

joblib.dump(evaluation_results, "evaluation_results.joblib")

print("Evaluation results saved.")

print("\n--- MODEL COMPARISON ---")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nRandom Forest:")
print(f"MAE:  {rf_mae:.2f} MW")
print(f"RMSE: {rf_rmse:.2f} MW")

print("\nPrevious-Day Baseline:")
print(f"MAE:  {baseline_mae:.2f} MW")
print(f"RMSE: {baseline_rmse:.2f} MW")

if rf_mae < baseline_mae:
    print("\nRandom Forest has lower MAE than the baseline.")
else:
    print("\nRandom Forest does not have lower MAE than the baseline.")

# Save model and feature names
joblib.dump(
    {"model": model, "features": features},
    "electricity_model.joblib"
)

print("\nModel saved as electricity_model.joblib")
print("Training completed successfully!")

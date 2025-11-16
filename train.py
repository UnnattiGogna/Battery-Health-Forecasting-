import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv("final_merged_ev_dataset.csv")

# Select features and target
X = df[["soc", "battery_temp", "battery_capacity_kwh", "model_cluster"]]
y = df["soh"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# Model
# --------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------
# Evaluate
# --------------------------
preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("R² Score:", r2_score(y_test, preds))

# --------------------------
# Save model
# --------------------------
joblib.dump(model, "final_rf_model.joblib")

print("Model saved as final_rf_model.joblib")

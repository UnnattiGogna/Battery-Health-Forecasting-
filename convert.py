import pickle
import joblib

with open("final_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

joblib.dump(model, "final_rf_model.joblib")
print("Converted successfully!")

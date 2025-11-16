import pandas as pd
import joblib

# Load your cleaned dataset
df = pd.read_csv("final_merged_ev_dataset.csv")

# Create dictionary from model → cluster
cluster_map = dict(zip(df["model"].str.lower(), df["model_cluster"]))

# Save the map
joblib.dump(cluster_map, "model_cluster_map.joblib")

print("Saved model_cluster_map.joblib successfully!")

import streamlit as st
import re
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="EV Battery Health Predictor",
    layout="centered",
)

# ------------------------------
# PAGE TEXT
# ------------------------------
st.title("Your Personal EV Battery Health Predictor and Advisor")
st.write("Type SOC, temp, kWh, and EV model. Example: `SOC 80, temp 25, 64 kWh, kona`.")

# ------------------------------
# MODEL LOADING
# ------------------------------
try:
    model = joblib.load("final_rf_model.joblib")
    cluster_map = joblib.load("model_cluster_map.joblib")
except:
    st.error("Model files not found. Place final_rf_model.joblib and model_cluster_map.joblib in the same folder.")
    st.stop()

# ------------------------------
# INPUT PARSER
# ------------------------------
def parse_user_input(text):
    text = text.lower()

    # Extract SOC
    soc_match = re.search(r"soc\s*(\d+)", text)
    soc_val = int(soc_match.group(1)) if soc_match else None

    # Extract temperature
    temp_match = re.search(r"(temp|temperature)\s*(\d+)", text)
    temp_val = int(temp_match.group(2)) if temp_match else None

    # Extract capacity in kWh
    cap_match = re.search(r"(\d+)\s*kwh", text)
    cap_val = int(cap_match.group(1)) if cap_match else None

    # Extract model: take last word that is not a number or 'kwh'
    words = re.findall(r'\b[a-zA-Z\-]+\b', text)
    model_name_val = words[-1] if words else None

    return soc_val, temp_val, cap_val, model_name_val

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def predict_soh(soc, temp, cap, model_name):
    cluster = cluster_map.get(model_name, 0)
    x = np.array([[soc, temp, cap, cluster]])
    return round(model.predict(x)[0], 2)

# ------------------------------
# SEMI-CIRCLE GAUGE FUNCTION
# ------------------------------
def soh_gauge(soh):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.2)
    ax.axis('off')

    # Colored arcs for SOH ranges
    ranges = [(0, 60, 'red'), (60, 80, 'orange'), (80, 100, 'lime')]
    for start, end, color in ranges:
        theta1 = 180 * (1 - start / 100)
        theta2 = 180 * (1 - end / 100)
        ax.add_patch(plt.matplotlib.patches.Wedge((0,0), 1, theta2, theta1, width=0.3, facecolor=color, alpha=0.7))

    # Needle
    angle = 180 * (1 - soh / 100)
    ax.plot([0, 0.9 * np.cos(np.radians(angle))], [0, 0.9 * np.sin(np.radians(angle))], lw=3, color='white')

    # Center circle
    ax.add_patch(plt.Circle((0,0), 0.05, color='white'))

    # Text labels
    ax.text(0, -0.2, f"{soh}%", fontsize=20, fontweight='bold', ha='center', color='white')
    ax.text(-0.95, 0, "0%", fontsize=12, color='white')
    ax.text(0, 1.05, "50%", fontsize=12, color='white', ha='center')
    ax.text(0.95, 0, "100%", fontsize=12, color='white', ha='right')

    # Background
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')

    return fig

# ------------------------------
# CHATBOT ADVICE FUNCTION
# ------------------------------
def battery_advice(soc, temp, cap, model_name, soh):
    advice = []

    # SOH insight
    if soh > 90:
        advice.append(f"✅ Your battery is excellent at {soh}% SOH.")
    elif soh > 80:
        advice.append(f"✅ Your battery is healthy at {soh}% SOH.")
    elif soh > 60:
        advice.append(f"⚠️ Battery at {soh}% SOH — consider monitoring and adjusting charging habits.")
    else:
        advice.append(f"❌ Low battery health ({soh}% SOH). Service recommended soon.")

    # Charging recommendations
    advice.append("🔌 For best battery life, keep SOC between 20-80% for daily use.")
    if soc > 90:
        advice.append("⚠️ SOC is high — avoid keeping battery fully charged for long periods.")

    # Temperature advice
    if temp < 15:
        advice.append("❄️ Battery is cold — precondition your battery before driving.")
    elif temp > 40:
        advice.append("🔥 Battery is hot — avoid fast charging until cooled.")

    # Range estimation
    usable_capacity = cap * soc / 100
    efficiency = 5  # km per kWh (adjust as needed for real EVs)
    estimated_range = round(usable_capacity * efficiency, 1)
    advice.append(f"📏 Estimated usable range: ~{estimated_range} km based on current SOC.")

    # Maintenance tips
    advice.append("🛠️ Tips: Avoid frequent full charges and deep discharges. Keep battery within optimal temperature.")

    # Model-specific advice
    advice.append(f"🚗 Model-specific tip ({model_name.title()}): Follow manufacturer charging guidelines for best battery life.")

    # Fun analogy
    advice.append("💡 Think of your battery like a water tank — 85% means some storage has been worn out, but it still works!")

    return advice

# ------------------------------
# USER INPUT
# ------------------------------
user_input = st.text_input("Enter EV details:", placeholder="SOC 75 temp 30 98 kWh mustang")

if user_input:
    soc, temp, cap, model_name = parse_user_input(user_input)

    if None in [soc, temp, cap, model_name]:
        st.warning("❗ Please include SOC, temperature, kWh capacity, and model.")
    else:
        soh = predict_soh(soc, temp, cap, model_name)
        st.success(f"🔋 Predicted SOH: {soh}%")
        st.pyplot(soh_gauge(soh))

        # Display AI chatbot advice
        advice_list = battery_advice(soc, temp, cap, model_name, soh)
        for item in advice_list:
            st.info(item)

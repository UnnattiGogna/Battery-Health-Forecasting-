
# 🔋 EV Battery Health Prediction ML

Predict and monitor the **battery health (State of Health – SOH)** of Electric Vehicles (EVs) using real-world performance, charging, and environmental data.
This helps manufacturers, fleet owners, and individual users **detect battery degradation early** and **extend battery lifespan** efficiently.

---

## 🧩 Problem Statement

Battery degradation is one of the most critical challenges for Electric Vehicles.
Over time, due to charging cycles, temperature fluctuations, and driving behavior, the **battery’s State of Health (SOH)** declines — reducing range and performance.

The goal of this project was to build an **ML model** that can accurately **predict the battery’s SOH** based on various measurable factors like:

* Battery voltage and current
* Charging and discharging cycles
* Temperature and ambient conditions
* Energy consumption patterns
* Vehicle model and manufacturer

This prediction allows users to monitor **battery condition**, **plan replacements**, and **optimize charging behavior** proactively.

---

## 🛠️ Approach / Solution

I have designed and trained a regression model to estimate the **SOH** of EV batteries.
Here’s how I've approached the problem step-by-step:

### 1️⃣ Data Collection & Cleaning

* Merged multiple EV datasets containing **battery specs**, **charging patterns**, and **vehicle information**.
* Removed missing or inconsistent entries.
* Engineered new features such as:

  * Temperature-centered and temperature-squared columns
  * Normalized voltage and cycle-based degradation ratios


---

### 2️⃣ Exploratory Data Analysis (EDA)

* Analyzed correlations between temperature, cycles, and SOH.
* Visualized degradation patterns across brands and time.
* Identified **key influencing features** — temperature, charge cycles, and voltage had the strongest correlation with SOH drop.

---

### 3️⃣ Model Selection

Tested multiple regression models:

| Model                         | R² Score  | MAE      | Remarks                        |
| ----------------------------- | --------- | -------- | ------------------------------ |
| Linear Regression             | 0.21      | 3.6      | Weak correlation               |
| Decision Tree                 | 0.43      | 2.9      | Moderate performance           |
| **Random Forest Regressor** ✅ | **0.66+** | **1.47** | Best performance and stability |

✅ **Selected Model:** Random Forest Regressor

---

### 4️⃣ Model Training & Evaluation

* **Train/Test Split:** 80% training, 20% testing
* **Evaluation Metrics:** R² Score, Mean Absolute Error (MAE)
* **Final Model Performance:**

| Metric   | Value      |
| -------- | ---------- |
| R² Score | **0.6678** |
| MAE      | **1.4786** |

---

## 📊 Dataset

**Key Features Used:**

* `voltage`
* `current`
* `temperature`
* `cycles`
* `charging_time`
* `vehicle_model`
* `manufacturer`
* Engineered: `temp_centered`, `temp_squared`

**Target Variable:**

* `soh` — State of Health (in %)

---

📦 Model Download

The trained Random Forest model for predicting EV Battery Health (SOH) is available here:
🔗 Download final_rf_model.pkl (Google Drive) [https://drive.google.com/file/d/1hjMp2rsHz_U0vKUwe2s722NsosUkRebh/view?usp=share_link]

ℹ️ Due to GitHub’s 25 MB limit, the model is hosted externally on Google Drive.
Download it and place it in your working directory before running predictions.

---

✨Enhancements & Future Scope

* Integrate real-time telemetry data for continuous SOH monitoring.
* Deploy as an **API endpoint** for EV dashboards.
* Implement **LSTM or XGBoost** for time-series degradation tracking.
* Add explainability layer using **SHAP** or **LIME** to interpret feature impact.





import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- About this App ---
st.title("Perovskite Automated Synthesis System - Pro Version")
st.write("""
This app predicts the **Bandgap** and **Efficiency** of perovskite solar cell inks based on:
- **Ink composition**
- **Additives used**
- **Light intensity during synthesis**

### Why this matters
Perovskite solar cells are a next-generation photovoltaic technology with high efficiency and low production costs.
Optimizing their synthesis conditions (like ink formulation and light exposure) directly impacts their performance.

This tool uses **a machine learning model trained on experimental data** to predict the optimal bandgap and efficiency
under different synthesis conditions — helping researchers fine-tune recipes for high-performance solar materials.
""")

# --- Load Data ---
@st.cache_data
def load_data():
    data = pd.read_excel("PL VALUES.xlsx")
    data['Bandgap'] = 1240 / data['Wavelength'].replace(0, np.nan)
    data = data.dropna(subset=['Bandgap'])
    return data

data = load_data()

# Efficiency Calculation Function
def calculate_efficiency(bandgap):
    E_loss = 0.3  # Energy loss (approximate)
    E_max = 2.0   # Maximum theoretical efficiency factor (Shockley-Queisser limit)
    efficiency = (bandgap - E_loss) / E_max
    return np.clip(efficiency, 0, 1)

data['Efficiency'] = calculate_efficiency(data['Bandgap'])

# --- Optional Data Preview ---
if st.checkbox("Show Raw Data with Bandgap & Efficiency"):
    st.write(data)

# --- Preprocessing (Encode categorical variables) ---
data_encoded = pd.get_dummies(data, columns=["Ink", "Additive"], drop_first=True)

# Features & Target
X = data_encoded[['Intensity'] + [col for col in data_encoded.columns if "Ink_" in col or "Additive_" in col]]
y = data_encoded['Bandgap']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training (Using Optimized Parameters) ---
model = RandomForestRegressor(
    n_estimators=103,
    max_depth=19,
    min_samples_split=7,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# --- Model Evaluation ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Model Performance (Optimized):** MAE = {mae:.4f}, R² = {r2:.4f}")

# --- Feature Importance ---
importance = model.feature_importances_
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)

st.write("### Feature Importance")
st.bar_chart(importance_df.set_index("Feature"))

# --- Efficiency Distribution Plot ---
st.write("### Efficiency Distribution (From Historical Data)")
plt.figure(figsize=(8, 5))
sns.histplot(data['Efficiency'], bins=30, kde=True, color='green')
plt.xlabel("Efficiency")
plt.ylabel("Count")
plt.grid(True)
st.pyplot(plt)

# --- Prediction Form ---
st.write("## Predict Bandgap & Efficiency for New Conditions")

intensity = st.number_input("Enter Intensity:", min_value=0, value=5000)
ink_options = data['Ink'].unique().tolist()
additive_options = data['Additive'].unique().tolist()

ink = st.selectbox("Select Ink Type:", ink_options)
additive = st.selectbox("Select Additive Type:", additive_options)

# --- Prepare Input for Prediction ---
input_data = pd.DataFrame([[intensity]], columns=['Intensity'])
for col in X.columns:
    if "Ink_" in col or "Additive_" in col:
        input_data[col] = 0

# Set correct ink/additive
ink_col = f"Ink_{ink}"
additive_col = f"Additive_{additive}"

if ink_col in input_data.columns:
    input_data[ink_col] = 1
if additive_col in input_data.columns:
    input_data[additive_col] = 1

# --- Predict Bandgap & Efficiency ---
predicted_bandgap = model.predict(input_data)[0]
predicted_efficiency = calculate_efficiency(predicted_bandgap)

st.write(f"### Predicted Bandgap: **{predicted_bandgap:.4f} eV**")
st.write(f"### Predicted Efficiency: **{predicted_efficiency:.2%}**")

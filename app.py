
import streamlit as st
import pandas as pd
import joblib

st.title("🚗 Vehicle Maintenance Dashboard")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")  # now only top 5
top_features_df = joblib.load("top_features.pkl")

input_data = []

# ✅ Only top 5 features exist now
for feature in features:

    if feature == "Reported_Issues":
        value = st.slider("Reported Issues", 0, 5)

    elif feature == "Battery_Status":
        option = st.selectbox("Battery Status", ["Weak", "Good", "New"])
        value = {"Weak":0, "Good":1, "New":2}[option]

    elif feature == "Brake_Condition":
        option = st.selectbox("Brake Condition", ["Worn Out", "Good", "New"])
        value = {"Worn Out":0, "Good":1, "New":2}[option]

    elif feature == "Service_History":
        value = st.slider("Service History", 1, 10)

    elif feature == "Maintenance_History":
        option = st.selectbox("Maintenance History", ["Poor", "Average", "Good"])
        value = {"Poor":0, "Average":1, "Good":2}[option]

    else:
        value = st.number_input(feature, 0, 10)

    input_data.append(value)

input_df = pd.DataFrame([input_data], columns=features)

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    pred = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    if pred[0] == 1:
        st.error("⚠️ Vehicle Likely to FAIL")
        st.write(f"🔴 Risk Probability: {prob:.2f}")
    else:
        st.success("✅ Vehicle is SAFE")
        st.write(f"🟢 Risk Probability: {prob:.2f}")

# Show top features
st.subheader("🔝 Top 5 Important Features")
st.dataframe(top_features_df)

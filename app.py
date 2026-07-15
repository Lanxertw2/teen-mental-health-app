import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Teen Mental Health Predictor",
    page_icon="🧠",
    layout="centered",
)

# Title & Description
st.title("🧠 Teen Mental Health Risk Assessment")
st.write(
    "Enter the daily habits and metrics below to estimate the likelihood of a high depression risk profile based on our trained machine learning model."
)
st.markdown("---")


# ==========================================
# LOAD THE MODEL & COLUMNS
# ==========================================
@st.cache_resource  # Keeps the model in memory so it doesn't reload on every click
def load_model_assets():
    model = joblib.load("mental_health_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns


try:
    model, model_columns = load_model_assets()
except FileNotFoundError:
    st.error(
        "Model assets not found! Please run 'train_and_save.py' first to generate the model files."
    )
    st.stop()

# ==========================================
# USER INPUTS (SIDEBAR & MAIN PANELS)
# ==========================================
st.subheader("📋 Enter Teenager's Daily Metrics")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=13, max_value=19, value=16)
    gender = st.selectbox("Gender", ["male", "female"])
    sleep_hours = st.slider(
        "Sleep Hours (per night)", min_value=3.0, max_value=10.0, value=7.0, step=0.5
    )
    screen_time_before_sleep = st.slider(
        "Screen Time Before Sleep (hours)",
        min_value=0.0,
        max_value=4.0,
        value=1.5,
        step=0.1,
    )
    daily_social_media_hours = st.slider(
        "Daily Social Media Usage (hours)",
        min_value=0.0,
        max_value=12.0,
        value=3.0,
        step=0.5,
    )
    platform_usage = st.selectbox("Primary Platform Used", ["Instagram", "TikTok", "Both"])

with col2:
    academic_performance = st.slider(
        "Academic Performance (GPA/Scale)",
        min_value=1.0,
        max_value=4.0,
        value=3.0,
        step=0.1,
    )
    physical_activity = st.slider(
        "Physical Activity (hours/day)",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )
    social_interaction_level = st.selectbox(
        "Social Interaction Level", ["low", "medium", "high"]
    )
    stress_level = st.slider(
        "Self-Reported Stress Level", min_value=1, max_value=10, value=5
    )
    anxiety_level = st.slider(
        "Self-Reported Anxiety Level", min_value=1, max_value=10, value=5
    )
    addiction_level = st.slider(
        "Screen/Social Addiction Level", min_value=1, max_value=10, value=5
    )


# ==========================================
# PREPROCESSING USER INPUT TO MATCH MODEL
# ==========================================
# 1. Put raw inputs into a DataFrame row
raw_input = {
    "age": age,
    "gender": gender,
    "daily_social_media_hours": daily_social_media_hours,
    "platform_usage": platform_usage,
    "sleep_hours": sleep_hours,
    "screen_time_before_sleep": screen_time_before_sleep,
    "academic_performance": academic_performance,
    "physical_activity": physical_activity,
    "social_interaction_level": social_interaction_level,
    "stress_level": stress_level,
    "anxiety_level": anxiety_level,
    "addiction_level": addiction_level,
}
input_df = pd.DataFrame([raw_input])

# 2. Apply One-Hot Encoding just like we did in training
categorical_cols = ["gender", "platform_usage", "social_interaction_level"]
encoded_input = pd.get_dummies(input_df, columns=categorical_cols)

# 3. Match columns exactly with training data structure (filling missing dummy columns with 0)
final_features = pd.DataFrame(columns=model_columns)
for col in model_columns:
    if col in encoded_input.columns:
        final_features[col] = encoded_input[col]
    else:
        final_features[col] = 0  # Fill categorical variables that weren't selected with 0

# Fill potential NaN values safely
final_features = final_features.fillna(0)

# ==========================================
# PREDICTION GENERATION
# ==========================================
st.markdown("---")
if st.button("Generate Prediction", type="primary"):
    # Run prediction and fetch probability scores
    prediction = model.predict(final_features)[0]
    probabilities = model.predict_proba(final_features)[0]

    # Display Results
    st.subheader("📊 Prediction Analysis")

    # Predicts "Depressed"
    if prediction == 1:
        st.error(
            f"⚠️ **High Risk Detected** (Confidence: {probabilities[1]*100:.1f}%)"
        )
        st.write(
            "The model identifies patterns consistent with high depression risk profiles. Please consider speaking with a mental health professional."
        )
    # Predicts "Not Depressed"
    else:
        st.success(
            f"✅ **Low Risk Detected** (Confidence: {probabilities[0]*100:.1f}%)"
        )
        st.write(
            "The model indicates a low correlation with depressive risk factors based on the current metrics."
        )

    # Visual gauge metric
    st.progress(float(probabilities[1]))
    st.caption(
        f"Probability of being classified with depression risk: {probabilities[1]*100:.1f}%"
    )
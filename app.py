import streamlit as st
import pandas as pd
import joblib

# --- PAGE CONFIG ---
st.set_page_config(page_title="Appointment No-Show Predictor", layout="centered")

# --- LOAD MODELS & ENCODERS ---
# Note: Ensure these files are in the same folder as your app.py
@st.cache_resource
def load_assets():
    model = joblib.load("appointment_rf_model_final_compressed.pkl")
    le_gender = joblib.load("le_gender.pkl")
    le_neigh = joblib.load("le_neighbourhood.pkl")
    return model, le_gender, le_neigh

try:
    rf_loaded, le_gender_loaded, le_neigh_loaded = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- APP UI ---
st.title("🏥 Patient No-Show Predictor")
st.markdown("Enter the patient details below to predict the probability of a no-show and get recommended actions.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=le_gender_loaded.classes_)
    neighbourhood = st.selectbox("Neighbourhood", options=le_neigh_loaded.classes_)
    lead_days = st.number_input("Lead Days (Wait Time)", min_value=0, value=7)
    
    # New Informative Slider
    age_group_map = {
        "0-18": 0, "19-30": 1, "31-45": 2, "46-60": 3, "61+": 4
    }
    age_label = st.select_slider("Patient Age Group", options=list(age_group_map.keys()))
    age_group = age_group_map[age_label]

with col2:
    appt_day = st.selectbox("Day of Week", 
                            options=[0, 1, 2, 3, 4, 5, 6], 
                            format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
    handcap = st.number_input("Handicap Level", min_value=0, max_value=4, value=0)
    sms_received = st.checkbox("SMS Received")
    scholarship = st.checkbox("Scholarship (Bolsa Família)")
    hipertension = st.checkbox("Hipertension")
    diabetes = st.checkbox("Diabetes")
    alcoholism = st.checkbox("Alcoholism")

# --- PREDICTION LOGIC ---
if st.button("Predict Risk"):
    # 1. Encode Categorical Inputs
    gender_enc = le_gender_loaded.transform([gender])[0]
    neighbourhood_enc = le_neigh_loaded.transform([neighbourhood])[0]

    # 2. Build DataFrame
    X_input = pd.DataFrame([{
        'Gender': gender_enc,
        'Neighbourhood': neighbourhood_enc,
        'Scholarship': 1 if scholarship else 0,
        'Hipertension': 1 if hipertension else 0,
        'Diabetes': 1 if diabetes else 0,
        'Alcoholism': 1 if alcoholism else 0,
        'Handcap': handcap,
        'SMS_received': 1 if sms_received else 0,
        'LeadDays': lead_days,
        'ApptDayOfWeek': appt_day,
        'AgeGroup': age_group
    }])

    # 3. Predict
    prob_no_show = rf_loaded.predict_proba(X_input)[0][1]

    # 4. Determine Tiers
    if prob_no_show < 0.30:
        risk, color, action = "Low", "green", "No action needed."
    elif prob_no_show < 0.70:
        risk, color, action = "Medium", "orange", "Send an automated SMS reminder."
    else:
        risk, color, action = "High", "red", "Priority: SMS + Personal phone call."

    # 5. Display Results
    st.divider()
    st.subheader(f"Results")
    
    c1, c2 = st.columns(2)
    c1.metric("No-Show Probability", f"{prob_no_show:.1%}")
    c2.markdown(f"**Risk Tier:** :{color}[{risk}]")
    
    st.info(f"**Recommended Action:** {action}")
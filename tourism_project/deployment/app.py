import os
import streamlit as st
import pandas as pd
import joblib

# Ensure the directory exists (for local testing without full pipeline)
# os.makedirs('tourism_project/deployment', exist_ok=True)

# Load the trained model
model_path = "tourism_project/deployment/tourism_package_prediction_v1.joblib"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Please ensure the model is trained and committed to this location.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Tourism Package Purchase Prediction")
st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# Input fields for features
with st.sidebar:
    st.header("Customer Information")
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    typeofcontact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0)
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Freelancer'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
    productpitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
    preferredpropertystar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    maritalstatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    numberoftrips = st.number_input("Number of Trips Annually", min_value=1.0, value=2.0)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0.0, value=0.0)
    designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director'])
    monthlyincome = st.number_input("Monthly Income", min_value=0.0, value=20000.0)
    numberoffollowups = st.number_input("Number of Followups", min_value=0.0, value=3.0)


if st.button("Predict Purchase"): # Moved the button here
    # Create DataFrame from inputs
    input_data = pd.DataFrame([{
        'Age': age,
        'TypeofContact': typeofcontact,
        'CityTier': citytier,
        'DurationOfPitch': durationofpitch,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': numberofpersonvisiting,
        'ProductPitched': productpitched,
        'PreferredPropertyStar': preferredpropertystar,
        'MaritalStatus': maritalstatus,
        'NumberOfTrips': numberoftrips,
        'Passport': passport,
        'PitchSatisfactionScore': pitchsatisfactionscore,
        'OwnCar': owncar,
        'NumberOfChildrenVisiting': numberofchildrenvisiting,
        'Designation': designation,
        'MonthlyIncome': monthlyincome,
        'NumberOfFollowups': numberoffollowups
    }])

    # Make prediction
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba >= 0.45).astype(int) # Using the same threshold as in training

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"The customer is likely to purchase the package! (Probability: {prediction_proba[0]:.2f})")
    else:
        st.warning(f"The customer is not likely to purchase the package. (Probability: {prediction_proba[0]:.2f})")
    
    st.write("\n--- Raw Input Data ---")
    st.dataframe(input_data)

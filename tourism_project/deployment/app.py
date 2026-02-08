import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Predictor", page_icon="✈️", layout="wide")

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id="tourism-package-prediction-model", filename="XGBoost_model.pkl", repo_type="model")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_input(data):
    mappings = {
        'TypeofContact': {'Company Invited': 0, 'Self Enquiry': 1},
        'Occupation': {'Salaried': 0, 'Small Business': 1, 'Large Business': 2, 'Free Lancer': 3},
        'Gender': {'Male': 0, 'Female': 1},
        'ProductPitched': {'Basic': 0, 'Standard': 1, 'Deluxe': 2, 'Super Deluxe': 3, 'King': 4},
        'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2, 'Unmarried': 3},
        'Designation': {'Executive': 0, 'Manager': 1, 'Senior Manager': 2, 'AVP': 3, 'VP': 4}
    }
    processed = data.copy()
    for col, mapping in mappings.items():
        if col in processed.columns:
            processed[col] = processed[col].map(mapping)
    return processed

def main():
    st.title("✈️ Tourism Package Prediction System")
    st.markdown("### Predict whether a customer will purchase the Wellness Tourism Package")
    st.markdown("---")

    model = load_model()
    if model is None:
        return

    st.sidebar.header("Input Method")
    input_method = st.sidebar.radio("Choose input method:", ["Manual Input", "CSV Upload"])

    if input_method == "Manual Input":
        st.header("📝 Enter Customer Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age", 18, 100, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
            designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            monthly_income = st.number_input("Monthly Income", 0, value=20000)

        with col2:
            st.subheader("Travel Preferences")
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
            number_of_person_visiting = st.number_input("Persons Visiting", 1, 10, 2)
            number_of_children_visiting = st.number_input("Children Visiting", 0, 5, 0)
            number_of_trips = st.number_input("Trips per Year", 0.0, 20.0, 2.0)
            passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        with col3:
            st.subheader("Interaction Details")
            type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
            product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
            duration_of_pitch = st.number_input("Duration of Pitch (min)", 0.0, 60.0, 15.0)
            number_of_followups = st.number_input("Follow-ups", 0.0, 10.0, 3.0)
            pitch_satisfaction_score = st.selectbox("Pitch Satisfaction", [1, 2, 3, 4, 5])

        input_data = pd.DataFrame({
            'Age': [age], 'TypeofContact': [type_of_contact], 'CityTier': [city_tier],
            'DurationOfPitch': [duration_of_pitch], 'Occupation': [occupation], 'Gender': [gender],
            'NumberOfPersonVisiting': [number_of_person_visiting], 'NumberOfFollowups': [number_of_followups],
            'ProductPitched': [product_pitched], 'PreferredPropertyStar': [preferred_property_star],
            'MaritalStatus': [marital_status], 'NumberOfTrips': [number_of_trips], 'Passport': [passport],
            'PitchSatisfactionScore': [pitch_satisfaction_score], 'OwnCar': [own_car],
            'NumberOfChildrenVisiting': [number_of_children_visiting], 'Designation': [designation],
            'MonthlyIncome': [monthly_income]
        })

        if st.button("🔮 Predict", type="primary"):
            processed_data = preprocess_input(input_data)
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]
            st.markdown("---")
            st.header("📊 Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.success("✅ Customer is likely to purchase!")
                else:
                    st.error("❌ Customer is unlikely to purchase.")
            with col2:
                st.metric("Confidence", f"{max(prediction_proba)*100:.2f}%")

    else:
        st.header("📁 Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            if st.button("🔮 Predict All", type="primary"):
                processed_data = preprocess_input(df)
                predictions = model.predict(processed_data)
                df['Prediction'] = predictions
                df['Prediction_Label'] = df['Prediction'].map({0: 'Will Not Purchase', 1: 'Will Purchase'})
                st.header("📊 Results")
                st.metric("Total", len(df))
                st.metric("Likely to Purchase", (predictions == 1).sum())
                st.dataframe(df[['Prediction_Label']])
                st.download_button("📥 Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")

if __name__ == "__main__":
    main()

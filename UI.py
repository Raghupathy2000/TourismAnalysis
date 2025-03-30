# import streamlit as st
# import pandas as pd
# import joblib

# # Load preprocessed dataset and saved model
# consolidated_df = pd.read_excel("preprocessed_data.xlsx")
# # For this example, we'll load the Random Forest Regressor.
# model = joblib.load("Random_Forest_Regressor.pkl")
# # Load label encoders (assumes you saved them as "label_encoders.pkl")
# label_encoders = joblib.load("label_encoders.pkl")

# st.title("Travel Recommendation System - Rating Predictor")

# st.sidebar.header("User Input Features")

# # Input: User ID (numeric)
# user_id = st.sidebar.number_input("User ID", min_value=1, value=70456, step=1)

# # Input: City ID selection (assumes CityId is numeric; adjust if needed)
# city_ids = consolidated_df["CityId"].unique()
# city_id = st.sidebar.selectbox("City ID", city_ids)

# # Input: Visit Year
# visit_year = st.sidebar.number_input("Visit Year", min_value=2020, value=2025, step=1)

# # Input: Visit Month (1-12)
# visit_month = st.sidebar.selectbox("Visit Month", list(range(1, 13)))

# # Input: Visit Mode using the label encoder's classes
# visit_mode_options = label_encoders["VisitMode"].classes_
# visit_mode_input = st.sidebar.selectbox("Visit Mode", visit_mode_options)

# # Encode the Visit Mode using the pre-fitted LabelEncoder
# encoded_visit_mode = label_encoders["VisitMode"].transform([visit_mode_input])[0]

# # Create input DataFrame for prediction
# input_data = pd.DataFrame({
#     "UserId": [user_id],
#     "CityId": [city_id],
#     "VisitYear": [visit_year],
#     "VisitMonth": [visit_month],
#     "VisitMode": [encoded_visit_mode]
# })

# # When button is pressed, predict the normalized rating
# if st.sidebar.button("Predict Rating"):
#     prediction = model.predict(input_data)[0]
#     st.subheader(f"Predicted Normalized Rating: {prediction:.3f}")

# # Optional: Show a sample of the preprocessed data
# st.subheader("Sample Preprocessed Data")
# st.write(consolidated_df.head())


import streamlit as st
import pandas as pd
import joblib

# Load preprocessed dataset and saved models
consolidated_df = pd.read_excel("preprocessed_data.xlsx")
regression_model = joblib.load("Random_Forest_Regressor.pkl")
classification_model = joblib.load("Random_Forest_Classifier.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Travel Recommendation System")
st.sidebar.header("User Input Features")

# User Inputs
user_id = st.sidebar.number_input("User ID", min_value=1, value=70456, step=1)
city_ids = consolidated_df["CityId"].unique()
city_id = st.sidebar.selectbox("City ID", city_ids)
visit_year = st.sidebar.number_input("Visit Year", min_value=2020, value=2025, step=1)
visit_month = st.sidebar.selectbox("Visit Month", list(range(1, 13)))

# Visit Mode (for classification prediction)
visit_mode_options = label_encoders["VisitMode"].classes_
visit_mode_input = st.sidebar.selectbox("Visit Mode", visit_mode_options)
encoded_visit_mode = label_encoders["VisitMode"].transform([visit_mode_input])[0]

# Create input DataFrame
# input_data = pd.DataFrame({
#     "UserId": [user_id],
#     "CityId": [city_id],
#     "VisitYear": [visit_year],
#     "VisitMonth": [visit_month]
# })

input_data = pd.DataFrame({
    "UserId": [user_id],
    "CityId": [city_id],    
    "VisitYear": [visit_year],
    "VisitMonth": [visit_month],
    "VisitMode": [encoded_visit_mode]
})

input_data_1 = pd.DataFrame({
    "UserId": [user_id],
    "CityId": [city_id],    
    "VisitYear": [visit_year],
    "VisitMonth": [visit_month],
    "AttractionTypeId": [13]
})

# Predict Rating
if st.sidebar.button("Predict Rating"):
    predicted_rating = regression_model.predict(input_data)[0]
    st.subheader(f"Predicted Rating: {predicted_rating:.3f}")

# Predict Visit Mode
if st.sidebar.button("Predict Visit Mode"):
    predicted_mode_encoded = classification_model.predict(input_data_1)[0]
    predicted_mode = label_encoders["VisitMode"].inverse_transform([predicted_mode_encoded])[0]
    st.subheader(f"Predicted Visit Mode: {predicted_mode}")

# Show Sample Data
st.subheader("Sample Preprocessed Data")
st.write(consolidated_df.head())


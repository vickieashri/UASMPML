import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

# Load the trained model and preprocessing transformer
model = joblib.load('best_model.pkl')
ct = joblib.load('preprocessor.pkl')  # Assuming you saved the ColumnTransformer

# Load data for encoding and scaling
data = pd.read_csv('onlinefoods.csv')
data = data.drop('Unnamed: 12', axis=1)  # Adjust if necessary

# Create LabelEncoders for each categorical feature
label_encoders = {}
categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications']
for column in categorical_features:
    if data[column].dtype == 'object' or data[column].dtype.name == 'category':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Function to preprocess input
def preprocess_input(user_input):
    processed_input = pd.DataFrame([user_input])
    for column in categorical_features:
        if column in processed_input.columns:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])[0]
            else:
                processed_input[column] = -1  # Or another placeholder value
    return ct.transform(processed_input)

# Streamlit app layout
st.markdown("""
    <style>
    .main {
        background-color: #FAF3F3; /* Pastel pink background */
    }
    h1 {
        color: #6A5ACD; /* Slate blue */
        text-align: center;
        margin-bottom: 20px;
    }
    h3 {
        color: #4B0082; /* Indigo */
        text-align: center;
    }
    .stButton>button {
        background-color: #FFB6C1; /* Light pink */
        color: white;
        padding: 12px 25px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF69B4; /* Hot pink */
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 15px;
        border: 1px solid #DDA0DD; /* Plum */
        border-radius: 5px;
        padding: 10px;
    }
    .info-box {
        background-color: #E6E6FA; /* Lavender */
        border-left: 5px solid #9370DB; /* Medium purple */
        padding: 15px;
        margin-top: 20px;
    }
    .info-title {
        font-weight: bold;
        font-size: 18px;
        color: #4B0082; /* Indigo */
    }
    .info-content {
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1>Prediksi Feedback Pelanggan Online Food</h1>', unsafe_allow_html=True)

# User inputs
age = st.number_input('Age', min_value=18, max_value=100, step=1)
gender = st.radio('Gender', ['Male', 'Female'])
marital_status = st.radio('Marital Status', ['Single', 'Married', 'Prefer not to say'])
occupation = st.radio('Occupation', ['Student', 'Employee', 'Self Employed', 'House wife'])
monthly_income = st.radio('Monthly Income', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])
educational_qualifications = st.radio('Educational Qualifications', ['Under Graduate', 'Graduate', 'Post Graduate', 'Ph.D', 'Uneducated', 'School'])
family_size = st.number_input('Family size', min_value=1, max_value=20, step=1)
latitude = st.number_input('Latitude', format="%.6f")
longitude = st.number_input('Longitude', format="%.6f")
pin_code = st.number_input('Pin code', min_value=0, step=1)
feedback = st.radio('Feedback', ['Positive', 'Negative'])

user_input = {
    'Age': age,
    'Gender': gender,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income,
    'Educational Qualifications': educational_qualifications,
    'Family size': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Pin code': pin_code,
    'Feedback': feedback
}

# Map numbers to labels
label_mapping = {0: 'No', 1: 'Yes'}

if st.button('Predict'):
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        prediction_label = label_mapping.get(prediction[0], 'Yes')
        st.write(f'Prediction: {prediction_label}')
        
        st.markdown("""
            <div class="info-box">
                <div class="info-title">Informasi Tentang Hasil Prediksi:</div>
                <div class="info-content">
                    <p><strong>Yes:</strong> Pelanggan melakukan tindakan tertentu</p>
                    <p><strong>No:</strong> Pelanggan tidak melakukan tindakan tersebut.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    except ValueError as e:
        st.error(f"Error in prediction: {e}")

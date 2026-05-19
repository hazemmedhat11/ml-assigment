import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os

# Load the trained Random Forest Regressor model and preprocessor
model = None
preprocessor = None

if os.path.exists("SwimModel.pkl") and os.path.exists("SwimPreprocessor.pkl"):
    model = joblib.load(open("SwimModel.pkl", 'rb'))
    preprocessor = joblib.load(open("SwimPreprocessor.pkl", 'rb'))
else:
    st.warning("⚠️ Model files (SwimModel.pkl / SwimPreprocessor.pkl) not found. Prediction will not work.")

# Function to load animations (if needed)
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Prediction function
def predict(features):
    prediction = model.predict(features)
    return prediction

# Page configuration
st.set_page_config(
    page_title='Swimmer Timer Prediction',
    page_icon=':trophy:',
    initial_sidebar_state='collapsed'
)

# Sidebar menu
with st.sidebar:
    choose = option_menu(
        None,
        ["Home", "Graphs", "About", "Contact"],
        icons=["house", "bar-chart", "book", "envelope"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "#6c757d", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#02ab21"}
        }
    )

# Home page
if choose == "Home":
    st.write('# 🏊 Swimmer Timer Prediction')
    st.subheader('Enter your details to predict the swimmer time')

    # User inputs
    location = st.text_input("Enter the location:")
    year = st.number_input("Enter the year:", min_value=1900, max_value=2100)
    distance = st.text_input("Enter the distance (in meters):")
    stroke = st.selectbox("Select stroke:", ["Freestyle", "Backstroke", "Breaststroke", "Butterfly", "Medley"])
    relay = st.selectbox("Is it a relay event?", [0, 1])
    gender = st.selectbox("Select gender:", ["Men", "Women"])
    team = st.text_input("Enter the team name:")
    athlete = st.text_input("Enter the athlete name:")

    try:
        distance = int(str(distance).replace('m', ''))
    except ValueError:
        st.error("Please enter distance in meters (e.g., 100 or 100m).")
        distance = 0

    features = pd.DataFrame({
        'Location': [location],
        'Year': [year],
        'Distance (in meters)': [distance],
        'Stroke': [stroke],
        'Relay?': [relay],
        'Gender': [gender],
        'Team': [team],
        'Athlete': [athlete]
    })

    if st.button("Predict"):
        if model is None or preprocessor is None:
            st.error("❌ Cannot predict — model files are missing. Please add SwimModel.pkl and SwimPreprocessor.pkl.")
        else:
            features_processed = preprocessor.transform(features)
            result = predict(features_processed)
            st.success(f'✅ The predicted result time is: **{result[0]:.2f} seconds**')

# Graphs page
elif choose == "Graphs":
    st.write("# 📊 Graphs")
    if not os.path.exists("Olympic_Swimming.csv"):
        st.error("❌ Olympic_Swimming.csv not found. Please add it to the same folder.")
    else:
        df = pd.read_csv("Olympic_Swimming.csv")

        st.write("## Dataset Overview")
        st.dataframe(df.head())

        st.write("## Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        correlation = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        st.pyplot(plt)

# About page
elif choose == "About":
    st.write("# About")
    st.write("This app provides swim time predictions based on user input.")

# Contact page
elif choose == "Contact":
    st.write("# Contact")
    st.write("For inquiries, please contact us at contact@example.com")





    

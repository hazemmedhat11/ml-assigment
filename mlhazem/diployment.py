import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Function to load data
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")


@st.cache_data
def train_model(df):
    
    df['ChestPainType'] = df['ChestPainType'].replace({'ATA': 1, 'NAP': 2, 'ASY':3, 'TA':4})
    df['Sex'] = df['Sex'].replace({'F': 0, 'M': 1})
   

    X = df[['Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS']]
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    return rf_model

def predict(rf_model, df):
    prediction = rf_model.predict(df)
    return prediction

def main():
    df = load_data()
    rf_model = train_model(df)
    st.title("The Heart Disease Prediction")
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Select From Menu", ["Predict"])

    if menu == "Predict":
        st.subheader("Predict")
        sex = st.radio("Sex (0: Female, 1: Male)", [0, 1], index=1)
        chest_pain_type = st.selectbox("Chest Pain Type(1: ATA, 2: NAP, 3: ASY, 4: TA )", [1, 2, 3, 4], index=2)
        resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=150)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=300, value= 130)
        fasting_bs = st.radio("Fasting Blood Sugar", [0, 1], index=1)

        if st.button("Predict Now"):
            user_sample = pd.DataFrame({
                'Sex': [sex],
                'ChestPainType': [chest_pain_type],
                'RestingBP': [resting_bp],
                'Cholesterol': [cholesterol],
                'FastingBS': [fasting_bs],
            })

            prediction = predict(rf_model, user_sample)
            if prediction[0] == 1:
                st.write("Prediction: Heart Disease (Presence)")
            else:
                st.write("Prediction: No Heart Disease (Absence)")
                
if __name__ == '__main__':
    main()


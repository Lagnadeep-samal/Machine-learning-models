import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_model():
    with open("C:/Users/KIIT0001/Downloads/saved_steps.pkl", 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Ensure all known countries are included
known_countries = [
    "United States", "India", "United Kingdom", "Germany", "Canada", "Brazil",
    "France", "Spain", "Australia", "Netherlands", "Poland", "Italy",
    "Russian Federation", "Sweden", "Mexico", "China", "South Korea", "Japan",
    "South Africa", "Indonesia", "Turkey", "Argentina", "Singapore",
    "Vietnam", "Ukraine", "Pakistan", "Bangladesh", "Egypt", "Malaysia"
]

education_levels = [
    "Less than a Bachelor's", "Bachelor's degree",
    "Master's degree", "Post grad"
]

# Retrain label encoders to ensure all expected values are included
if set(known_countries) != set(le_country.classes_):
    le_country = LabelEncoder()
    le_country.fit(known_countries)

if set(education_levels) != set(le_education.classes_):
    le_education = LabelEncoder()
    le_education.fit(education_levels)

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("### We need some information to predict the salary")

    country = st.selectbox("Country", known_countries)
    education = st.selectbox("Education Level", education_levels)
    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")

    if ok:
        try:
            # Validate country & education encoding before transformation
            if country not in le_country.classes_:
                st.error(f"Country '{country}' is not recognized. Try another option.")
                return

            if education not in le_education.classes_:
                st.error(f"Education level '{education}' is not recognized. Try another option.")
                return

            X = np.array([[country, education, experience]])

            # Proper encoding transformation
            X[:, 0] = le_country.transform([country])
            X[:, 1] = le_education.transform([education])
            X = X.astype(float)

            salary = regressor.predict(X)
            st.subheader(f"The estimated salary is ${salary[0]:,.2f}")

        except ValueError as ve:
            st.error(f"Encoding error: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

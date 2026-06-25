import streamlit as st
import pickle
import numpy as np

# Page Config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.stButton>button {
    height: 3em;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Load model and scaler
model = pickle.load(open("model/best_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

st.sidebar.title("❤️ About Project")

st.sidebar.info("""
This project predicts whether a patient is likely
to have heart disease based on medical parameters.

Models Tested:
✔ Logistic Regression
✔ KNN
✔ SVM
✔ Decision Tree
✔ Naive Bayes

Best Model: Decision Tree
Accuracy: 98.5%
""")

# Header
st.title("❤️ Heart Disease Prediction System")
st.markdown("""
Predict the likelihood of heart disease using Machine Learning.

Fill in the patient's medical details below and click **Predict**.
""")


st.divider()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 0)
    chol = st.number_input("Cholesterol", 0)
    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate", 0)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Old Peak", 0.0, step=0.1)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

# Convert sex to model format
sex = 1 if sex == "Male" else 0

st.divider()

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    features = np.array([[age, sex, cp, trestbps, chol,
                          fbs, restecg, thalach, exang,
                          oldpeak, slope, ca, thal]])

    

    prediction = model.predict(features)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

st.divider()

st.caption("Built using Python, Scikit-Learn and Streamlit")
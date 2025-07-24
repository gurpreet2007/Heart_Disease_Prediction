import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Multiple Disease Prediction",layout="wide")


# Sidebar Menu
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to",["Home","Predict Heart Disease","Predict Diabetes","About Project"])

# HOME PAGE
if option == "Home":
    st.title("Multiple Disease Prediction App")

    st.markdown("""
    Welcome to the App!

    This app uses **Machine Learning** to help predict your risk of two major lifestyle diseases:

    -Heart Disease 
    
    -Diabetes
    """)
    
    # Image columns
    col1, col2 = st.columns(2)

    with col1:
        st.image("https://media.istockphoto.com/id/1359314170/photo/heart-attack-and-heart-disease-3d-illustration.jpg?s=612x612&w=0&k=20&c=K5Y-yzsfs7a7CyuAw-B222EMkT04iRmiEWzhIqF0U9E=", caption="Heart Disease", width=300)
    with col2:
        st.image("https://f.hubspotusercontent30.net/hubfs/2027031/diabetes.jpeg", caption="Diabetes",width=300)
        
        
    st.markdown("""
    
    ###  How It Works:
    1. Select a disease prediction page from the sidebar (Heart or Diabetes).
    2. Enter your health-related inputs in the form.
    3. Get an instant prediction about your risk status.
    
    ###  Disclaimer:
    > This tool is for **educational and informational purposes only**.  
    > It does **not replace professional medical diagnosis or advice**.  
    > Please consult a qualified doctor for any health-related concerns.
    
    👉 **Select a disease prediction model from the sidebar to begin!**

    """)




# 
elif option == "Predict Heart Disease":
    st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
    st.title("Heart Disease Prediction using Machine Learning")

    # Load the trained model
    model_path = "saved_models/heart.joblib"  
    try:
        loaded_heart_model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    st.markdown("### Enter the patient's data below:")
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)

    
    chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=70, max_value=210,value=150)

    
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment (slope)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored (ca)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3])

    # Prediction
    if st.button("Predict"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,thalach, exang, oldpeak, slope, ca, thal]])
        prediction = loaded_heart_model.predict(input_data)

        if prediction[0] == 0:
            st.success(" The person is **NOT** affected with heart disease.")
        else:
            st.error(" The person **IS** affected with heart disease.")


elif option == "Predict Diabetes":
    st.set_page_config(page_title="Diabetes Disease Prediction", layout="wide")
    st.title("Diabetes Disease Prediction using Machine Learning")

    # Load the trained model
    model_path = "saved_models/diabetes.joblib"  
    try:
        loaded_diabetes_model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}")
        st.stop()
        
    st.markdown("### Enter the patient's data below:")
    
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 50, 300, 120)
    bp = st.number_input("Blood Pressure", 40, 150, 70)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)   
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 70.0, 25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
    age = st.number_input("Age", 18, 130, 30)


    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, bp, skin_thickness,insulin, bmi, dpf, age]])
        result = loaded_diabetes_model.predict(input_data)[0]
        if result == 1:
            st.error(" High risk of Diabetes detected!")
        else:
            st.success(" No Diabetes risk detected.")
            
elif option == "About Project":
    st.set_page_config(page_title="Project Details", page_icon="🩺")

    st.title("🩺 Multiple Disease Prediction using Machine Learning")
    st.markdown("---")

    # Problem Statement
    st.subheader("🧩 Problem Statement")
    st.write("""
    Early detection of chronic diseases such as **heart disease** and **diabetes** can significantly improve treatment outcomes and reduce health risks.
    However, many patients do not have easy access to predictive tools that assist in identifying risks based on simple medical data.
    """)

    # Solution
    st.subheader("💡 Solution")   
    st.write("""
    This project provides a **Streamlit-based web application** that uses **machine learning models** to predict whether a person is likely to have **heart disease** or **diabetes**, 
    based on input features like glucose level, blood pressure, age, cholesterol, and other key health indicators.
    """)

    # Project Overview
    st.subheader("📝 Project Overview")
    st.write("""
    This ML-powered application allows users to input medical parameters and get an instant prediction for:

    - 🫀 **Heart Disease Prediction**
    - 💉 **Diabetes Prediction**

    Separate models are trained for each disease using real-world datasets to make predictions on user input.""")

    # Target Variable 
    
    st.subheader("🎯 Target Variables")
    st.markdown("""- **Heart Disease Prediction:** `target` (1 = disease, 0 = no disease)  
                
    - **Diabetes Prediction:** `Outcome` (1 = diabetes, 0 = no diabetes)""")

    # Features Considered
    st.subheader("🔍 Features Considered")

    with st.expander("Heart Disease Features"):
        st.markdown("""
        - Age  
        - Sex  
        - Chest Pain Type (`cp`)  
        - Resting Blood Pressure (`trestbps`)  
        - Cholesterol (`chol`)  
        - Fasting Blood Sugar (`fbs`)  
        - Maximum Heart Rate Achieved (`thalach`)  
        - Exercise Induced Angina (`exang`)  
        - ST Depression (`oldpeak`)  
        - Slope  
        - CA  
        - Thal
        """)

    with st.expander("Diabetes Features"):
        st.markdown("""
        - Pregnancies  
        - Glucose  
        - Blood Pressure  
        - Skin Thickness  
        - Insulin  
        - BMI  
        - Diabetes Pedigree Function  
        - Age
        """)

    # Steps Performed
    st.subheader(" Steps Performed")
    st.markdown("""
    1. **Data Collection** : Dataset sourced from **Kaggle**
    2. **Data Cleaning** (handling missing or zero values, encoding)  
    3. **Exploratory Data Analysis (EDA)**  
    4. **Feature Selection**  
    5. **Model Training** with Logistic Regression, Random Forest, XGBoost, SVM, Decision Tree ,Catboost,LightGBM
    6. **Model Evaluation** (accuracy, precision, recall, F1-score)  
    7. **Building Web App** using Streamlit  
    8. **Model Deployment** using `joblib` to save and load trained models in the application
    """)
    
    st.subheader(" Deployment")
    st.write("""This machine learning web application is deployed live on the cloud using **Render**.  
    It allows users to interact with the model in real-time from any device with internet access.
    """)


    # Tools & Technologies
    st.subheader("Tools & Technologies Used")
    st.markdown("""
    - **Programming Language**: Python  
    - **Machine Learning Libraries**: scikit-learn, XGBoost  
    - **Data Analysis & Processing**: pandas, numpy  
    - **Model Deployment**: joblib, Streamlit  
    - **Visualization**: matplotlib, seaborn  
    - **Web App Framework**: Streamlit  
    - **Cloud Deployment**: Render  
    - **Version Control**: Git, GitHub  
    - **Development Tools**: Jupyter Notebook, VS Code  
    """)


    st.subheader(" Submitted By")

    st.markdown("""
    **Gurpreet Kaur**  
    B.Tech in Electronics and Computer Engineering (2023–2027)  
    **Guru anak Dev University**  
    CGPA: *8.26* (till 4th Sem)

    --
    **12th Grade (PSEB)**  
    Govt. Girls Senior Secondary School,2023  
    **Score**: 90.4%

    --
    **10th Grade (PSEB)**  
    Nav Bharat High School, 2021  
    **Score**: 100%

    """)

    st.subheader("Skills & Achievements")

    # Programming Languages
    st.markdown("** Languages:** C, C++, Python")

    # Domains 
    st.markdown("**Domains:** IoT, Machine Learning & Artificial Intelligence")

    # Tools & Platforms
    st.markdown("**Tools:** Streamlit, GitHub, Git")

    # Other Achievements
    st.markdown(""" 
    **🌟 Others:**  
    - Performed **Giddha** at the **Jashan Fest** in Guru Nanak Dev University  
    - Represented at **State Level** cultural event and secured **3rd Prize**
    """)
    
    st.markdown("""
    ---

    ### Thank You for Visiting the App!

    We appreciate your time and interest in this project.  
    This app was created with the goal of spreading awareness about chronic illnesses like **heart disease** and **diabetes** using the power of **machine learning**.

    If you found this tool useful or insightful, feel free to explore my work on [GitHub](https://github.com/gurpreet2007) and connect for feedback or collaboration!

    ---

    ### ⚠️ Health & Safety Note:

    > 🩺 **Important:** This application is intended solely for **educational and informational purposes**.  
    > It is **not a substitute** for professional medical advice, diagnosis, or treatment.  
    >  
    > If you experience any symptoms or receive a positive prediction from this app, please **consult a certified medical professional immediately** for proper guidance and care.

    Stay safe. Stay healthy. 💙

    ---
    """)

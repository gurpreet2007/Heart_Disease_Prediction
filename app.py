import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", [
    "Home",
    "Predict Heart Disease",
    "Predict Diabetes",
    "Predict Kidney Disease",
    "About Project"
])


# -------------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------------
if option == "Home":
    st.title("🩺 Multiple Disease Prediction App")

    st.markdown("""
    Welcome to the **Multiple Disease Prediction App**!

    This app uses **Machine Learning** to predict your risk of:

    - 🫀 Heart Disease  
    - 💉 Diabetes  
    - 🧫 Chronic Kidney Disease (CKD)
    """)

    # Image columns (3 cols now)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRHULKB1zMRoo735wOxbjz64fYdlvoTA4Usmg&s",
            caption="Heart Disease", width=200)

    with col2:
        st.image(
            "https://f.hubspotusercontent30.net/hubfs/2027031/diabetes.jpeg",
            caption="Diabetes", width=200)

    with col3:
        st.image("https://www.niddk.nih.gov/-/media/Images/Health-Information/Diabetes/landing/kidney.png?h=300&iar=0&w=400&hash=047A290E886DA372DED3CAE293A2AFBA",caption = "Kidney disease",width=200)

    st.markdown("""
    ### How It Works:
    1. Choose a disease prediction model from sidebar  
    2. Enter patient medical details  
    3. Get instant machine-learning based prediction  

    ### ⚠️ Disclaimer:
    This tool is for **education & awareness only** — not medical advice.
    """)


# -------------------------------------------------------------
# HEART DISEASE PAGE
# -------------------------------------------------------------
elif option == "Predict Heart Disease":
    st.title("🫀 Heart Disease Prediction")

    model_path = "./saved_models/heart.joblib"

    try:
        loaded_heart_model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Heart model not found at {model_path} ({e})")
        st.stop()

    st.markdown("### Enter the patient's data:")

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Vessels Colored", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3])

    if st.button("Predict"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]])

        prediction = loaded_heart_model.predict(input_data)

        if prediction[0] == 0:
            st.success("✔ Person does NOT have heart disease.")
        else:
            st.error("❗ Person IS likely to have heart disease.")


# -------------------------------------------------------------
# DIABETES PAGE
# -------------------------------------------------------------
elif option == "Predict Diabetes":
    st.title("💉 Diabetes Prediction")

    model_path = "saved_models/diabetes_disease.joblib"
    scaler_path = "saved_models/diabetes_scaler.joblib"

    try:
        loaded_diabetes_model = joblib.load(model_path)
        diabetes_scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Diabetes model or scaler not found ({e})")
        st.stop()

    st.markdown("### Enter the patient's data:")

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 50, 300, 120)
    bp = st.number_input("Blood Pressure", 40, 150, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 18, 130, 30)

    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, bp, skin_thickness,
                                insulin, bmi, dpf, age]])

        input_scaled = diabetes_scaler.transform(input_data)

        result = loaded_diabetes_model.predict(input_scaled)[0]

        if result == 1:
            st.error("❗ High risk of Diabetes detected!")
        else:
            st.success("✔ No Diabetes risk detected.")

# -------------------------------------------------------------
# KIDNEY DISEASE PAGE
# -------------------------------------------------------------
elif option == "Predict Kidney Disease":
    st.title("🧫 Kidney Disease Prediction")

    model_path = "saved_models/kidney.joblib"
    scaler_path = "saved_models/kidney_scaler.joblib"

    try:
        kidney_model = joblib.load(model_path)
        kidney_scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading kidney model or scaler: {e}")
        st.stop()

    st.markdown("### Enter the patient's data:")

    # ----------------- 24 INPUT FIELDS -----------------
    age = st.number_input("Age", 1, 120, 48)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
    al = st.number_input("Albumin (0-5)", 0, 5, 1)
    su = st.number_input("Sugar (0-5)", 0, 5, 0)

    rbc = st.selectbox("Red Blood Cells (1=Normal, 0=Abnormal)", [1, 0], index=0)
    pc = st.selectbox("Pus Cell (1=Normal, 0=Abnormal)", [1, 0], index=0)
    pcc = st.selectbox("Pus Cell Clumps (0=No, 1=Yes)", [0, 1], index=0)
    ba = st.selectbox("Bacteria (0=No, 1=Yes)", [0, 1], index=0)

    bgr = st.number_input("Blood Glucose Random", 50, 500, 121)
    bu = st.number_input("Blood Urea", 1, 300, 36)
    sc = st.number_input("Serum Creatinine", 0.1, 15.0, 1.2)
    sod = st.number_input("Sodium", 50, 200, 135)
    pot = st.number_input("Potassium", 1.0, 10.0, 4.5)

    hemo = st.number_input("Hemoglobin", 5.0, 20.0, 15.4)
    pcv = st.number_input("Packed Cell Volume", 10, 60, 44)
    wc = st.number_input("White Blood Cell Count", 3000, 30000, 7800)
    rc = st.number_input("Red Blood Cell Count", 1.0, 10.0, 5.2)

    htn = st.selectbox("Hypertension (1=Yes, 0=No)", [1, 0], index=0)
    dm = st.selectbox("Diabetes Mellitus (1=Yes, 0=No)", [0, 1], index=0)
    cad = st.selectbox("Coronary Artery Disease (1=Yes, 0=No)", [0, 1], index=0)
    appet = st.selectbox("Appetite (Good=1, Poor=0)", [1, 0], index=0)
    pe = st.selectbox("Pedal Edema (1=Yes, 0=No)", [1, 0], index=0)
    ane = st.selectbox("Anemia (1=Yes, 0=No)", [0, 1], index=0)

    # ----------------- PREDICTION ----------------------
    if st.button("Predict Kidney Disease"):

        # Feature order
        feature_list = [
            'age', 'bp', 'sg', 'al', 'su',
            'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot',
            'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]

        # Numeric columns used during training for scaling
        num_cols = [
            'age','bp','sg','al','su',
            'bgr','bu','sc','sod','pot',
            'hemo','pcv','wc','rc'
        ]

        # Create dataframe
        input_df = pd.DataFrame([[age, bp, sg, al, su, rbc, pc, pcc, ba,
                                bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                                htn, dm, cad, appet, pe, ane]],
                                columns=feature_list)

        # Scale ONLY numeric columns
        input_df[num_cols] = kidney_scaler.transform(input_df[num_cols])

        # Predict
        pred = kidney_model.predict(input_df)[0]

        if pred == 1:
            st.error("❗ Patient is likely to have Chronic Kidney Disease (CKD).")
        else:
            st.success("✔ No Chronic Kidney Disease detected.")
# -------------------------------------------------------------
# ABOUT PROJECT PAGE
# -------------------------------------------------------------
elif option == "About Project":
    
    st.title("🩺 Multiple Disease Prediction using Machine Learning")
    st.markdown("---")

    # Problem Statement
    st.subheader("🧩 Problem Statement")
    st.write("""
    Early detection of chronic diseases such as **heart disease**, **diabetes**, and **chronic kidney disease (CKD)** 
    can significantly improve treatment outcomes and reduce health risks.
    However, many patients do not have easy access to predictive tools that assist in identifying risks based on simple medical data.
    """)

    # Solution
    st.subheader("💡 Solution")   
    st.write("""
    This project provides a **Streamlit-based web application** that uses **machine learning models** to predict whether a person 
    is likely to have **heart disease**, **diabetes**, or **kidney disease**, based on input features such as glucose level, 
    blood pressure, age, creatinine level, cholesterol, and other key health indicators.
    """)

    # Project Overview
    st.subheader("📝 Project Overview")
    st.write("""
    This ML-powered application allows users to input medical parameters and get instant predictions for:

    - 🫀 **Heart Disease Prediction**
    - 💉 **Diabetes Prediction**
    - 🧫 **Kidney Disease Prediction**

    Separate models are trained for each disease using real-world datasets to make predictions on user input.
    """)

    # Target Variables
    st.subheader("🎯 Target Variables")
    st.markdown("""
    - **Heart Disease Prediction:** `target` (1 = disease, 0 = no disease)  
    - **Diabetes Prediction:** `Outcome` (1 = diabetes, 0 = no diabetes)  
    - **Kidney Disease Prediction:** `classification` (1 = CKD, 0 = Not CKD)  
    """)

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

    with st.expander("Kidney Disease Features"):
        st.markdown("""
        - Age  
        - Blood Pressure (`bp`)  
        - Specific Gravity (`sg`)  
        - Albumin  
        - Sugar  
        - Red Blood Cells  
        - Pus Cell  
        - Pus Cell Clumps  
        - Bacteria  
        - Blood Glucose Random (`bgr`)  
        - Blood Urea (`bu`)  
        - Serum Creatinine (`sc`)  
        - Sodium  
        - Potassium  
        - Hemoglobin  
        - Packed Cell Volume  
        - White Blood Cell Count (`wc`)  
        - Red Blood Cell Count (`rc`)  
        - Hypertension  
        - Diabetes Mellitus  
        - Coronary Artery Disease  
        - Appetite  
        - Pedal Edema  
        - Anemia  
        """)

    # Steps Performed
    st.subheader("Steps Performed")
    st.markdown("""
    1. **Data Collection** : Datasets sourced from **Kaggle**  
    2. **Data Cleaning** (handling missing values, encoding, selecting important features)  
    3. **Exploratory Data Analysis (EDA)**  
    4. **Feature Selection**  
    5. **Model Training** with Logistic Regression, Random Forest, XGBoost, SVM, Decision Tree, Catboost, LightGBM  
    6. **Model Evaluation** (accuracy, precision, recall, F1-score)  
    7. **Building Web App** using Streamlit  
    8. **Model Deployment** using `joblib` to save and load trained models  
    """)

    st.subheader(" Deployment")
    st.write("""
    This machine learning web application is deployed live on the cloud using **Render**.  
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
    **Guru Nanak Dev University**  
    CGPA: *8.26* (till 4th Sem)

    --
    **12th Grade (PSEB)**  
    Govt. Girls Senior Secondary School, 2023  
    **Score**: 90.4%

    --
    **10th Grade (PSEB)**  
    Nav Bharat High School, 2021  
    **Score**: 100%
    """)

    st.subheader("Skills & Achievements")

    st.markdown("**Languages:** C, C++, Python")
    st.markdown("**Domains:** IoT, Machine Learning & Artificial Intelligence")
    st.markdown("**Tools:** Streamlit, GitHub, Git")

    st.markdown(""" 
    **🌟 Others:**  
    - Performed **Giddha** at the **Jashan Fest** in Guru Nanak Dev University  
    - Represented at **State Level** cultural event and secured **3rd Prize**
    """)

    st.markdown("""
    ---
    ### Thank You for Visiting the App!

    We appreciate your time and interest in this project.  
    This app was created to spread awareness about chronic illnesses using the power of **machine learning**.

    If you found this tool useful, feel free to explore my work on [GitHub](https://github.com/gurpreet2007)!
    ---

    ### ⚠️ Health & Safety Note:
    > 🩺 **Important:** This application is intended for **educational and informational purposes only**.  
    > It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
    > Please consult a certified doctor for medical concerns.
    """)

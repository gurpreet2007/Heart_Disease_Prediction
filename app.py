import streamlit as st
import pandas as pd
import numpy as np
import joblib
@st.cache_resource
def load_heart_model():

    model = joblib.load("saved_models/heart.joblib")

    return model
def load_diabetes_model():

    model = joblib.load("saved_models/diabetes_disease.joblib")

    scaler = joblib.load("saved_models/diabetes_scaler.joblib")

    return model, scaler
@st.cache_resource
def load_kidney_model():

    model = joblib.load("saved_models/kidney.joblib")

    scaler = joblib.load("saved_models/kidney_scaler.joblib")

    return model, scaler

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard")

option = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Heart Disease Prediction",
        "Diabetes Prediction",
        "Kidney Disease Prediction",
        "Model Analytics",
        "About Project"
    ]
)

# =========================================================
# HOME PAGE
# =========================================================

if option == "Home":

    st.title("Multiple Disease Prediction Dashboard")

    st.markdown("""
    This web application uses Machine Learning models to predict the likelihood of:

    - Heart Disease
    - Diabetes
    - Chronic Kidney Disease (CKD)

    based on patient medical information.
    """)

    st.markdown("---")

    # =====================================================
    # DISEASE CARDS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.image(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRHULKB1zMRoo735wOxbjz64fYdlvoTA4Usmg&s"
        )

        st.subheader("Heart Disease")

        st.write("""
        Predicts cardiovascular disease risk
        using parameters such as cholesterol,
        blood pressure, chest pain type,
        and heart rate.
        """)

    with col2:

        st.image(
            "https://f.hubspotusercontent30.net/hubfs/2027031/diabetes.jpeg"
        )

        st.subheader("Diabetes")

        st.write("""
        Detects diabetes risk using
        glucose level, BMI, insulin,
        age, and related health factors.
        """)

    with col3:

        st.image(
            "https://www.niddk.nih.gov/-/media/Images/Health-Information/Diabetes/landing/kidney.png"
        )

        st.subheader("Kidney Disease")

        st.write("""
        Predicts Chronic Kidney Disease (CKD)
        using clinical parameters such as
        creatinine, hemoglobin, and albumin.
        """)

    st.markdown("---")

    # =====================================================
    # HOW IT WORKS
    # =====================================================

    st.subheader("How It Works")

    st.markdown("""
    1. Select a disease prediction model from the sidebar  
    2. Enter patient medical details  
    3. The trained Machine Learning model analyzes the input  
    4. The system generates an instant prediction result  
    """)

    st.markdown("---")

    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.warning("""
    Disclaimer:
    This application is intended for educational and informational purposes only.
    It is not a substitute for professional medical advice or diagnosis.
    """)
# =========================================================
# HEART DISEASE PREDICTION
# =========================================================

elif option == "Heart Disease Prediction":

    st.title("Heart Disease Prediction")
    # LOAD CACHED MODEL
    model = load_heart_model()


    st.markdown("""
    Enter the patient's medical information below
    to predict the likelihood of heart disease.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # =====================================================
    # LEFT COLUMN
    # =====================================================

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

        sex = st.selectbox(
            "Sex",
            [0, 1],
            help="""
            0 = Female
            1 = Male
            """
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            help="""
            0 = Typical Angina
            
            1 = Atypical Angina
            
            2 = Non-anginal Pain
            
            3 = Asymptomatic
            """
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=80,
            max_value=200,
            value=120
        )

        chol = st.number_input(
            "Cholesterol Level",
            min_value=100,
            max_value=600,
            value=200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            help="""
            0 = False
            
            1 = True
            """
        )

        restecg = st.selectbox(
            "Resting ECG Results",
            [0, 1, 2],
            help="""
            0 = Normal
            
            1 = ST-T Wave Abnormality
            
            2 = Left Ventricular Hypertrophy
            """
        )

    # =====================================================
    # RIGHT COLUMN
    # =====================================================

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=70,
            max_value=220,
            value=150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1],
            help="""
            0 = No
            
            1 = Yes
            """
        )

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "ST Segment Slope",
            [0, 1, 2],
            help="""
            0 = Upsloping
            
            1 = Flat
            
            2 = Downsloping
            """
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            [0, 1, 2, 3],
            help="""
            Number of major vessels colored by fluoroscopy
            """
        )

        thal = st.selectbox(
            "Thalassemia",
            [1, 2, 3],
            help="""
            1 = Normal
            
            2 = Fixed Defect
            
            3 = Reversible Defect
            """
        )

    st.markdown("---")

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if st.button("Predict Heart Disease"):

        input_data = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]])

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        st.metric(
            "Heart Disease Risk",
            f"{probability * 100:.2f}%"
        )

        if probability >= 0.7:

            st.error("""
            High risk of Heart Disease detected.
            Please consult a healthcare professional.
            """)

        elif probability >= 0.4:

            st.warning("""
            Moderate risk detected.
            Regular health monitoring is recommended.
            """)

        else:

            st.success("""
            Low risk of Heart Disease detected.
            """)

        st.markdown("---")

        # =================================================
        # GENERAL RECOMMENDATIONS
        # =================================================

        st.subheader("General Recommendations")

        st.write("""
        - Maintain a healthy diet
        
        - Exercise regularly
        
        - Monitor blood pressure and cholesterol
        
        - Avoid smoking and excessive alcohol consumption
        
        - Consult a doctor for proper medical advice
        """)
# =========================================================
# DIABETES
# =========================================================

elif option == "Diabetes Prediction":
    
    st.title("Diabetes Prediction")

    # LOAD CACHED MODEL
    model, scaler = load_diabetes_model()

    st.markdown("""
    Enter the patient's medical information below
    to predict the likelihood of diabetes.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # =====================================================
    # LEFT COLUMN
    # =====================================================

    with col1:

        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            help="Number of times the patient has been pregnant"
        )

        glucose = st.number_input(
            "Glucose Level",
            min_value=50,
            max_value=300,
            value=120,
            help="Plasma glucose concentration"
        )

        bp = st.number_input(
            "Blood Pressure",
            min_value=40,
            max_value=150,
            value=70,
            help="Diastolic blood pressure (mm Hg)"
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            max_value=100,
            value=20,
            help="Triceps skin fold thickness (mm)"
        )

    # =====================================================
    # RIGHT COLUMN
    # =====================================================

    with col2:

        insulin = st.number_input(
            "Insulin Level",
            min_value=0,
            max_value=900,
            value=80,
            help="2-Hour serum insulin level"
        )

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Body Mass Index = weight(kg) / height(m²)"
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
            help="Indicates hereditary diabetes risk"
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=30
        )

    st.markdown("---")

    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button("Predict Diabetes"):

        input_data = np.array([[
            pregnancies,
            glucose,
            bp,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Prediction Result")

        st.metric(
            "Diabetes Risk",
            f"{probability * 100:.2f}%"
        )

        if probability >= 0.7:

            st.error("""
            High risk of Diabetes detected.
            Please consult a healthcare professional.
            """)

        elif probability >= 0.4:

            st.warning("""
            Moderate risk detected.
            Regular monitoring is recommended.
            """)

        else:

            st.success("""
            Low risk of Diabetes detected.
            """)

        st.markdown("---")

        # =================================================
        # RECOMMENDATIONS
        # =================================================

        st.subheader("General Recommendations")

        st.write("""
        - Maintain a healthy and balanced diet
        
        - Exercise regularly
        
        - Monitor blood sugar levels
        
        - Reduce excessive sugar intake
        
        - Maintain healthy body weight
        
        - Consult a healthcare professional for proper medical advice
        """)# KIDNEY DISEASE
# =========================================================

# =========================================================
# KIDNEY DISEASE PREDICTION
# =========================================================

elif option == "Kidney Disease Prediction":

    st.title("Kidney Disease Prediction")

    # LOAD CACHED MODEL
    kidney_model, kidney_scaler = load_kidney_model()

    st.markdown("""
    Enter the patient's medical information below
    to predict the likelihood of Chronic Kidney Disease (CKD).
    """)

    st.markdown("---")

    # =====================================================
    # COLUMN LAYOUT
    # =====================================================

    col1, col2, col3 = st.columns(3)

    # =====================================================
    # COLUMN 1
    # =====================================================

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=48
        )

        bp = st.number_input(
            "Blood Pressure",
            min_value=50,
            max_value=200,
            value=80,
            help="Blood pressure measured in mm/Hg"
        )

        sg = st.selectbox(
            "Specific Gravity",
            [1.005, 1.010, 1.015, 1.020, 1.025],
            index=3,
            help="Concentration level of urine"
        )

        al = st.number_input(
            "Albumin",
            min_value=0,
            max_value=5,
            value=1,
            help="Albumin protein level in urine"
        )

        su = st.number_input(
            "Sugar",
            min_value=0,
            max_value=5,
            value=0,
            help="Sugar level in urine"
        )

        rbc = st.selectbox(
            "Red Blood Cells",
            [1, 0],
            index=0,
            help="""
            1 = Normal
            
            0 = Abnormal
            """
        )

        pc = st.selectbox(
            "Pus Cell",
            [1, 0],
            index=0,
            help="""
            1 = Normal
            
            0 = Abnormal
            """
        )

        pcc = st.selectbox(
            "Pus Cell Clumps",
            [0, 1],
            index=0,
            help="""
            0 = No
            
            1 = Yes
            """
        )

    # =====================================================
    # COLUMN 2
    # =====================================================

    with col2:

        ba = st.selectbox(
            "Bacteria",
            [0, 1],
            index=0,
            help="""
            0 = No
            
            1 = Yes
            """
        )

        bgr = st.number_input(
            "Blood Glucose Random",
            min_value=50,
            max_value=500,
            value=121
        )

        bu = st.number_input(
            "Blood Urea",
            min_value=1,
            max_value=300,
            value=36,
            help="Amount of urea nitrogen in blood"
        )

        sc = st.number_input(
            "Serum Creatinine",
            min_value=0.1,
            max_value=15.0,
            value=1.2,
            step=0.1,
            help="Waste product level filtered by kidneys"
        )

        sod = st.number_input(
            "Sodium",
            min_value=50,
            max_value=200,
            value=135
        )

        pot = st.number_input(
            "Potassium",
            min_value=1.0,
            max_value=10.0,
            value=4.5,
            step=0.1
        )

        hemo = st.number_input(
            "Hemoglobin",
            min_value=5.0,
            max_value=20.0,
            value=15.4,
            step=0.1
        )

        pcv = st.number_input(
            "Packed Cell Volume",
            min_value=10,
            max_value=60,
            value=44
        )

    # =====================================================
    # COLUMN 3
    # =====================================================

    with col3:

        wc = st.number_input(
            "White Blood Cell Count",
            min_value=3000,
            max_value=30000,
            value=7800
        )

        rc = st.number_input(
            "Red Blood Cell Count",
            min_value=1.0,
            max_value=10.0,
            value=5.2,
            step=0.1
        )

        htn = st.selectbox(
            "Hypertension",
            [1, 0],
            index=0,
            help="""
            1 = Yes
            
            0 = No
            """
        )

        dm = st.selectbox(
            "Diabetes Mellitus",
            [1, 0],
            index=1,
            help="""
            1 = Yes
            
            0 = No
            """
        )

        cad = st.selectbox(
            "Coronary Artery Disease",
            [1, 0],
            index=1,
            help="""
            1 = Yes
            
            0 = No
            """
        )

        appet = st.selectbox(
            "Appetite",
            [1, 0],
            index=0,
            help="""
            1 = Good
            
            0 = Poor
            """
        )

        pe = st.selectbox(
            "Pedal Edema",
            [1, 0],
            index=1,
            help="""
            1 = Yes
            
            0 = No
            """
        )

        ane = st.selectbox(
            "Anemia",
            [1, 0],
            index=1,
            help="""
            1 = Yes
            
            0 = No
            """
        )

    st.markdown("---")

    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button("Predict Kidney Disease"):

        # FEATURE ORDER
        feature_list = [
            'age', 'bp', 'sg', 'al', 'su',
            'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot',
            'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet',
            'pe', 'ane'
        ]

        # NUMERIC COLUMNS
        num_cols = [
            'age', 'bp', 'sg', 'al', 'su',
            'bgr', 'bu', 'sc', 'sod', 'pot',
            'hemo', 'pcv', 'wc', 'rc'
        ]

        # CREATE DATAFRAME
        input_df = pd.DataFrame([[
            age, bp, sg, al, su,
            rbc, pc, pcc, ba,
            bgr, bu, sc, sod, pot,
            hemo, pcv, wc, rc,
            htn, dm, cad, appet,
            pe, ane
        ]], columns=feature_list)

        # SCALE NUMERIC FEATURES
        input_df[num_cols] = kidney_scaler.transform(
            input_df[num_cols]
        )

        # PREDICTION
        pred = kidney_model.predict(input_df)[0]

        probability = kidney_model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        st.metric(
            "Kidney Disease Risk",
            f"{probability * 100:.2f}%"
        )

        if probability >= 0.7:

            st.error("""
            High risk of Chronic Kidney Disease detected.
            Please consult a healthcare professional.
            """)

        elif probability >= 0.4:

            st.warning("""
            Moderate risk detected.
            Regular medical monitoring is recommended.
            """)

        else:

            st.success("""
            Low risk of Chronic Kidney Disease detected.
            """)

        st.markdown("---")

        # =================================================
        # RECOMMENDATIONS
        # =================================================

        st.subheader("General Recommendations")

        st.write("""
        - Drink sufficient water regularly
        
        - Monitor blood pressure and blood sugar
        
        - Reduce excessive salt intake
        
        - Avoid smoking and alcohol
        
        - Maintain healthy diet and exercise
        
        - Consult a healthcare professional for proper medical advice
        """)
# =========================================================
# MODEL ANALYTICS
# =========================================================

elif option == "Model Analytics":

    st.title("Model Analytics Dashboard")

    st.markdown("""
    This section presents model evaluation
    and visualization results for all disease prediction models.
    """)

    st.markdown("---")

    analytics_option = st.selectbox(
        "Select Disease Analytics",
        [
            "Heart Disease",
            "Diabetes",
            "Kidney Disease"
        ]
    )

    st.markdown("---")

    # =====================================================
    # HEART DISEASE ANALYTICS
    # =====================================================

    if analytics_option == "Heart Disease":

        st.header("Heart Disease Analytics")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Model Comparison")

            st.image(
                "images/heart_model_comparison.png",
                use_container_width=True
            )

        with col2:

            st.subheader("Confusion Matrix")

            st.image(
                "images/heart_confusion_matrix.png",
                use_container_width=True
            )

        st.subheader("Correlation Heatmap")

        st.image(
            "images/heart_heatmap.png",
            use_container_width=True
        )

        st.subheader("Feature Importance")

        st.image(
            "images/heart_feature_imp.png",
            use_container_width=True
        )

    # =====================================================
    # DIABETES ANALYTICS
    # =====================================================

    elif analytics_option == "Diabetes":

        st.header("Diabetes Analytics")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Model Comparison")

            st.image(
                "images/diabetes_model_comparison.png",
                use_container_width=True
            )

        with col2:

            st.subheader("Confusion Matrix")

            st.image(
                "images/diabetes_confusion_matrix.png",
                use_container_width=True
            )

        st.subheader("Feature Importance")

        st.image(
            "images/diabetes_feature_imp.png",
            use_container_width=True
        )

        st.subheader("Correlation Heatmap")

        st.image(
            "images/diabetes_heatmap.png",
            use_container_width=True
        )

    # =====================================================
    # KIDNEY DISEASE ANALYTICS
    # =====================================================

    elif analytics_option == "Kidney Disease":

        st.header("Kidney Disease Analytics")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Model Comparison")

            st.image(
                "images/kidney_model_comparison.png",
                use_container_width=True
            )

        with col2:

            st.subheader("Confusion Matrix")

            st.image(
                "images/kidney_confusion_matrix.png",
                use_container_width=True
            )

        st.subheader("Feature Importance")

        st.image(
            "images/kidney_feature_imp.png",
            use_container_width=True
        )

        st.subheader("Correlation Heatmap")

        st.image(
            "images/kidney_heatmap.png",
            use_container_width=True
        )

# ABOUT PROJECT
# =========================================================

elif option == "About Project":

    st.title("About Project")

    st.markdown("---")

    # =====================================================
    # PROJECT OVERVIEW
    # =====================================================

    st.header("Project Overview")

    st.write("""
    The Multiple Disease Prediction System is a machine learning-based web application
    designed to predict the risk of three major chronic diseases:

    - Heart Disease  
    - Diabetes  
    - Chronic Kidney Disease  

    The system uses trained ML models and patient medical input data to provide
    real-time predictions in an interactive dashboard.
    """)

    # =====================================================
    # OBJECTIVE
    # =====================================================

    st.header("Project Objective")

    st.write("""
    The main objective of this project is to assist in early detection of diseases
    using machine learning techniques and provide an easy-to-use interface for
    health risk prediction.
    """)

    # =====================================================
    # TECHNOLOGIES USED
    # =====================================================

    st.header("Technologies Used")

    st.write("""
    - Python  
    - Streamlit  
    - Scikit-learn  
    - XGBoost  
    - CatBoost  
    - Pandas  
    - NumPy  
    - Matplotlib  
    - Seaborn  
    - Joblib  
    - Git & GitHub  
    - Render (Deployment)  
    """)

    # =====================================================
    # SELECTED MODELS
    # =====================================================

    st.header("Selected Machine Learning Models")

    st.write("""
    - Heart Disease: Logistic Regression  
    - Diabetes: Stacking Classifier  
    - Kidney Disease: Tuned Random Forest  
    """)

    # =====================================================
    # MACHINE LEARNING PIPELINE
    # =====================================================

    st.header("Machine Learning Pipeline")

    st.write("""
    1. Data Collection  
    2. Data Preprocessing  
    3. Exploratory Data Analysis (EDA)  
    4. Feature Selection  
    5. Model Training  
    6. Model Evaluation  
    7. Model Deployment using Streamlit  
    """)

    # =====================================================
    # KEY FEATURES
    # =====================================================

    st.header("Key Features")

    st.write("""
    - Real-time disease prediction  
    - Interactive user-friendly dashboard  
    - Multiple disease prediction system  
    - Visual model analytics section  
    - Cloud deployment using Render  
    """)

    # =====================================================
    # DEVELOPER
    # =====================================================

    st.header("Developer")

    st.write("""
    Gurpreet Kaur  
    B.Tech Electronics and Computer Engineering  
    Guru Nanak Dev University  
    """)

    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.header("Disclaimer")

    st.warning("""
    This application is developed for educational and learning purposes only.

    It should not be used as a substitute for professional medical diagnosis or treatment.
    """)
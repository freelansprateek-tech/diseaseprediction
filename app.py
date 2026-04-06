import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json

st.set_page_config(
    page_title="Multi-Disease Predictorr",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model(disease):
    model_path = f'models/{disease}_model.pkl'
    scaler_path = f'models/{disease}_scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

@st.cache_data
def load_metadata():
    if os.path.exists('models/model_metadata.json'):
        with open('models/model_metadata.json', 'r') as f:
            return json.load(f)
    return {}

def predict_disease(disease, features):
    model, scaler = load_model(disease)
    
    if model is None or scaler is None:
        return None
    
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        prob_neg = float(proba[0])
        prob_pos = float(proba[1])
    else:
        prob_neg = 0.5
        prob_pos = 0.5
    
    confidence = max(prob_neg, prob_pos)
    
    if prob_pos < 0.3:
        risk = 'Low'
        risk_color = 'green'
    elif prob_pos < 0.7:
        risk = 'Medium'
        risk_color = 'orange'
    else:
        risk = 'High'
        risk_color = 'red'
    
    metadata = load_metadata()
    model_name = metadata.get(disease, {}).get('best_model', 'Unknown')
    
    return {
        'prediction': int(prediction),
        'prediction_label': 'Positive' if prediction == 1 else 'Negative',
        'prob_neg': prob_neg,
        'prob_pos': prob_pos,
        'risk_level': risk,
        'risk_color': risk_color,
        'model_used': model_name,
        'confidence': confidence
    }

if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0

if 'last_predictions' not in st.session_state:
    st.session_state.last_predictions = []

st.sidebar.title("🏥 Navi")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🩺 Diabetes Prediction", "❤️ Heart Disease Prediction", 
     "🧠 Parkinson's Prediction", "📊 Model Performance", "ℹ️ About"]
)

if page == "🏠 Home":
    st.title("🏥 Multi-Disease Predictionn System")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the AI-Powered Disease Prediction Platform
    
    This application uses advanced Machine Learning models to predict three major diseases:
    - **Diabetes** - Blood sugar level disorder
    - **Heart Disease** - Cardiovascular conditions
    - **Parkinson's Disease** - Neurological movement disorder
    
    #### 🎯 How It Works
    1. Select a disease prediction page from the sidebar
    2. Enter the required medical parameters
    3. Click "Predict" to get instant results
    4. View detailed probability scores and risk levels
    """)
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metadata = load_metadata()
    
    with col1:
        st.metric("Total Models Trained", "33", "11 per disease")
    
    with col2:
        if 'diabetes' in metadata:
            acc = metadata['diabetes']['accuracy']
            st.metric("Diabetes Model", metadata['diabetes']['best_model'], f"{acc:.1%}")
    
    with col3:
        if 'heart' in metadata:
            acc = metadata['heart']['accuracy']
            st.metric("Heart Model", metadata['heart']['best_model'], f"{acc:.1%}")
    
    with col4:
        if 'parkinsons' in metadata:
            acc = metadata['parkinsons']['accuracy']
            st.metric("Parkinson's Model", metadata['parkinsons']['best_model'], f"{acc:.1%}")
    
    st.markdown("---")
    
    st.info(f"📈 Total Predictions Made: **{st.session_state.prediction_count}**")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🩺 Diabetes")
        st.write("A metabolic disorder affecting blood sugar regulation. Early detection helps prevent complications.")
    
    with col2:
        st.markdown("### ❤️ Heart Disease")
        st.write("Cardiovascular conditions affecting heart function. Leading cause of death worldwide.")
    
    with col3:
        st.markdown("### 🧠 Parkinson's")
        st.write("Progressive neurological disorder affecting movement and coordination.")

elif page == "🩺 Diabetes Prediction":
    st.title("🩺 Diabetes Prediction")
    st.markdown("Enter patient information to predict diabetes risk")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, help="Number of times pregnant")
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=200, value=120, help="Plasma glucose concentration")
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=140, value=70, help="Diastolic blood pressure")
        skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness")
    
    with col2:
        insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=80, help="2-Hour serum insulin")
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1, help="Body Mass Index")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01, help="Diabetes heredity score")
        age = st.number_input("Age", min_value=21, max_value=100, value=30, help="Age in years")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_btn = st.button("🔍 Predict Diabetes", use_container_width=True, type="primary")
    
    if predict_btn:
        features = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        result = predict_disease('diabetes', features)
        
        if result:
            st.session_state.prediction_count += 1
            st.session_state.last_predictions.append({
                'disease': 'Diabetes',
                'result': result['prediction_label'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.markdown("---")
            st.subheader("📋 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result['prediction'] == 1:
                    st.error(f"### ⚠️ {result['prediction_label']}")
                else:
                    st.success(f"### ✅ {result['prediction_label']}")
            
            with col2:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            
            with col3:
                if result['risk_color'] == 'red':
                    st.error(f"### Risk: {result['risk_level']}")
                elif result['risk_color'] == 'orange':
                    st.warning(f"### Risk: {result['risk_level']}")
                else:
                    st.success(f"### Risk: {result['risk_level']}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Probability Scores")
                st.progress(result['prob_neg'], text=f"Negative: {result['prob_neg']*100:.1f}%")
                st.progress(result['prob_pos'], text=f"Positive: {result['prob_pos']*100:.1f}%")
            
            with col2:
                st.markdown("#### 🤖 Model Information")
                st.info(f"**Model Used:** {result['model_used']}")
                st.info(f"**Prediction Time:** {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.error("Model not loaded. Please train the model first.")

elif page == "❤️ Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction")
    st.markdown("Enter patient information to predict heart disease risk")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=29, max_value=80, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 1 if sex == "Male" else 0
        
        cp = st.selectbox("Chest Pain Type", 
                         options=[0, 1, 2, 3],
                         format_func=lambda x: f"Type {x} - {['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x]}")
        cp_val = cp
        
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs_val = 1 if fbs == "Yes" else 0
        
        restecg = st.selectbox("Resting ECG", 
                              options=[0, 1, 2],
                              format_func=lambda x: f"{x} - {['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][x]}")
        restecg_val = restecg
        
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exang_val = 1 if exang == "Yes" else 0
    
    with col3:
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        
        slope = st.selectbox("Slope of Peak Exercise ST", 
                           options=[0, 1, 2],
                           format_func=lambda x: f"{x} - {['Upsloping', 'Flat', 'Downsloping'][x]}")
        slope_val = slope
        
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        
        thal = st.selectbox("Thalassemia", 
                          options=[0, 1, 2, 3],
                          format_func=lambda x: f"{x} - {['Normal', 'Fixed Defect', 'Reversible Defect', 'Not Described'][x]}")
        thal_val = thal
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_btn = st.button("🔍 Predict Heart Disease", use_container_width=True, type="primary")
    
    if predict_btn:
        features = [age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, 
                   thalach, exang_val, oldpeak, slope_val, ca, thal_val]
        result = predict_disease('heart', features)
        
        if result:
            st.session_state.prediction_count += 1
            st.session_state.last_predictions.append({
                'disease': 'Heart Disease',
                'result': result['prediction_label'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.markdown("---")
            st.subheader("📋 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result['prediction'] == 1:
                    st.error(f"### ⚠️ {result['prediction_label']}")
                else:
                    st.success(f"### ✅ {result['prediction_label']}")
            
            with col2:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            
            with col3:
                if result['risk_color'] == 'red':
                    st.error(f"### Risk: {result['risk_level']}")
                elif result['risk_color'] == 'orange':
                    st.warning(f"### Risk: {result['risk_level']}")
                else:
                    st.success(f"### Risk: {result['risk_level']}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Probability Scores")
                st.progress(result['prob_neg'], text=f"Negative: {result['prob_neg']*100:.1f}%")
                st.progress(result['prob_pos'], text=f"Positive: {result['prob_pos']*100:.1f}%")
            
            with col2:
                st.markdown("#### 🤖 Model Information")
                st.info(f"**Model Used:** {result['model_used']}")
                st.info(f"**Prediction Time:** {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.error("Model not loaded. Please train the model first.")

elif page == "🧠 Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction")
    st.markdown("Enter vocal features to predict Parkinson's disease")
    st.markdown("---")
    
    st.info("💡 These are vocal measurements. You can use sample data or enter custom values.")
    
    if st.button("Load Sample Data"):
        st.session_state.sample_loaded = True
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Frequency Features")
        mdvp_fo = st.number_input("MDVP:Fo(Hz)", min_value=80.0, max_value=300.0, value=120.0, 
                                 help="Average vocal fundamental frequency")
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", min_value=100.0, max_value=600.0, value=157.0,
                                  help="Maximum vocal fundamental frequency")
        mdvp_flo = st.number_input("MDVP:Flo(Hz)", min_value=60.0, max_value=250.0, value=75.0,
                                  help="Minimum vocal fundamental frequency")
        
        st.markdown("#### Jitter Features")
        mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=0.05, 
                                             value=0.007, step=0.001, format="%.5f")
        mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.0005, 
                                         value=0.00007, step=0.00001, format="%.5f")
        mdvp_rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=0.05, 
                                  value=0.004, step=0.001, format="%.5f")
        mdvp_ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=0.05, 
                                  value=0.005, step=0.001, format="%.5f")
        jitter_ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=0.15, 
                                    value=0.011, step=0.001, format="%.5f")
    
    with col2:
        st.markdown("#### Shimmer Features")
        mdvp_shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=0.2, 
                                      value=0.044, step=0.001, format="%.5f")
        mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=2.0, 
                                         value=0.426, step=0.01)
        shimmer_apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=0.1, 
                                      value=0.022, step=0.001, format="%.5f")
        shimmer_apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=0.1, 
                                      value=0.031, step=0.001, format="%.5f")
        mdvp_apq = st.number_input("MDVP:APQ", min_value=0.0, max_value=0.15, 
                                  value=0.030, step=0.001, format="%.5f")
        shimmer_dda = st.number_input("Shimmer:DDA", min_value=0.0, max_value=0.3, 
                                     value=0.065, step=0.001, format="%.5f")
        
        st.markdown("#### Harmonicity Features")
        nhr = st.number_input("NHR", min_value=0.0, max_value=0.5, 
                             value=0.022, step=0.001, format="%.5f")
        hnr = st.number_input("HNR", min_value=0.0, max_value=40.0, 
                             value=21.0, step=0.1)
    
    with col3:
        st.markdown("#### Nonlinear Features")
        rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, 
                              value=0.415, step=0.001, format="%.5f")
        dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, 
                             value=0.815, step=0.001, format="%.5f")
        spread1 = st.number_input("spread1", min_value=-10.0, max_value=0.0, 
                                 value=-4.8, step=0.1)
        spread2 = st.number_input("spread2", min_value=0.0, max_value=1.0, 
                                 value=0.27, step=0.01)
        d2 = st.number_input("D2", min_value=0.0, max_value=5.0, 
                            value=2.3, step=0.1)
        ppe = st.number_input("PPE", min_value=0.0, max_value=1.0, 
                             value=0.28, step=0.01)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_btn = st.button("🔍 Predict Parkinson's", use_container_width=True, type="primary")
    
    if predict_btn:
        features = [mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs,
                   mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
                   shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr,
                   rpde, dfa, spread1, spread2, d2, ppe]
        
        result = predict_disease('parkinsons', features)
        
        if result:
            st.session_state.prediction_count += 1
            st.session_state.last_predictions.append({
                'disease': "Parkinson's",
                'result': result['prediction_label'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.markdown("---")
            st.subheader("📋 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result['prediction'] == 1:
                    st.error(f"### ⚠️ {result['prediction_label']}")
                else:
                    st.success(f"### ✅ {result['prediction_label']}")
            
            with col2:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            
            with col3:
                if result['risk_color'] == 'red':
                    st.error(f"### Risk: {result['risk_level']}")
                elif result['risk_color'] == 'orange':
                    st.warning(f"### Risk: {result['risk_level']}")
                else:
                    st.success(f"### Risk: {result['risk_level']}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Probability Scores")
                st.progress(result['prob_neg'], text=f"Negative: {result['prob_neg']*100:.1f}%")
                st.progress(result['prob_pos'], text=f"Positive: {result['prob_pos']*100:.1f}%")
            
            with col2:
                st.markdown("#### 🤖 Model Information")
                st.info(f"**Model Used:** {result['model_used']}")
                st.info(f"**Prediction Time:** {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.error("Model not loaded. Please train the model first.")

elif page == "📊 Model Performance":
    st.title("📊 Model Performance Analysis")
    st.markdown("Comprehensive visualizations and metrics for all trained models")
    st.markdown("---")
    
    metadata = load_metadata()
    
    disease_tabs = st.tabs(["🩺 Diabetes", "❤️ Heart Disease", "🧠 Parkinson's"])
    
    diseases = ['diabetes', 'heart', 'parkinsons']
    disease_names = ['Diabetes', 'Heart Disease', "Parkinson's"]
    
    for idx, (tab, disease, disease_name) in enumerate(zip(disease_tabs, diseases, disease_names)):
        with tab:
            st.header(f"{disease_name} Model Analysis")
            
            if disease in metadata:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Best Model", metadata[disease]['best_model'])
                with col2:
                    st.metric("Accuracy", f"{metadata[disease]['accuracy']:.2%}")
                with col3:
                    st.metric("Precision", f"{metadata[disease]['precision']:.2%}")
                with col4:
                    st.metric("F1 Score", f"{metadata[disease]['f1']:.2%}")
                
                st.markdown("---")
                
                st.subheader("📈 Model Comparison")
                if os.path.exists(f'static/images/{disease}/model_comparison.png'):
                    st.image(f'static/images/{disease}/model_comparison.png', 
                            caption=f'{disease_name} - All Models Performance')
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Confusion Matrix")
                    if os.path.exists(f'static/images/{disease}/confusion_matrix.png'):
                        st.image(f'static/images/{disease}/confusion_matrix.png')
                
                with col2:
                    st.subheader("📉 ROC Curve")
                    if os.path.exists(f'static/images/{disease}/roc_curve.png'):
                        st.image(f'static/images/{disease}/roc_curve.png')
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔥 Correlation Heatmap")
                    if os.path.exists(f'static/images/{disease}/correlation_heatmap.png'):
                        st.image(f'static/images/{disease}/correlation_heatmap.png')
                
                with col2:
                    st.subheader("📊 Target Distribution")
                    if os.path.exists(f'static/images/{disease}/target_distribution.png'):
                        st.image(f'static/images/{disease}/target_distribution.png')
                
                st.markdown("---")
                
                st.subheader("⭐ Feature Importance")
                if os.path.exists(f'static/images/{disease}/feature_importance.png'):
                    st.image(f'static/images/{disease}/feature_importance.png',
                            caption=f'Top features for {disease_name} prediction')
                
                st.markdown("---")
                
                st.subheader("📋 All Models Performance")
                if 'all_models' in metadata[disease]:
                    models_df = pd.DataFrame(metadata[disease]['all_models']).T
                    models_df = models_df.sort_values('accuracy', ascending=False)
                    models_df = models_df.style.format({
                        'accuracy': '{:.2%}',
                        'precision': '{:.2%}',
                        'recall': '{:.2%}',
                        'f1': '{:.2%}',
                        'roc_auc': '{:.2%}'
                    }).background_gradient(cmap='RdYlGn', subset=['accuracy'])
                    
                    st.dataframe(models_df, use_container_width=True)
            else:
                st.warning(f"No metadata available for {disease_name}")

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This **Multi-Disease Prediction System** uses advanced Machine Learning algorithms to predict three major diseases:
    Diabetes, Heart Disease, and Parkinson's Disease. The system achieves high accuracy rates using state-of-the-art
    ensemble learning techniques.
    
    ### 📊 Dataset Information
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🩺 Diabetes Dataset**
        - **Samples:** 768
        - **Features:** 8
        - **Source:** PIMA Indians Diabetes Database
        - **Target:** Binary (0/1)
        """)
    
    with col2:
        st.markdown("""
        **❤️ Heart Disease Dataset**
        - **Samples:** 303
        - **Features:** 13
        - **Source:** Cleveland Heart Disease Database
        - **Target:** Binary (0/1)
        """)
    
    with col3:
        st.markdown("""
        **🧠 Parkinson's Dataset**
        - **Samples:** 195
        - **Features:** 22
        - **Source:** Oxford Parkinson's Disease Detection
        - **Target:** Binary (0/1)
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🤖 Machine Learning Models Used
    
    For each disease, we trained and evaluated 10 different algorithms:
    
    1. **Logistic Regression** - Linear classification
    2. **Decision Tree** - Rule-based learning
    3. **Random Forest** - Ensemble of decision trees
    4. **K-Nearest Neighbors (KNN)** - Instance-based learning
    5. **Naive Bayes** - Probabilistic classifier
    6. **Support Vector Machine (SVM)** - Margin-based classification
    7. **AdaBoost** - Adaptive boosting
    8. **Bagging** - Bootstrap aggregating
    9. **XGBoost** - Extreme gradient boosting
    10. **Voting Classifier** - Ensemble of multiple models
    11. **Stacking Classifier** - Meta-learning ensemble model
    
    ### 🔬 Methodology
    
    - **Data Preprocessing:** StandardScaler normalization
    - **Class Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique)
    - **Evaluation:** 5-fold cross-validation
    - **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
    - **Model Selection:** Best performing model saved for deployment
    
    ### 🛠️ Technology Stack
    
    - **Frontend:** Streamlit
    - **Backend:** Flask (API)
    - **ML Libraries:** scikit-learn, XGBoost, imbalanced-learn
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Matplotlib, Seaborn
    - **Deployment:** Railway / Streamlit Cloud
    
    ### 📈 Model Performance Summary
    """)
    
    metadata = load_metadata()
    
    if metadata:
        perf_data = []
        for disease in ['diabetes', 'heart', 'parkinsons']:
            if disease in metadata:
                perf_data.append({
                    'Disease': disease.capitalize(),
                    'Best Model': metadata[disease]['best_model'],
                    'Accuracy': f"{metadata[disease]['accuracy']:.2%}",
                    'Precision': f"{metadata[disease]['precision']:.2%}",
                    'Recall': f"{metadata[disease]['recall']:.2%}",
                    'F1-Score': f"{metadata[disease]['f1']:.2%}",
                    'ROC-AUC': f"{metadata[disease]['roc_auc']:.2%}"
                })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### ⚠️ Disclaimer
    
    This application is for **educational and research purposes only**. It should not be used as a substitute for 
    professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals 
    for medical concerns.
    
    ### 📚 References
    
    - Pima Indians Diabetes Database - UCI Machine Learning Repository
    - Cleveland Heart Disease Database - UCI Machine Learning Repository
    - Oxford Parkinson's Disease Detection Dataset
    - El-Sofany, H.F. (2024). "Predicting Heart Diseases Using Machine Learning and Different Data Classification Techniques"
    
    ### 👨‍💻 Developer
    
    **GitHub:** [Your Repository Link]
    
    **Contact:** your.email@example.com
    
    ### 📄 License
    
    This project is licensed under the MIT License.
    
    ---
    
    ### 🙏 Acknowledgments
    
    Special thanks to:
    - UCI Machine Learning Repository for datasets
    - Scikit-learn and XGBoost communities
    - Streamlit for the amazing framework
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📧 Report Issues")
    with col2:
        st.info("⭐ Star on GitHub")
    with col3:
        st.info("🤝 Contribute")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🩺 Additional Tools")

# Symptoms Checkup Button
st.sidebar.markdown("""
    <a href="https://health-checkup-rust.vercel.app/" target="_blank">
        <button style="
            background-color: #FF4B4B;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            font-weight: bold;
            margin-bottom: 10px;
        ">
            🔍 Symptoms Checkup
        </button>
    </a>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.metric("Total Predictions", st.session_state.prediction_count)

if st.session_state.last_predictions:
    st.sidebar.markdown("### 🕐 Recent Predictions")
    for pred in st.session_state.last_predictions[-3:]:
        st.sidebar.text(f"{pred['disease']}: {pred['result']}")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
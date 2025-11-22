# Multi-Disease Prediction System Backend

Flask backend for predicting Diabetes, Heart Disease, and Parkinson's Disease using Machine Learning.

## Features

- **3 Disease Predictions**: Diabetes, Heart Disease, Parkinson's
- **10 ML Models**: Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM, AdaBoost, Bagging, XGBoost, Voting
- **SMOTE**: Handles class imbalance
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: EDA plots, model comparisons, confusion matrices, ROC curves
- **Production-Ready**: Optimized for Railway deployment

## Project Structure
```
disease-prediction-backend/
├── app.py
├── train_all_models.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── datasets/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.csv
├── models/
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── parkinsons_model.pkl
│   └── model_metadata.json
├── static/images/
│   ├── diabetes/
│   ├── heart/
│   └── parkinsons/
└── src/
    ├── preprocessing.py
    ├── train_models.py
    ├── evaluate.py
    ├── visualizations.py
    └── predict.py
```

## Railway Deployment Steps

### 1. Train Models Locally (Optional)
```bash
python train_all_models.py
```

This generates:
- Model files in `models/`
- Visualizations in `static/images/`
- Metadata in `models/model_metadata.json`

### 2. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

### 3. Deploy to Railway

1. Go to [Railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Python and uses `Procfile`
5. Wait for deployment (~3-5 minutes)
6. Get your public URL (e.g., `https://your-app.railway.app`)

### 4. Train Models on Railway (if not trained locally)

SSH into Railway or use their terminal:
```bash
python train_all_models.py
```

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00"
}
```

### Models Info
```bash
GET /models/info
```

Response:
```json
{
  "diabetes": {
    "model": "XGBoost",
    "accuracy": 0.9757,
    "features": ["Pregnancies", "Glucose", ...],
    "n_features": 8
  }
}
```

### Performance Metrics
```bash
GET /models/performance
```

Response:
```json
{
  "diabetes": {
    "best_model": {
      "name": "XGBoost",
      "accuracy": 0.9757,
      "precision": 0.9661,
      "recall": 0.9048,
      "f1": 0.9268,
      "roc_auc": 0.98
    },
    "all_models": {...}
  }
}
```

### Predict Diabetes
```bash
POST /predict/diabetes
Content-Type: application/json

{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

### Predict Heart Disease
```bash
POST /predict/heart
Content-Type: application/json

{
  "age": 63,
  "sex": 1,
  "cp": 3,
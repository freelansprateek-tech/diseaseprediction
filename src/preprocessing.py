import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df, target_col, drop_cols=None, test_size=0.25, random_state=42):
    df = df.copy()
    
    # Drop non-numeric columns (except target) automatically
    for col in df.columns:
        if col != target_col and df[col].dtype == 'object':
            df = df.drop(columns=[col])
            print(f"Dropped non-numeric column: {col}")
    
    # Also drop specified columns
    if drop_cols:
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"Dropped specified column: {col}")
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Handle missing values in numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    feature_names = list(X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, feature_names
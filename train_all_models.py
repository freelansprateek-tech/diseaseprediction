import os
import json
import warnings
from datetime import datetime
from src.preprocessing import load_data, preprocess_data
from src.train_models import train_all_models
from src.visualizations import generate_all_visualizations

warnings.filterwarnings('ignore')

def create_directories():
    dirs = [
        'models',
        'static/images/diabetes',
        'static/images/heart',
        'static/images/parkinsons'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def train_disease_models(disease_name, csv_file, target_col, drop_cols=None):
    print(f"\n{'='*60}")
    print(f"Training {disease_name.upper()} prediction models")
    print(f"{'='*60}\n")
    
    # Load and inspect data
    df = load_data(csv_file)
    print(f"Loaded columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}\n")
    
    # Drop the name column explicitly for parkinsons
    if disease_name == 'parkinsons' and 'name' in df.columns:
        df = df.drop(columns=['name'])
        print("Dropped 'name' column for parkinsons")
    
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
        df, target_col, drop_cols
    )
    
    print(f"Dataset: {len(df)} samples, {len(feature_names)} features")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}\n")
    
    results = train_all_models(X_train, X_test, y_train, y_test)
    
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    model_path = f'models/{disease_name}_model.pkl'
    scaler_path = f'models/{disease_name}_scaler.pkl'
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        'disease': disease_name,
        'best_model': best_model_name,
        'accuracy': results[best_model_name]['accuracy'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'f1': results[best_model_name]['f1'],
        'roc_auc': results[best_model_name]['roc_auc'],
        'features': feature_names,
        'n_features': len(feature_names),
        'n_samples': len(df),
        'trained_at': datetime.now().isoformat(),
        'all_models': {k: {
            'accuracy': v['accuracy'],
            'precision': v['precision'],
            'recall': v['recall'],
            'f1': v['f1'],
            'roc_auc': v['roc_auc']
        } for k, v in results.items()}
    }
    
    generate_all_visualizations(
        df, X_train, X_test, y_train, y_test,
        results, best_model, feature_names,
        disease_name, target_col
    )
    
    return metadata

if __name__ == '__main__':
    import numpy as np
    import joblib
    
    create_directories()
    
    all_metadata = {}
    
    diseases = [
        ('diabetes', 'datasets/diabetes.csv', 'Outcome', None),
        ('heart', 'datasets/heart.csv', 'target', None),
        ('parkinsons', 'datasets/parkinsons.csv', 'status', ['name'])
    ]
    
    for disease, csv, target, drop in diseases:
        try:
            metadata = train_disease_models(disease, csv, target, drop)
            all_metadata[disease] = metadata
        except Exception as e:
            print(f"Error training {disease}: {e}")
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training complete! Summary:")
    print(f"{'='*60}")
    for disease, meta in all_metadata.items():
        print(f"{disease.capitalize()}: {meta['best_model']} - {meta['accuracy']:.2%}")
    print(f"{'='*60}\n")
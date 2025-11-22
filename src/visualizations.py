import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

plt.switch_backend('Agg')
sns.set_style('whitegrid')

def save_fig(fig, path):
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def plot_target_distribution(df, target_col, disease):
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = df[target_col].value_counts()
    ax.bar(['Negative', 'Positive'], counts.values, color=['#2ecc71', '#e74c3c'])
    ax.set_title(f'{disease.capitalize()} - Target Distribution')
    ax.set_ylabel('Count')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha='center', fontsize=12)
    save_fig(fig, f'static/images/{disease}/target_distribution.png')

def plot_correlation_heatmap(df, target_col, disease):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax)
    ax.set_title(f'{disease.capitalize()} - Correlation Heatmap')
    save_fig(fig, f'static/images/{disease}/correlation_heatmap.png')

def plot_feature_distributions(df, target_col, disease):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != target_col][:9]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        if idx < 9:
            axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black')
            axes[idx].set_title(col)
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
    
    for idx in range(len(numeric_cols), 9):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    save_fig(fig, f'static/images/{disease}/feature_distributions.png')

def plot_boxplots(df, target_col, disease):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != target_col][:8]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        if idx < 8:
            axes[idx].boxplot(df[col].dropna(), vert=True)
            axes[idx].set_title(col)
            axes[idx].set_ylabel('Value')
    
    for idx in range(len(numeric_cols), 8):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    save_fig(fig, f'static/images/{disease}/boxplots.png')

def plot_model_comparison(results, disease):
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        values = [results[m][metric] for m in models]
        axes[idx].barh(models, values, color='steelblue')
        axes[idx].set_xlabel(metric.capitalize())
        axes[idx].set_title(f'{disease.capitalize()} - {metric.capitalize()}')
        axes[idx].set_xlim(0, 1)
        
        for i, v in enumerate(values):
            axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    save_fig(fig, f'static/images/{disease}/model_comparison.png')

def plot_confusion_matrix(cm, disease, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{disease.capitalize()} - {model_name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    save_fig(fig, f'static/images/{disease}/confusion_matrix.png')

def plot_roc_curve(y_test, y_proba, disease, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{disease.capitalize()} - {model_name} ROC Curve')
    ax.legend(loc='lower right')
    save_fig(fig, f'static/images/{disease}/roc_curve.png')

def plot_feature_importance(model, feature_names, disease):
    if not hasattr(model, 'feature_importances_'):
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([feature_names[i] for i in indices], importances[indices], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(f'{disease.capitalize()} - Feature Importance')
    ax.invert_yaxis()
    save_fig(fig, f'static/images/{disease}/feature_importance.png')

def generate_all_visualizations(df, X_train, X_test, y_train, y_test, results, best_model, feature_names, disease, target_col):
    print(f"Generating visualizations for {disease}...")
    
    plot_target_distribution(df, target_col, disease)
    plot_correlation_heatmap(df, target_col, disease)
    plot_feature_distributions(df, target_col, disease)
    plot_boxplots(df, target_col, disease)
    plot_model_comparison(results, disease)
    
    best_name = [k for k, v in results.items() if v['model'] == best_model][0]
    best_cm = results[best_name]['confusion_matrix']
    best_proba = results[best_name]['y_proba']
    
    plot_confusion_matrix(best_cm, disease, best_name)
    plot_roc_curve(y_test, best_proba, disease, best_name)
    plot_feature_importance(best_model, feature_names, disease)
    
    print(f"Visualizations saved to static/images/{disease}/")
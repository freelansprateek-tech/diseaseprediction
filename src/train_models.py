from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.evaluate import evaluate_model

def get_classifiers():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Bagging': BaggingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'Voting': VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')),
                ('lr', LogisticRegression(max_iter=1000, random_state=42))
            ],
            voting='soft'
        ),
        'Stacking': StackingClassifier(
                    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('nb', GaussianNB())
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)
    }

def train_all_models(X_train, X_test, y_train, y_test):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    classifiers = get_classifiers() 
    results = {}
    
    for name, clf in classifiers.items():
        print(f"Training {name}...", end=' ')
        
        clf.fit(X_train_res, y_train_res)
        
        metrics = evaluate_model(clf, X_test, y_test)
        
        results[name] = {
            'model': clf,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'confusion_matrix': metrics['confusion_matrix'],
            'y_pred': metrics['y_pred'],
            'y_proba': metrics['y_proba']
        }
        
        print(f"Acc: {metrics['accuracy']:.4f}")
    
    return results
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Load and prepare the dataset"""
    df = pd.read_csv('data/processed/location_features.csv')
    
    # Create target variable based on presence of Walmart stores
    y = (df['walmart_stores_within_50km'] > 0).astype(int)
    
    # Select features for training
    feature_columns = [
        'population', 'population_density', 'per_capita_income',
        'employment_rate', 'market_potential', 'employment_health_score',
        'distance_to_nearest_walmart'
    ]
    X = df[feature_columns]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X, y

def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train a baseline XGBoost model without any class imbalance handling"""
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_estimators=100,
        learning_rate=0.1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return evaluate_model(y_test, y_pred, y_prob, "Baseline Model")

def train_weighted_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model with scale_pos_weight"""
    # Calculate scale_pos_weight as ratio of negative to positive samples
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return evaluate_model(y_test, y_pred, y_prob, "Weighted Model")

def train_smote_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model with SMOTE oversampling"""
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_estimators=100,
        learning_rate=0.1
    )
    
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return evaluate_model(y_test, y_pred, y_prob, "SMOTE Model")

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Evaluate model performance and return metrics"""
    # Calculate metrics
    class_report = classification_report(y_true, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Calculate PR AUC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    metrics = {
        'model_name': model_name,
        'accuracy': class_report['accuracy'],
        'precision': class_report['1']['precision'],
        'recall': class_report['1']['recall'],
        'f1_score': class_report['1']['f1-score'],
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    logging.info(f"\nResults for {model_name}:")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall: {metrics['recall']:.4f}")
    logging.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logging.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logging.info(f"PR AUC: {metrics['pr_auc']:.4f}")
    
    return metrics

def plot_results(results):
    """Plot comparison of model performances"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    model_names = [result['model_name'] for result in results]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, result in enumerate(results):
        values = [result[metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=result['model_name'])
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('data/processed/model_comparison.png')
    plt.close()

def main():
    """Run imbalanced learning experiments"""
    logging.info("Starting imbalanced learning experiments...")
    
    # Create output directory if it doesn't exist
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y = load_data()
    logging.info(f"Dataset shape: {X.shape}")
    logging.info(f"Class distribution: \n{pd.Series(y).value_counts(normalize=True)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train and evaluate models
    results = []
    
    # Baseline model
    baseline_metrics = train_baseline_model(X_train, y_train, X_test, y_test)
    results.append(baseline_metrics)
    
    # Weighted model
    weighted_metrics = train_weighted_model(X_train, y_train, X_test, y_test)
    results.append(weighted_metrics)
    
    # SMOTE model
    smote_metrics = train_smote_model(X_train, y_train, X_test, y_test)
    results.append(smote_metrics)
    
    # Plot results
    plot_results(results)
    
    # Save results
    with open('data/processed/imbalanced_learning_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info("\nExperiments complete! Results have been saved to:")
    logging.info("- data/processed/imbalanced_learning_results.json")
    logging.info("- data/processed/model_comparison.png")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_score, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Load the feature matrix and prepare it for training."""
    # Load the features
    df = pd.read_csv('data/processed/location_features.csv')
    
    # Define features to use for training (excluding walmart_stores_within_50km)
    feature_columns = [
        'population_density',
        'per_capita_income',
        'employment_rate',
        'rental_price',
        'distance_to_nearest_walmart',
        'market_potential',
        'employment_health_score',
        'has_target'  # Include competitor presence
    ]
    
    # Prepare features and target
    X = df[feature_columns]
    y = (df['walmart_stores_within_50km'] > 0).astype(int)  # Binary target: 1 if has Walmart within 50km, 0 otherwise
    
    # Print class distribution
    print("\nClass distribution:")
    print(f"Positive class (has Walmart within 50km): {y.mean():.2%}")
    print(f"Negative class (no Walmart within 50km): {(1 - y.mean()):.2%}")
    
    return X, y, feature_columns

def train_and_evaluate_model(X, y, feature_columns):
    """Train XGBoost model and evaluate its performance."""
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Add stratification
    )
    
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print(f"Positive class distribution - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    # Define hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'scale_pos_weight': [1, 2, 3, 4, 5]  # Add class weight parameter
    }
    
    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )
    
    # Perform randomized search for hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=25,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    # Get best model
    best_model = random_search.best_estimator_
    print("\nBest hyperparameters:", random_search.best_params_)
    
    # Make predictions
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate Precision@k for different k values
    k_values = [10, 50, 100, 500]
    precision_at_k = {}
    for k in k_values:
        top_k_indices = np.argsort(y_pred_proba)[-k:]
        precision_at_k[k] = precision_score(y_test.iloc[top_k_indices], y_pred[top_k_indices])
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"MSE: {mse:.4f}")
    for k, precision in precision_at_k.items():
        print(f"Precision@{k}: {precision:.4f}")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Walmart Store Presence Prediction')
    plt.tight_layout()
    plt.savefig('data/processed/feature_importance.png')
    
    # Save feature importance to CSV
    feature_importance.to_csv('data/processed/feature_importance.csv', index=False)
    print("\nFeature importance plot saved to: data/processed/feature_importance.png")
    print("Feature importance data saved to: data/processed/feature_importance.csv")
    
    return best_model, feature_importance

if __name__ == "__main__":
    # Load and prepare data
    X, y, feature_columns = load_and_prepare_data()
    
    # Train and evaluate model
    best_model, feature_importance = train_and_evaluate_model(X, y, feature_columns) 
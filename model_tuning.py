import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
import joblib
from pathlib import Path

def load_and_prepare_data():
    """Load and prepare data for model training."""
    print("Loading data...")
    df = pd.read_csv('data/processed/location_features.csv')
    
    # Define feature columns (using normalized versions)
    feature_columns = [
        'population_density',
        'per_capita_income',
        'employment_rate',
        'unemployment_rate',
        'labor_force_participation',
        'market_potential',
        'employment_health_score',
        'distance_to_nearest_walmart'
    ]
    
    # Prepare features and target
    X = df[feature_columns]
    y = (df['distance_to_nearest_walmart'] <= 50).astype(int)  # 1 if Walmart exists within 50km
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_columns

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model."""
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate precision@k
    k_values = [10, 50, 100, 500]
    precision_at_k = {}
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    
    # Convert y_test to numpy array if it's not already
    y_test_np = np.array(y_test)
    
    for k in k_values:
        if k > len(y_test_np):
            precision_at_k[f'precision@{k}'] = np.nan
            continue
            
        top_k_indices = sorted_indices[:k]
        precision_at_k[f'precision@{k}'] = np.mean(y_test_np[top_k_indices])
    
    return {
        'model_name': model_name,
        'auc_roc': auc_roc,
        'mse': mse,
        **precision_at_k,
        'model': model
    }

def get_feature_importance(model, feature_columns, model_name):
    """Extract feature importance from the model."""
    if model_name == 'XGBoost':
        importance = model.feature_importances_
    elif model_name == 'LightGBM':
        importance = model.feature_importances_
    elif model_name == 'RandomForest':
        importance = model.feature_importances_
    else:
        return None
    
    return pd.DataFrame({
        'feature': feature_columns,
        'importance': importance
    }).sort_values('importance', ascending=False)

def plot_feature_importance(importance_df, model_name):
    """Plot feature importance."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'data/processed/feature_importance_{model_name.lower()}.png')
    plt.close()

def tune_hyperparameters():
    """Perform hyperparameter tuning for all models."""
    X_train, X_test, y_train, y_test, feature_columns = load_and_prepare_data()
    
    # Define parameter spaces
    xgb_params = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'scale_pos_weight': [1, 2, 5]
    }
    
    lgb_params = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_samples': randint(1, 50),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }
    
    rf_params = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Initialize models
    models = {
        'XGBoost': (xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_params),
        'LightGBM': (lgb.LGBMClassifier(), lgb_params),
        'RandomForest': (RandomForestClassifier(), rf_params)
    }
    
    results = []
    feature_importance_dict = {}
    
    for model_name, (model, param_space) in models.items():
        print(f"\nTuning {model_name}...")
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            model,
            param_space,
            n_iter=50,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Evaluate model
        model_results = train_and_evaluate_model(
            best_model, X_train, X_test, y_train, y_test, model_name
        )
        
        # Get feature importance
        importance_df = get_feature_importance(best_model, feature_columns, model_name)
        feature_importance_dict[model_name] = importance_df
        
        # Plot feature importance
        plot_feature_importance(importance_df, model_name)
        
        # Save best model
        joblib.dump(best_model, f'data/processed/best_{model_name.lower()}_model.pkl')
        
        # Add results
        results.append({
            **model_results,
            'best_params': random_search.best_params_
        })
        
        print(f"\n{model_name} Best Parameters:")
        for param, value in random_search.best_params_.items():
            print(f"{param}: {value}")
    
    return results, feature_importance_dict

def analyze_geographic_errors(best_model, model_name):
    """Analyze prediction errors by geographic region."""
    # Load full dataset
    df = pd.read_csv('data/processed/location_features.csv')
    
    # Prepare features
    feature_columns = [
        'population_density',
        'per_capita_income',
        'employment_rate',
        'unemployment_rate',
        'labor_force_participation',
        'market_potential',
        'employment_health_score',
        'distance_to_nearest_walmart'
    ]
    
    X = df[feature_columns]
    y_true = (df['distance_to_nearest_walmart'] <= 50).astype(int)
    
    # Make predictions
    y_pred_proba = best_model.predict_proba(X)[:, 1]
    y_pred = best_model.predict(X)
    
    # Calculate errors
    errors = np.abs(y_true - y_pred_proba)
    
    # Add predictions and errors to DataFrame
    df['predicted_probability'] = y_pred_proba
    df['prediction_error'] = errors
    
    # Create error map
    plt.figure(figsize=(15, 10))
    plt.scatter(df['lng'], df['lat'], 
               c=df['prediction_error'], 
               cmap='RdYlBu_r',
               s=20, 
               alpha=0.6)
    plt.colorbar(label='Prediction Error')
    plt.title(f'Geographic Distribution of Prediction Errors - {model_name}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f'data/processed/error_map_{model_name.lower()}.png')
    plt.close()
    
    # Analyze errors by region
    df['region'] = pd.cut(df['lng'], 
                         bins=[-180, -115, -95, -75, 0],
                         labels=['West', 'Central', 'East', 'Other'])
    
    region_errors = df.groupby('region')['prediction_error'].agg(['mean', 'std']).round(4)
    return region_errors

def main():
    # Create output directory
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Tune and evaluate models
    print("Starting model tuning and evaluation...")
    results, feature_importance_dict = tune_hyperparameters()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/processed/model_comparison_results.csv', index=False)
    
    # Print summary
    print("\nModel Comparison Results:")
    print(results_df[['model_name', 'auc_roc', 'mse', 'precision@10', 'precision@50']].to_string())
    
    # Find best model
    best_model_name = results_df.loc[results_df['auc_roc'].idxmax(), 'model_name']
    best_model = joblib.load(f'data/processed/best_{best_model_name.lower()}_model.pkl')
    
    # Analyze geographic errors for best model
    print(f"\nAnalyzing geographic errors for best model ({best_model_name})...")
    region_errors = analyze_geographic_errors(best_model, best_model_name)
    
    print("\nPrediction Errors by Region:")
    print(region_errors)
    
    # Save region errors
    region_errors.to_csv('data/processed/region_errors.csv')

if __name__ == "__main__":
    main() 
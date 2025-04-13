import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from datetime import datetime

# Set up logging with timestamp
log_filename = f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    logging.info("Loading and preprocessing data...")
    
    try:
        # Load enhanced features with chunking for large files
        chunks = []
        for chunk in pd.read_csv('data/processed/enhanced_features.csv', chunksize=10000):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        logging.info(f"Successfully loaded {len(df)} rows of data")
        
        # Create target variable
        y = (df['walmart_stores_within_50km'] > 0).astype(int)
        
        # Select features (using the same feature set as in optimize_weighted_model.py)
        feature_columns = [
            # Base features
            'population', 'population_density', 'per_capita_income',
            'employment_rate', 'market_potential', 'employment_health_score',
            'distance_to_nearest_walmart',
            
            # Store density features
            'walmart_density_5km', 'walmart_density_10km', 
            'walmart_density_25km', 'walmart_density_50km',
            
            # Interaction terms
            'population_employment', 'density_income', 'market_employment',
            'income_density', 'market_density', 'employment_density',
            
            # Real estate features
            'rental_price', 'rental_price_volatility', 'rental_price_trend',
            
            # Competitor density features
            'target_density_5km', 'target_density_10km', 'target_density_25km',
            'costco_density_5km', 'costco_density_10km', 'costco_density_25km',
            'grocery_density_5km', 'grocery_density_10km', 'grocery_density_25km'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in df.columns]
        logging.info(f"Using {len(available_features)} features: {available_features}")
        X = df[available_features]
        
        # Handle missing values by imputing with median
        X = X.fillna(X.median())
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Double check for any remaining NaN values
        if X_scaled.isna().any().any():
            logging.warning("Found NaN values after scaling, filling with 0")
            X_scaled = X_scaled.fillna(0)
        
        return X, X_scaled, y, df[['zip', 'lat', 'lng']]
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def optimize_xgboost(trial, X_train, y_train, X_val, y_val):
    """Optimize XGBoost hyperparameters using Optuna"""
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 6),  # Reduced range
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),  # Reduced range
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.3)  # Reduced range
    }
    
    model = xgb.XGBClassifier(
        **params,
        objective='binary:logistic',
        random_state=42,
        scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1)
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,  # Reduced from 20
        eval_metric='auc',
        verbose=False
    )
    
    return model.best_score

def optimize_random_forest(trial, X_train, y_train):
    """Optimize Random Forest hyperparameters using Optuna"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),  # Reduced range
        'max_depth': trial.suggest_int('max_depth', 3, 6),  # Reduced range
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }
    
    model = RandomForestClassifier(
        **params,
        random_state=42,
        class_weight='balanced'
    )
    
    # Use cross-validation score with fewer splits
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # Reduced from 3
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    return scores.mean()

def optimize_neural_network(trial, X_train, y_train, X_val, y_val):
    """Optimize Neural Network hyperparameters using Optuna"""
    # Get number of layers and units for each layer
    n_layers = trial.suggest_int('n_layers', 1, 2)
    hidden_layer_sizes = tuple(
        trial.suggest_int(f'n_units_l{i}', 32, 128)
        for i in range(n_layers)
    )
    
    params = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'learning_rate_init': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64])
    }
    
    model = MLPClassifier(
        **params,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

def train_and_evaluate_models():
    """Train and evaluate all models"""
    # Load and preprocess data
    X, X_scaled, y, location_info = load_and_preprocess_data()
    logging.info(f"Dataset shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Split location_info to match the test set
    _, location_info_test = train_test_split(
        location_info, test_size=0.2, random_state=42
    )
    
    # Further split training data for validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    X_train_scaled_final, X_val_scaled, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Dictionary to store results and predictions
    results = {}
    predictions = pd.DataFrame(index=X.index)
    
    # Optimize XGBoost
    logging.info("Optimizing XGBoost...")
    xgb_study = optuna.create_study(direction='maximize')
    xgb_study.optimize(
        lambda trial: optimize_xgboost(trial, X_train_final, y_train_final, X_val, y_val),
        n_trials=20
    )
    logging.info(f"Best XGBoost trial: {xgb_study.best_trial.value:.4f}")
    
    # Train final XGBoost model
    best_xgb = xgb.XGBClassifier(
        **xgb_study.best_params,
        objective='binary:logistic',
        random_state=42,
        scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1)
    )
    best_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        eval_metric='auc',
        verbose=False
    )
    results['xgboost'] = evaluate_model(best_xgb, X_test, y_test, 'XGBoost', location_info_test)
    predictions['xgboost_predictions'] = best_xgb.predict(X)
    
    # Optimize Random Forest
    logging.info("Optimizing Random Forest...")
    rf_study = optuna.create_study(direction='maximize')
    rf_study.optimize(
        lambda trial: optimize_random_forest(trial, X_train_final, y_train_final),
        n_trials=20
    )
    logging.info(f"Best Random Forest trial: {rf_study.best_trial.value:.4f}")
    
    # Train final Random Forest model
    best_rf = RandomForestClassifier(
        **rf_study.best_params,
        random_state=42,
        class_weight='balanced'
    )
    best_rf.fit(X_train, y_train)
    results['random_forest'] = evaluate_model(best_rf, X_test, y_test, 'Random Forest', location_info_test)
    predictions['random_forest_predictions'] = best_rf.predict(X)
    
    # Optimize Neural Network
    logging.info("Optimizing Neural Network...")
    nn_study = optuna.create_study(direction='maximize')
    nn_study.optimize(
        lambda trial: optimize_neural_network(
            trial, X_train_scaled_final, y_train_final, X_val_scaled, y_val
        ),
        n_trials=20
    )
    logging.info(f"Best Neural Network trial: {nn_study.best_trial.value:.4f}")
    
    # Train final Neural Network model
    n_layers = nn_study.best_params['n_layers']
    hidden_layer_sizes = tuple(
        nn_study.best_params[f'n_units_l{i}']
        for i in range(n_layers)
    )
    
    best_nn = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=nn_study.best_params['learning_rate'],
        alpha=nn_study.best_params['alpha'],
        batch_size=nn_study.best_params['batch_size'],
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    best_nn.fit(X_train_scaled, y_train)
    results['neural_network'] = evaluate_model(
        best_nn, X_test_scaled, y_test, 'Neural Network', location_info_test
    )
    predictions['neural_network_predictions'] = best_nn.predict(X_scaled)
    
    # Save predictions to enhanced features dataset
    enhanced_features = pd.read_csv('data/processed/enhanced_features.csv')
    enhanced_features = pd.concat([enhanced_features, predictions], axis=1)
    enhanced_features.to_csv('data/processed/enhanced_features.csv', index=False)
    
    return results

def calculate_spatial_metrics(y_true, y_pred, location_info):
    """Calculate spatial validation metrics"""
    # Mean Distance Error (MDE)
    # For each false prediction, calculate the distance to the nearest correct prediction
    false_predictions = y_true != y_pred
    if not any(false_predictions):
        mde = 0
    else:
        distances = []
        for idx in np.where(false_predictions)[0]:
            pred_zip = location_info.iloc[idx]
            # Find nearest correct prediction
            correct_indices = np.where(y_true == y_pred)[0]
            if len(correct_indices) > 0:
                min_dist = float('inf')
                for correct_idx in correct_indices:
                    correct_zip = location_info.iloc[correct_idx]
                    dist = np.sqrt(
                        (pred_zip['lat'] - correct_zip['lat'])**2 +
                        (pred_zip['lng'] - correct_zip['lng'])**2
                    )
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)
        mde = np.mean(distances) if distances else 0
    
    # Intersection-over-Union (IoU)
    # Calculate IoU for predicted Walmart presence areas
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    
    intersection = true_positive
    union = true_positive + false_positive + false_negative
    iou = intersection / union if union > 0 else 0
    
    return {
        'mean_distance_error': mde,
        'intersection_over_union': iou
    }

def analyze_regional_performance(y_true, y_pred, location_info):
    """Analyze prediction errors by region"""
    # Define regions based on latitude/longitude
    regions = {
        'Northeast': {'lat': (40, 50), 'lng': (-80, -60)},
        'Southeast': {'lat': (25, 40), 'lng': (-100, -75)},
        'Midwest': {'lat': (35, 50), 'lng': (-100, -80)},
        'West': {'lat': (30, 50), 'lng': (-125, -100)},
        'Southwest': {'lat': (25, 35), 'lng': (-125, -100)}
    }
    
    # Convert arrays to numpy arrays if they're pandas Series
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    
    regional_metrics = {}
    for region, bounds in regions.items():
        mask = (
            (location_info['lat'] >= bounds['lat'][0]) &
            (location_info['lat'] <= bounds['lat'][1]) &
            (location_info['lng'] >= bounds['lng'][0]) &
            (location_info['lng'] <= bounds['lng'][1])
        ).values  # Convert pandas boolean Series to numpy array
        
        if np.sum(mask) > 0:
            regional_metrics[region] = {
                'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
                'precision': precision_score(y_true[mask], y_pred[mask]),
                'recall': recall_score(y_true[mask], y_pred[mask]),
                'f1': f1_score(y_true[mask], y_pred[mask]),
                'sample_size': np.sum(mask)
            }
    
    return regional_metrics

def analyze_confidence_scores(model, X_test, y_test):
    """Analyze probability distributions for predictions"""
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate confidence metrics
    confidence_metrics = {
        'mean_confidence': np.mean(y_prob),
        'std_confidence': np.std(y_prob),
        'confidence_by_class': {
            'positive': np.mean(y_prob[y_test == 1]),
            'negative': np.mean(y_prob[y_test == 0])
        }
    }
    
    # Create confidence distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob, bins=50, alpha=0.7)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.savefig('data/processed/confidence_distribution.png')
    plt.close()
    
    return confidence_metrics

def evaluate_model(model, X_test, y_test, model_name, location_info=None):
    """Evaluate a single model with enhanced metrics"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    basic_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob)
    }
    
    # Spatial metrics
    spatial_metrics = {}
    if location_info is not None:
        spatial_metrics = calculate_spatial_metrics(y_test, y_pred, location_info)
    
    # Regional performance
    regional_metrics = {}
    if location_info is not None:
        regional_metrics = analyze_regional_performance(y_test, y_pred, location_info)
    
    # Confidence analysis
    confidence_metrics = analyze_confidence_scores(model, X_test, y_test)
    
    results = {
        'model_name': model_name,
        'metrics': {
            **basic_metrics,
            **spatial_metrics,
            'confidence': confidence_metrics,
            'regional_performance': regional_metrics
        }
    }
    
    # Log results
    logging.info(f"\n{model_name} Results:")
    logging.info("Basic Metrics:")
    for metric, value in basic_metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    if spatial_metrics:
        logging.info("\nSpatial Metrics:")
        for metric, value in spatial_metrics.items():
            logging.info(f"{metric}: {value:.4f}")
    
    logging.info("\nConfidence Metrics:")
    for metric, value in confidence_metrics.items():
        if isinstance(value, dict):
            logging.info(f"{metric}:")
            for submetric, subvalue in value.items():
                logging.info(f"  {submetric}: {subvalue:.4f}")
        else:
            logging.info(f"{metric}: {value:.4f}")
    
    if regional_metrics:
        logging.info("\nRegional Performance:")
        for region, metrics in regional_metrics.items():
            logging.info(f"\n{region}:")
            for metric, value in metrics.items():
                logging.info(f"  {metric}: {value:.4f}")
    
    return results

def plot_model_comparison(results):
    """Plot comparison of model performances with enhanced visualizations"""
    # Basic metrics plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    model_names = list(results.keys())
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model_name in enumerate(model_names):
        values = [results[model_name]['metrics'][metric] for metric in metrics]
        plt.bar(x + i * width, values, width, label=model_name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison - Basic Metrics')
    plt.xticks(x + width, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/processed/model_comparison_basic.png')
    plt.close()
    
    # Regional performance plot
    if 'regional_performance' in results[model_names[0]]['metrics']:
        regions = list(results[model_names[0]]['metrics']['regional_performance'].keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        plt.figure(figsize=(15, 8))
        x = np.arange(len(regions))
        width = 0.2
        
        for i, model_name in enumerate(model_names):
            for j, metric in enumerate(metrics):
                values = [
                    results[model_name]['metrics']['regional_performance'][region][metric]
                    for region in regions
                ]
                plt.bar(x + (i * len(metrics) + j) * width, values, width, 
                       label=f'{model_name} - {metric}')
        
        plt.xlabel('Regions')
        plt.ylabel('Score')
        plt.title('Regional Performance Comparison')
        plt.xticks(x + width * len(metrics) * len(model_names) / 2, regions, rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('data/processed/model_comparison_regional.png')
        plt.close()

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def main():
    """Run model comparison pipeline"""
    # Create output directory
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    X, X_scaled, y, location_info = load_and_preprocess_data()
    
    # Train and evaluate models
    results = train_and_evaluate_models()
    
    # Plot comparisons
    plot_model_comparison(results)
    
    # Save results to JSON
    with open('data/processed/model_comparison_results.json', 'w') as f:
        serializable_results = convert_to_serializable(results)
        json.dump(serializable_results, f, indent=4)
    
    logging.info("\nModel comparison complete! Results have been saved to:")
    logging.info("- data/processed/model_comparison_results.json")
    logging.info("- data/processed/model_comparison_basic.png")
    logging.info("- data/processed/model_comparison_regional.png")
    logging.info("- data/processed/confidence_distribution.png")

if __name__ == "__main__":
    main() 
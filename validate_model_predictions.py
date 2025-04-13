import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import json
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from math import radians, sin, cos, sqrt, atan2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in miles."""
    R = 3959.87433  # Earth's radius in miles

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

def calculate_iou(actual_locations, predicted_locations, distance_threshold=10):
    """Calculate IoU based on proximity of predicted locations to actual locations."""
    true_positives = 0
    for pred_lat, pred_lng in zip(predicted_locations['lat'], predicted_locations['lng']):
        for actual_lat, actual_lng in zip(actual_locations['lat'], actual_locations['lng']):
            if haversine_distance(pred_lat, pred_lng, actual_lat, actual_lng) <= distance_threshold:
                true_positives += 1
                break
    
    union = len(actual_locations) + len(predicted_locations) - true_positives
    if union == 0:
        return 0
    return true_positives / union

def validate_model_predictions():
    """Validate model predictions against actual Walmart store locations."""
    logger.info("Starting model validation...")

    # Load training data
    train_data = pd.read_csv('data/processed/training_data.csv')
    logger.info(f"Loaded {len(train_data)} records from training data")

    # Prepare feature columns
    feature_columns = [
        'population', 'population_density', 'per_capita_income',
        'employment_rate', 'unemployment_rate', 'labor_force_participation',
        'market_potential', 'distance_to_nearest_walmart'
    ]

    X = train_data[feature_columns]
    y = train_data['has_walmart']

    # Define best parameters from previous tuning
    best_params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42
    }

    # Train model
    model = xgb.XGBClassifier(**best_params)
    model.fit(X, y)
    logger.info("Model training completed")

    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Create DataFrame with predictions
    predicted_locations = train_data[y_pred == 1][['lat', 'lng', 'zip', 'city', 'state']]
    actual_locations = train_data[train_data['has_walmart'] == 1][['lat', 'lng', 'zip', 'city', 'state']]

    # Calculate validation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    iou = calculate_iou(actual_locations, predicted_locations)

    # Calculate mean distance error
    mean_distance_error = 0
    for pred_lat, pred_lng in zip(predicted_locations['lat'], predicted_locations['lng']):
        min_distance = float('inf')
        for actual_lat, actual_lng in zip(actual_locations['lat'], actual_locations['lng']):
            distance = haversine_distance(pred_lat, pred_lng, actual_lat, actual_lng)
            min_distance = min(min_distance, distance)
        mean_distance_error += min_distance
    mean_distance_error /= len(predicted_locations) if len(predicted_locations) > 0 else 1

    # Log validation metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"IoU (10-mile threshold): {iou:.4f}")
    logger.info(f"Mean Distance Error: {mean_distance_error:.2f} miles")

    # Save validation results
    validation_results = pd.DataFrame({
        'zip': train_data['zip'],
        'city': train_data['city'],
        'state': train_data['state'],
        'actual': y,
        'predicted': y_pred,
        'prediction_probability': y_pred_proba
    })
    
    # Create output directory if it doesn't exist
    os.makedirs('data/model', exist_ok=True)
    
    # Save validation results and metrics
    validation_results.to_csv('data/processed/model_validation_results.csv', index=False)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'iou': iou,
        'mean_distance_error': mean_distance_error
    }
    
    with open('data/model/validation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info("Validation results and metrics saved")

if __name__ == "__main__":
    validate_model_predictions() 
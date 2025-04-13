import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import xgboost as xgb
from pathlib import Path
import logging
import json
from sklearn.model_selection import train_test_split
from math import radians, sin, cos, sqrt, atan2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth."""
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

def load_best_params():
    """Load the best parameters from tuning results."""
    try:
        with open('data/model/tuning_results.json', 'r') as f:
            results = json.load(f)
        return max(results, key=lambda x: x['metrics']['test_accuracy'])['params']
    except Exception as e:
        logger.error(f"Error loading best parameters: {str(e)}")
        raise

def calculate_distance_error(y_true_coords, y_pred_coords):
    """Calculate mean distance error in miles using Haversine formula."""
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 3959.87433  # Earth's radius in miles
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    distances = [
        haversine_distance(true[0], true[1], pred[0], pred[1])
        for true, pred in zip(y_true_coords, y_pred_coords)
    ]
    return np.mean(distances), np.std(distances)

def calculate_iou(y_true_coords, y_pred_coords, radius_miles=10):
    """Calculate Intersection over Union for predicted locations."""
    def circle_intersection_area(d, r1, r2):
        """Calculate intersection area of two circles."""
        if d >= r1 + r2:
            return 0
        if d <= abs(r1 - r2):
            return np.pi * min(r1, r2)**2
        
        a = np.arccos((d**2 + r1**2 - r2**2) / (2*d*r1))
        b = np.arccos((d**2 + r2**2 - r1**2) / (2*d*r2))
        
        return (r1**2 * a + r2**2 * b - 
                d * r1 * np.sin(a))

    def calculate_single_iou(true_coord, pred_coord):
        d = haversine_distance(true_coord[0], true_coord[1], 
                             pred_coord[0], pred_coord[1])
        intersection = circle_intersection_area(d, radius_miles, radius_miles)
        union = 2 * np.pi * radius_miles**2 - intersection
        return intersection / union if union > 0 else 0

    ious = [
        calculate_single_iou(true, pred)
        for true, pred in zip(y_true_coords, y_pred_coords)
    ]
    return np.mean(ious)

def evaluate_model():
    """Perform comprehensive model evaluation."""
    logger.info("Starting model evaluation...")
    
    # Load data
    data = pd.read_csv('data/processed/training_data.csv')
    X = data.drop(['has_walmart', 'zip', 'city', 'state'], axis=1)
    y = data['has_walmart']
    
    # Load location data
    locations = pd.read_csv('data/processed/top_optimal_locations.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Load best parameters and create model
    best_params = load_best_params()
    model = xgb.XGBClassifier(**best_params, random_state=42)
    
    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate basic metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate error rate
    error_rate = 1 - metrics['Accuracy']
    
    # Calculate distance error and IoU
    test_locations = locations.iloc[y_test.index]
    pred_locations = test_locations.copy()
    pred_locations['has_walmart'] = y_pred
    
    true_coords = test_locations[y_test == 1][['lat', 'lng']].values
    pred_coords = pred_locations[y_pred == 1][['lat', 'lng']].values
    
    mean_dist_error, std_dist_error = calculate_distance_error(true_coords, pred_coords)
    iou = calculate_iou(true_coords, pred_coords)
    
    # Log results
    logger.info("\nModel Performance Metrics:")
    logger.info("-" * 50)
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info(f"\nError Rate: {error_rate:.4f}")
    logger.info(f"Mean Distance Error: {mean_dist_error:.2f} miles")
    logger.info(f"Std Distance Error: {std_dist_error:.2f} miles")
    logger.info(f"Intersection-over-Union (IoU): {iou:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"True Negatives: {cm[0,0]}")
    logger.info(f"False Positives: {cm[0,1]}")
    logger.info(f"False Negatives: {cm[1,0]}")
    logger.info(f"True Positives: {cm[1,1]}")
    
    # Save results
    results = {
        'metrics': metrics,
        'error_rate': error_rate,
        'distance_error': {
            'mean': mean_dist_error,
            'std': std_dist_error
        },
        'iou': iou,
        'confusion_matrix': cm.tolist()
    }
    
    output_file = 'data/model/evaluation_results.json'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Feature importance analysis
    importance_scores = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nFeature Importance:")
    for _, row in importance_scores.iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    return results

if __name__ == "__main__":
    evaluate_model() 
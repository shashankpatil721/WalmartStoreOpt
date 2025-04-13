import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import logging
from pathlib import Path
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load the training data."""
    try:
        data = pd.read_csv('data/processed/training_data.csv')
        X = data.drop(['has_walmart', 'zip', 'city', 'state'], axis=1)
        y = data['has_walmart']
        return X, y
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_model(params):
    """Create an XGBoost model with specified parameters."""
    return xgb.XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        min_child_weight=params['min_child_weight'],
        gamma=params['gamma'],
        colsample_bytree=params['colsample_bytree'],
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

def evaluate_model(model, X, y):
    """Evaluate model using cross-validation."""
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = cross_validate(
        model, X, y,
        scoring=scoring,
        cv=cv,
        return_train_score=True
    )
    
    return {
        'test_accuracy': np.mean(scores['test_accuracy']),
        'test_precision': np.mean(scores['test_precision']),
        'test_recall': np.mean(scores['test_recall']),
        'test_f1': np.mean(scores['test_f1']),
        'train_accuracy': np.mean(scores['train_accuracy']),
        'train_precision': np.mean(scores['train_precision']),
        'train_recall': np.mean(scores['train_recall']),
        'train_f1': np.mean(scores['train_f1'])
    }

def tune_hyperparameters():
    """Perform hyperparameter tuning."""
    logger.info("Starting hyperparameter tuning...")
    
    # Load data
    X, y = load_data()
    
    # Define parameter sets to test
    param_sets = [
        {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.1,
            'min_child_weight': 3,
            'gamma': 0.1,
            'colsample_bytree': 0.9
        },
        {
            'n_estimators': 600,
            'max_depth': 6,
            'learning_rate': 0.05,
            'min_child_weight': 1,
            'gamma': 0.05,
            'colsample_bytree': 1.0
        }
    ]
    
    results = []
    
    for i, params in enumerate(param_sets, 1):
        logger.info(f"\nTesting parameter set {i}:")
        logger.info(json.dumps(params, indent=2))
        
        # Create and evaluate model
        model = create_model(params)
        metrics = evaluate_model(model, X, y)
        
        results.append({
            'params': params,
            'metrics': metrics
        })
        
        logger.info("\nCross-validation results:")
        logger.info("Test metrics:")
        logger.info(f"Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"Precision: {metrics['test_precision']:.4f}")
        logger.info(f"Recall: {metrics['test_recall']:.4f}")
        logger.info(f"F1-score: {metrics['test_f1']:.4f}")
        
        logger.info("\nTrain metrics:")
        logger.info(f"Accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"Precision: {metrics['train_precision']:.4f}")
        logger.info(f"Recall: {metrics['train_recall']:.4f}")
        logger.info(f"F1-score: {metrics['train_f1']:.4f}")
    
    # Save results
    output_file = 'data/model/tuning_results.json'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['metrics']['test_accuracy'])
    logger.info("\nBest parameters:")
    logger.info(json.dumps(best_result['params'], indent=2))
    logger.info("\nBest test metrics:")
    logger.info(f"Accuracy: {best_result['metrics']['test_accuracy']:.4f}")
    logger.info(f"Precision: {best_result['metrics']['test_precision']:.4f}")
    logger.info(f"Recall: {best_result['metrics']['test_recall']:.4f}")
    logger.info(f"F1-score: {best_result['metrics']['test_f1']:.4f}")
    
    return best_result

if __name__ == "__main__":
    tune_hyperparameters() 
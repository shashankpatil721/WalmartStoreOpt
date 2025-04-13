import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_data():
    """Load the trained model and nationwide features"""
    try:
        # Load nationwide features
        logger.info("Loading nationwide features...")
        nationwide_features = pd.read_csv('data/processed/features_nationwide.csv')
        
        # Load current test predictions for reference
        current_predictions = pd.read_csv('data/model/test_predictions.csv')
        
        # Get the list of features we have in our nationwide dataset
        available_features = [
            'population_density',
            'market_potential',
            'distance_to_nearest_walmart'
        ]
        
        # Merge current predictions with features
        current_predictions = current_predictions.merge(
            nationwide_features[['zip'] + available_features],
            on='zip',
            how='left'
        )
        
        logger.info(f"Features to use: {available_features}")
        return nationwide_features, current_predictions, available_features
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_features(df, feature_columns):
    """Prepare features for prediction"""
    try:
        # Select features
        X = df[feature_columns].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].isna().any():
                logger.info(f"Filling missing values in {col} with median")
                X[col] = X[col].fillna(X[col].median())
        
        # Add derived features
        X['log_population_density'] = np.log1p(X['population_density'])
        X['log_market_potential'] = np.log1p(X['market_potential'])
        
        return X
    
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise

def train_nationwide_model(current_predictions, feature_columns):
    """Train a new model using the current predictions as training data"""
    try:
        # Prepare training data
        X_train = prepare_features(current_predictions, feature_columns)
        y_train = current_predictions['actual']
        
        # Get final feature list including derived features
        all_features = feature_columns + ['log_population_density', 'log_market_potential']
        
        # Initialize and train model
        logger.info("Training model...")
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1
        )
        
        model.fit(X_train[all_features], y_train)
        logger.info("Model training completed")
        
        return model, all_features
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def generate_predictions(model, X, nationwide_features, all_features):
    """Generate predictions for all ZIP codes"""
    try:
        # Generate predictions and probabilities
        logger.info("Generating predictions...")
        predictions = model.predict(X[all_features])
        probabilities = model.predict_proba(X[all_features])[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'zip': nationwide_features['zip'],
            'city': nationwide_features['city'],
            'state': nationwide_features['state'],
            'latitude': nationwide_features['latitude'],
            'longitude': nationwide_features['longitude'],
            'predicted': predictions,
            'probability': probabilities
        })
        
        # Add actual column (initialize with 0)
        results['actual'] = 0
        
        # Update actual values where known
        known_locations = nationwide_features[nationwide_features['distance_to_nearest_walmart'] == 0]
        results.loc[results['zip'].isin(known_locations['zip']), 'actual'] = 1
        
        return results
    
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise

def save_predictions(results):
    """Save the nationwide predictions"""
    try:
        # Create output directory if it doesn't exist
        Path('data/model').mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        output_path = 'data/model/test_predictions_nationwide.csv'
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Print summary statistics
        logger.info("\nPrediction Summary:")
        logger.info(f"Total ZIP codes: {len(results)}")
        logger.info(f"Predicted locations: {results['predicted'].sum()}")
        logger.info(f"Actual locations: {results['actual'].sum()}")
        logger.info(f"Average prediction probability: {results['probability'].mean():.3f}")
        
        # Print state-level statistics
        state_stats = results.groupby('state').agg({
            'zip': 'count',
            'predicted': 'sum',
            'actual': 'sum'
        }).reset_index()
        
        logger.info("\nTop 10 states by predicted locations:")
        logger.info(state_stats.sort_values('predicted', ascending=False).head(10))
        
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise

def main():
    try:
        # Load data
        nationwide_features, current_predictions, feature_columns = load_model_and_data()
        
        # Prepare features for prediction
        X = prepare_features(nationwide_features, feature_columns)
        
        # Train model
        model, all_features = train_nationwide_model(current_predictions, feature_columns)
        
        # Generate predictions
        results = generate_predictions(model, X, nationwide_features, all_features)
        
        # Save predictions
        save_predictions(results)
        
        logger.info("Nationwide prediction generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
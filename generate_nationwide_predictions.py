import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load and prepare data for model training."""
    logger.info("Loading data...")
    
    # Load nationwide features
    features_df = pd.read_csv('data/processed/features_nationwide.csv')
    logger.info(f"Loaded {len(features_df)} ZIP codes")
    
    # Load actual Walmart locations
    try:
        walmart_locations = pd.read_csv('data/raw/Walmart_Store_Locations.csv')
        logger.info(f"Loaded {len(walmart_locations)} Walmart locations")
        
        # Extract and clean ZIP codes from both datasets
        walmart_zips = walmart_locations['zip_code'].astype(str).str.extract('(\d{5})')[0].dropna().unique()
        features_df['zip_clean'] = features_df['zip'].astype(str).str.extract('(\d{5})')[0]
        
        logger.info(f"Found {len(walmart_zips)} unique ZIP codes with Walmart stores")
        
        # Create actual column (1 if ZIP code has a Walmart, 0 otherwise)
        features_df['actual'] = features_df['zip_clean'].isin(walmart_zips).astype(int)
        logger.info(f"Found {features_df['actual'].sum()} ZIP codes with Walmart stores")
        
        # Drop temporary column
        features_df = features_df.drop('zip_clean', axis=1)
        
        # Add state information from Walmart locations to help with analysis
        walmart_state_counts = walmart_locations.groupby('state')['zip_code'].count()
        logger.info("\nActual Walmart stores by state:")
        for state, count in walmart_state_counts.nlargest(10).items():
            logger.info(f"{state}: {count:,} stores")
        
    except Exception as e:
        logger.error(f"Error loading Walmart locations: {str(e)}")
        raise
    
    return features_df

def prepare_features(df):
    """Prepare features for model training."""
    logger.info("Preparing features...")
    
    # List of features to use
    feature_columns = [
        'population_density',
        'market_potential',
        'distance_to_nearest_walmart'
    ]
    
    # Handle missing values
    for col in feature_columns:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled missing values in {col} with median: {median_val}")
    
    # Create derived features
    df['log_population_density'] = np.log1p(df['population_density'])
    df['log_market_potential'] = np.log1p(df['market_potential'])
    
    # Add final derived features to feature list
    feature_columns.extend(['log_population_density', 'log_market_potential'])
    
    return df, feature_columns

def train_nationwide_model(features_df):
    """Train model using data from all US regions."""
    logger.info("Training model on nationwide data...")
    
    # Prepare features
    features_df, feature_columns = prepare_features(features_df)
    
    # Create target variable (actual Walmart locations)
    y = features_df['actual'].values
    X = features_df[feature_columns].values
    
    # Calculate class weights to handle imbalanced data
    n_samples = len(y)
    n_positive = y.sum()
    class_weight = {
        0: 1.0,
        1: n_samples / (2 * n_positive)  # Give more weight to minority class
    }
    logger.info(f"Using class weights: {class_weight}")
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with class weights
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    # Calculate and log training metrics
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    logger.info(f"Training accuracy: {train_accuracy:.3f}")
    logger.info(f"Test accuracy: {test_accuracy:.3f}")
    
    return model, feature_columns

def generate_predictions(model, features_df, feature_columns):
    """Generate predictions for all ZIP codes."""
    logger.info("Generating predictions for all ZIP codes...")
    
    # Prepare features for prediction
    features_df, _ = prepare_features(features_df)
    
    # Generate predictions and probabilities
    probabilities = model.predict_proba(features_df[feature_columns].values)
    predictions = model.predict(features_df[feature_columns].values)
    
    # Add predictions to DataFrame
    features_df['predicted'] = predictions
    features_df['probability'] = probabilities[:, 1]  # Probability of positive class
    
    return features_df

def main():
    try:
        # Load and prepare data
        features_df = load_and_prepare_data()
        
        # Train model on nationwide data
        model, feature_columns = train_nationwide_model(features_df)
        
        # Generate predictions for all ZIP codes
        results_df = generate_predictions(model, features_df, feature_columns)
        
        # Save predictions
        output_file = 'data/model/test_predictions_nationwide.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved predictions to {output_file}")
        
        # Print summary statistics
        total_predictions = len(results_df)
        positive_predictions = results_df['predicted'].sum()
        actual_locations = results_df['actual'].sum()
        
        logger.info("\nPrediction Summary:")
        logger.info(f"Total ZIP codes processed: {total_predictions:,}")
        logger.info(f"Predicted Walmart locations: {positive_predictions:,}")
        logger.info(f"Actual Walmart locations: {actual_locations:,}")
        
        # Print top states by predicted locations
        state_stats = results_df[results_df['predicted'] == 1].groupby('state').size()
        logger.info("\nTop 10 states by predicted locations:")
        for state, count in state_stats.nlargest(10).items():
            logger.info(f"{state}: {count:,} locations")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
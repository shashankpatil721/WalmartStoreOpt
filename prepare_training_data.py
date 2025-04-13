import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_training_data():
    """Prepare training data for model tuning."""
    logger.info("Starting training data preparation...")
    
    # Load the predictions data
    predictions_file = 'data/processed/top_optimal_locations.csv'
    if not Path(predictions_file).exists():
        logger.error(f"Predictions file not found: {predictions_file}")
        return
    
    data = pd.read_csv(predictions_file)
    logger.info(f"Loaded {len(data)} records")
    
    # Create target variable with more balanced classes
    # Consider locations with high prediction scores as positive class
    # and locations with low prediction scores as negative class
    threshold = data['prediction_score'].median()
    data['has_walmart'] = (data['prediction_score'] >= threshold).astype(int)
    
    # Select features for training
    features = [
        'population',
        'population_density',
        'per_capita_income',
        'employment_rate',
        'unemployment_rate',
        'labor_force_participation',
        'market_potential',
        'distance_to_nearest_walmart'
    ]
    
    # Handle missing values and infinite values
    for col in features:
        # Replace infinite values with NaN
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with median
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            logger.info(f"Filled {data[col].isnull().sum()} missing values in {col}")
    
    # Remove outliers using IQR method
    for col in features:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower_bound, upper_bound)
    
    # Normalize numerical features
    for col in features:
        mean_val = data[col].mean()
        std_val = data[col].std()
        data[col] = (data[col] - mean_val) / std_val
        logger.info(f"Normalized column: {col}")
    
    # Create balanced dataset
    positive_samples = data[data['has_walmart'] == 1]
    negative_samples = data[data['has_walmart'] == 0].sample(
        n=len(positive_samples),
        random_state=42
    )
    
    balanced_data = pd.concat([positive_samples, negative_samples])
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the training data
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'training_data.csv'
    
    # Select relevant columns for training, including lat and lng
    training_data = balanced_data[features + ['has_walmart', 'zip', 'city', 'state', 'lat', 'lng']]
    training_data.to_csv(output_file, index=False)
    
    logger.info(f"Saved training data to {output_file}")
    logger.info(f"Training data shape: {training_data.shape}")
    logger.info(f"Positive class distribution: {(training_data['has_walmart'] == 1).mean():.2%}")
    
    # Print class distribution details
    class_dist = training_data['has_walmart'].value_counts()
    logger.info("\nClass distribution:")
    for label, count in class_dist.items():
        logger.info(f"Class {label}: {count} samples ({count/len(training_data):.2%})")
    
    # Print feature statistics
    logger.info("\nFeature statistics:")
    for col in features:
        logger.info(f"\n{col}:")
        logger.info(f"Mean: {training_data[col].mean():.4f}")
        logger.info(f"Std: {training_data[col].std():.4f}")
        logger.info(f"Min: {training_data[col].min():.4f}")
        logger.info(f"Max: {training_data[col].max():.4f}")

if __name__ == "__main__":
    prepare_training_data()
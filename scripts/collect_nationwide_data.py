import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data():
    """Load all raw data sources"""
    try:
        # Load ZIP code coordinates and basic info
        zip_coords = pd.read_csv('data/raw/uszips.csv')
        
        # Load existing features for structure reference
        existing_features = pd.read_csv('data/processed/features.csv')
        
        # Load current test predictions and add coordinates
        test_predictions = pd.read_csv('data/model/test_predictions.csv')
        
        # Add coordinates to test predictions
        test_predictions = test_predictions.merge(
            zip_coords[['zip', 'lat', 'lng']],
            on='zip',
            how='left'
        ).rename(columns={
            'lat': 'latitude',
            'lng': 'longitude'
        })
        
        return zip_coords, existing_features, test_predictions
    except Exception as e:
        logger.error(f"Error loading raw data: {str(e)}")
        raise

def calculate_market_potential(df):
    """Calculate market potential based on population and income"""
    if 'median_income' not in df.columns:
        logger.warning("Median income not found, using population as market potential")
        df['market_potential'] = df['population'] / df['population'].mean()
    else:
        df['market_potential'] = (df['population'] * df['median_income']) / 1e9
    return df

def calculate_distance_to_nearest_walmart(df, walmart_locations):
    """Calculate distance to nearest Walmart for each ZIP code"""
    from scipy.spatial.distance import cdist
    
    # Get coordinates of existing Walmart locations
    walmart_coords = walmart_locations[
        (walmart_locations['actual'] == 1) & 
        walmart_locations['latitude'].notna() & 
        walmart_locations['longitude'].notna()
    ][['latitude', 'longitude']].values
    
    if len(walmart_coords) == 0:
        logger.warning("No Walmart locations found with valid coordinates")
        df['distance_to_nearest_walmart'] = np.nan
        return df
    
    # Calculate distances for each ZIP code
    all_coords = df[['latitude', 'longitude']].values
    distances = cdist(all_coords, walmart_coords)
    
    # Get minimum distance for each ZIP code
    df['distance_to_nearest_walmart'] = np.min(distances, axis=1)
    return df

def estimate_missing_features(df, reference_data, feature_name):
    """Estimate missing features using reference data and geographical proximity"""
    if feature_name in reference_data.columns:
        # Map known values
        feature_map = reference_data.set_index('zip')[feature_name]
        df[feature_name] = df['zip'].map(feature_map)
        
        # For missing values, estimate based on state median
        state_medians = df[df[feature_name].notna()].groupby('state')[feature_name].median()
        
        # Fill missing values with state median
        for state in df['state'].unique():
            mask = (df['state'] == state) & (df[feature_name].isna())
            if state in state_medians:
                df.loc[mask, feature_name] = state_medians[state]
            else:
                df.loc[mask, feature_name] = df[feature_name].median()
    
    return df

def process_nationwide_data():
    """Process data for all US ZIP codes"""
    try:
        # Create data directory if it doesn't exist
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        
        # Load raw data
        logger.info("Loading raw data...")
        zip_coords, existing_features, test_predictions = load_raw_data()
        
        # Create nationwide dataset starting with ZIP coordinates
        logger.info("Creating nationwide dataset...")
        nationwide_data = zip_coords[['zip', 'lat', 'lng', 'city', 'state_id', 'population']].copy()
        nationwide_data = nationwide_data.rename(columns={
            'lat': 'latitude',
            'lng': 'longitude',
            'state_id': 'state'
        })
        
        # Calculate population density
        logger.info("Calculating population density...")
        if 'area' in zip_coords.columns:
            nationwide_data['population_density'] = nationwide_data['population'] / zip_coords['area']
        else:
            nationwide_data = estimate_missing_features(nationwide_data, existing_features, 'population_density')
        
        # Estimate median income
        logger.info("Estimating median income...")
        nationwide_data = estimate_missing_features(nationwide_data, existing_features, 'median_income')
        
        # Calculate market potential
        logger.info("Calculating market potential...")
        nationwide_data = calculate_market_potential(nationwide_data)
        
        # Calculate distance to nearest Walmart
        logger.info("Calculating distances to nearest Walmart...")
        nationwide_data = calculate_distance_to_nearest_walmart(nationwide_data, test_predictions)
        
        # Add employment metrics
        logger.info("Estimating employment metrics...")
        employment_metrics = ['labor_force_participation', 'employment_rate', 'unemployment_rate']
        for metric in employment_metrics:
            nationwide_data = estimate_missing_features(nationwide_data, existing_features, metric)
        
        # Save processed data
        logger.info("Saving processed data...")
        output_path = 'data/processed/features_nationwide.csv'
        nationwide_data.to_csv(output_path, index=False)
        logger.info(f"Nationwide data saved to {output_path}")
        
        # Print summary statistics
        logger.info("\nDataset Summary:")
        logger.info(f"Total ZIP codes: {len(nationwide_data)}")
        logger.info(f"States covered: {nationwide_data['state'].nunique()}")
        logger.info(f"Features included: {', '.join(nationwide_data.columns)}")
        
        # Print feature coverage
        logger.info("\nFeature Coverage:")
        for col in nationwide_data.columns:
            coverage = (nationwide_data[col].notna().sum() / len(nationwide_data)) * 100
            logger.info(f"{col}: {coverage:.1f}% complete")
        
        return nationwide_data
        
    except Exception as e:
        logger.error(f"Error processing nationwide data: {str(e)}")
        raise

if __name__ == "__main__":
    process_nationwide_data() 
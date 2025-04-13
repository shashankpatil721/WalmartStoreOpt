import pandas as pd
import numpy as np
from pathlib import Path
import requests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_base_features():
    """
    Create base features from Zillow data if they don't exist
    """
    try:
        logger.info("Creating base features...")
        
        # Load Zillow data for ZIP codes
        zillow_data = pd.read_csv('data/raw/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
        
        # Create basic features
        base_features = pd.DataFrame({
            'zip': zillow_data['RegionName'],
            'city': zillow_data['City'],
            'state': zillow_data['StateName'],
            'population': np.random.randint(1000, 100000, size=len(zillow_data)),  # Sample data
            'population_density': np.random.uniform(10, 5000, size=len(zillow_data)),  # Sample data
            'median_income': np.random.uniform(30000, 120000, size=len(zillow_data))  # Sample data
        })
        
        # Save base features
        output_path = 'data/processed/features.csv'
        base_features.to_csv(output_path, index=False)
        logger.info(f"Created base features file at {output_path}")
        
        return base_features
        
    except Exception as e:
        logger.error(f"Error creating base features: {str(e)}")
        return None

def load_zillow_data(filepath='data/raw/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'):
    """
    Load and process Zillow Home Value Index data
    """
    try:
        logger.info("Loading Zillow real estate data...")
        # Load Zillow data
        zillow_data = pd.read_csv(filepath)
        
        # Get the most recent value for each ZIP code
        recent_values = zillow_data[['RegionName', 'City', 'StateName']].copy()
        recent_values['median_property_value'] = zillow_data.iloc[:, -1]  # Last column has most recent values
        
        # Rename columns for consistency
        recent_values = recent_values.rename(columns={
            'RegionName': 'zip'
        })
        
        logger.info(f"Processed {len(recent_values)} ZIP codes with property values")
        return recent_values
        
    except Exception as e:
        logger.error(f"Error loading Zillow data: {str(e)}")
        return pd.DataFrame()

def load_competitor_data(filepath='data/raw/retail_locations.csv'):
    """
    Load and process competitor store location data
    """
    try:
        logger.info("Loading competitor data...")
        # Load competitor data
        competitor_data = pd.read_csv(filepath)
        
        # Group by ZIP code and retailer to get store counts
        store_counts = competitor_data.groupby(['zip', 'retailer'])['store_id'].count().reset_index()
        
        # Pivot the data to create columns for each retailer
        store_counts_pivot = store_counts.pivot(
            index='zip',
            columns='retailer',
            values='store_id'
        ).reset_index()
        
        # Fill NaN values with 0 (ZIP codes with no stores)
        store_counts_pivot = store_counts_pivot.fillna(0)
        
        # Rename columns to be more descriptive
        store_counts_pivot.columns.name = None
        retailer_cols = [col for col in store_counts_pivot.columns if col != 'zip']
        store_counts_pivot = store_counts_pivot.rename(columns={
            col: f'{col.lower()}_store_count' for col in retailer_cols
        })
        
        logger.info(f"Processed competitor data for {len(store_counts_pivot)} ZIP codes")
        return store_counts_pivot
        
    except Exception as e:
        logger.error(f"Error loading competitor data: {str(e)}")
        return pd.DataFrame()

def impute_missing_values(df, columns, strategy='median'):
    """
    Impute missing values in specified columns
    """
    imputer = SimpleImputer(strategy=strategy)
    df[columns] = imputer.fit_transform(df[columns])
    return df

def calculate_real_estate_score(df):
    """
    Calculate real estate cost score
    Higher costs → lower score
    """
    logger.info("Calculating real estate cost scores...")
    
    # Create a copy of property values
    property_values = df['median_property_value'].values.reshape(-1, 1)
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Scale property values to 0-1 range
    scaled_values = scaler.fit_transform(property_values)
    
    # Invert the scores (higher costs → lower score)
    real_estate_score = 1 - scaled_values.flatten()
    
    logger.info("Real estate scores calculated")
    return real_estate_score

def calculate_competitor_density_score(df):
    """
    Calculate competitor density score
    Higher density → lower score
    """
    logger.info("Calculating competitor density scores...")
    
    # Get competitor columns
    competitor_cols = [col for col in df.columns if 'store_count' in col]
    
    # Calculate total competitor stores per ZIP
    total_competitors = df[competitor_cols].sum(axis=1).values.reshape(-1, 1)
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Scale competitor counts to 0-1 range
    scaled_density = scaler.fit_transform(total_competitors)
    
    # Invert the scores (higher density → lower score)
    competitor_score = 1 - scaled_density.flatten()
    
    logger.info("Competitor density scores calculated")
    return competitor_score

def calculate_market_saturation_score(df, walmart_locations_path='data/raw/walmart_store_locations.csv'):
    """
    Calculate market saturation score
    Lower saturation → higher score
    """
    logger.info("Calculating market saturation scores...")
    
    try:
        # Load Walmart store locations
        walmart_data = pd.read_csv(walmart_locations_path)
        
        # Count Walmart stores per ZIP
        walmart_counts = walmart_data.groupby('zip_code').size().reset_index(name='walmart_store_count')
        
        # Merge Walmart counts with main dataframe
        df = df.merge(walmart_counts, left_on='zip', right_on='zip_code', how='left')
        df['walmart_store_count'] = df['walmart_store_count'].fillna(0)
        
    except Exception as e:
        logger.warning(f"Could not load Walmart locations: {str(e)}")
        logger.warning("Using sample Walmart data instead")
        # Generate sample Walmart counts if real data is unavailable
        df['walmart_store_count'] = np.random.randint(0, 3, size=len(df))
    
    # Get competitor columns
    competitor_cols = [col for col in df.columns if 'store_count' in col and 'walmart' not in col]
    
    # Calculate total competitor stores per ZIP
    total_competitors = df[competitor_cols].sum(axis=1)
    
    # Calculate saturation ratio (Walmart stores / Total stores)
    # Add small constant to avoid division by zero
    saturation_ratio = df['walmart_store_count'] / (total_competitors + df['walmart_store_count'] + 0.1)
    saturation_ratio = saturation_ratio.values.reshape(-1, 1)
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Scale saturation ratio to 0-1 range
    scaled_saturation = scaler.fit_transform(saturation_ratio)
    
    # Invert the scores (higher saturation → lower score)
    market_score = 1 - scaled_saturation.flatten()
    
    logger.info("Market saturation scores calculated")
    return market_score, df['walmart_store_count']

def merge_additional_data(base_data_path='data/processed/features.csv'):
    """
    Merge real estate and competitor data with existing features
    """
    try:
        logger.info("Starting data merge process...")
        
        # Check if base features exist, if not create them
        if not Path(base_data_path).exists():
            logger.info("Base features file not found, creating it...")
            base_data = create_base_features()
            if base_data is None:
                return None
        else:
            # Load base feature data
            base_data = pd.read_csv(base_data_path)
            
        logger.info(f"Loaded base features for {len(base_data)} locations")
        
        # Load and process new data sources
        real_estate_data = load_zillow_data()
        competitor_data = load_competitor_data()
        
        # Load Walmart locations for ZIP code coordinates
        try:
            walmart_locations = pd.read_csv('data/raw/Walmart_Store_Locations.csv')
            # Get average lat/lng for each ZIP code
            zip_coords = walmart_locations.groupby('zip_code').agg({
                'latitude': 'mean',
                'longitude': 'mean'
            }).reset_index()
            zip_coords.columns = ['zip', 'latitude', 'longitude']
            
            # Merge coordinates with base data
            base_data = base_data.merge(zip_coords[['zip', 'latitude', 'longitude']], 
                                      on='zip', 
                                      how='left')
            
            # For ZIP codes without Walmart stores, use nearby ZIP coordinates
            # This is a simple approximation - in production you'd want to use a proper geocoding service
            base_data['latitude'].fillna(base_data['latitude'].mean(), inplace=True)
            base_data['longitude'].fillna(base_data['longitude'].mean(), inplace=True)
            
            logger.info("Added ZIP code coordinates")
        except Exception as e:
            logger.warning(f"Could not load Walmart locations for coordinates: {str(e)}")
            # Generate sample coordinates if real data is unavailable
            base_data['latitude'] = np.random.uniform(25, 50, size=len(base_data))
            base_data['longitude'] = np.random.uniform(-125, -65, size=len(base_data))
        
        # Merge real estate data
        if not real_estate_data.empty:
            base_data = base_data.merge(real_estate_data[['zip', 'median_property_value']], 
                                      on='zip', 
                                      how='left')
            
            # Impute missing property values
            base_data = impute_missing_values(base_data, ['median_property_value'])
            logger.info("Added real estate data")
        
        # Merge competitor data
        if not competitor_data.empty:
            base_data = base_data.merge(competitor_data, 
                                      on='zip', 
                                      how='left')
            
            # Fill missing competitor counts with 0
            competitor_columns = [col for col in competitor_data.columns if col != 'zip']
            base_data[competitor_columns] = base_data[competitor_columns].fillna(0)
            logger.info("Added competitor data")
        
        # Calculate scores
        base_data['real_estate_score'] = calculate_real_estate_score(base_data)
        base_data['competitor_density_score'] = calculate_competitor_density_score(base_data)
        market_score, walmart_counts = calculate_market_saturation_score(base_data)
        base_data['market_saturation_score'] = market_score
        base_data['walmart_store_count'] = walmart_counts
        
        # Calculate combined opportunity score (simple average of all scores)
        base_data['opportunity_score'] = (
            base_data['real_estate_score'] + 
            base_data['competitor_density_score'] + 
            base_data['market_saturation_score']
        ) / 3
        
        # Save merged dataset
        output_path = 'data/processed/features_enhanced.csv'
        base_data.to_csv(output_path, index=False)
        logger.info(f"Saved enhanced feature set to {output_path}")
        
        # Print summary statistics
        logger.info("\nDataset Summary:")
        logger.info(f"Total locations: {len(base_data)}")
        
        logger.info("\nScore Statistics:")
        for score_col in ['real_estate_score', 'competitor_density_score', 
                         'market_saturation_score', 'opportunity_score']:
            logger.info(f"\n{score_col}:")
            logger.info(f"Mean: {base_data[score_col].mean():.3f}")
            logger.info(f"Median: {base_data[score_col].median():.3f}")
            logger.info(f"Std: {base_data[score_col].std():.3f}")
        
        logger.info("\nCompetitor store counts:")
        competitor_cols = [col for col in base_data.columns if 'store_count' in col]
        for col in competitor_cols:
            total = base_data[col].sum()
            logger.info(f"{col}: {total:,.0f} stores")
        
        logger.info("\nProperty value statistics:")
        logger.info(f"Median property value: ${base_data['median_property_value'].median():,.2f}")
        logger.info(f"Mean property value: ${base_data['median_property_value'].mean():,.2f}")
        
        return base_data
        
    except Exception as e:
        logger.error(f"Error in merge process: {str(e)}")
        return None

def main():
    """
    Main execution function
    """
    try:
        # Create necessary directories if they don't exist
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        
        # Merge all data sources
        merged_data = merge_additional_data()
        
        if merged_data is not None:
            logger.info("Data preprocessing completed successfully")
            logger.info(f"Final dataset shape: {merged_data.shape}")
            logger.info(f"Features available: {', '.join(merged_data.columns)}")
        else:
            logger.error("Data preprocessing failed")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 
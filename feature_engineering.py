import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import geopandas as gpd
from scipy.spatial.distance import cdist
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_store_density(df, radius_km=10):
    """Calculate Walmart store density within different radii"""
    logging.info("Calculating store density features...")
    
    try:
        # Convert lat/lng to radians for BallTree
        coords = np.deg2rad(df[['lat', 'lng']].values)
        tree = BallTree(coords, metric='haversine')
        
        # Calculate density for different radii
        radii = [5, 10, 25, 50]  # kilometers
        density_features = {}
        
        for radius in radii:
            # Convert km to radians (Earth radius ≈ 6371 km)
            radius_rad = radius / 6371.0
            
            # Count stores within radius
            counts = tree.query_radius(coords, r=radius_rad, count_only=True)
            density_features[f'walmart_density_{radius}km'] = counts
            
            # Normalize counts
            density_features[f'walmart_density_{radius}km_normalized'] = (
                counts - counts.mean()
            ) / counts.std()
        
        return pd.DataFrame(density_features, index=df.index)
    
    except Exception as e:
        logging.warning(f"Error in store density calculation: {str(e)}")
        return pd.DataFrame()

def create_interaction_terms(df):
    """Create interaction features from existing ones"""
    logging.info("Creating interaction terms...")
    
    try:
        interactions = {
            'population_employment': df['population'] * df['employment_rate'],
            'density_income': df['population_density'] * df['per_capita_income'],
            'market_employment': df['market_potential'] * df['employment_rate'],
            'income_density': df['per_capita_income'] * df['population_density'],
            'market_density': df['market_potential'] * df['population_density'],
            'employment_density': df['employment_rate'] * df['population_density']
        }
        
        # Normalize interaction terms
        for key in interactions:
            interactions[key] = (interactions[key] - interactions[key].mean()) / interactions[key].std()
        
        return pd.DataFrame(interactions)
    
    except Exception as e:
        logging.warning(f"Error in creating interaction terms: {str(e)}")
        return pd.DataFrame()

def enhance_real_estate_features(df):
    """Enhance real estate features with trends and additional metrics"""
    logging.info("Enhancing real estate features...")
    
    try:
        # Try different encodings for Zillow data
        encodings = ['utf-8', 'latin1', 'cp1252']
        zillow_data = None
        
        for encoding in encodings:
            try:
                zillow_data = pd.read_csv('data/raw/Zillow_Real_Estate_Data.csv', encoding=encoding)
                logging.info(f"Successfully loaded Zillow data with {encoding} encoding")
                break
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        
        if zillow_data is not None:
            # Calculate year-over-year price changes
            price_changes = zillow_data.groupby('zip').agg({
                'rental_price': ['mean', 'std', 'min', 'max'],
                'price_change_yoy': 'mean',
                'commercial_value': 'mean',
                'vacancy_rate': 'mean'
            })
            
            # Merge with main dataset
            df = df.merge(price_changes, on='zip', how='left')
        else:
            # Create synthetic features if data is not available
            logging.warning("Creating synthetic real estate features...")
            df['rental_price_volatility'] = df['rental_price'].std()
            df['rental_price_trend'] = df['rental_price'] / df['rental_price'].mean()
            df['rental_price_min'] = df['rental_price'] * 0.8
            df['rental_price_max'] = df['rental_price'] * 1.2
            df['price_change_yoy'] = 0.03  # Assume 3% average yearly change
            df['commercial_value'] = df['rental_price'] * 12 * 10  # Rough estimate
            df['vacancy_rate'] = 0.05  # Assume 5% average vacancy rate
        
        # Fill missing values with median
        df = df.fillna(df.median())
        
    except Exception as e:
        logging.warning(f"Error in real estate feature enhancement: {str(e)}")
        # Create basic features if all else fails
        df['rental_price_volatility'] = df['rental_price'].std()
        df['rental_price_trend'] = 1.0
        df['price_change_yoy'] = 0.0
    
    return df

def calculate_competitor_density(df):
    """Calculate density of competitor stores"""
    logging.info("Calculating competitor density...")
    
    try:
        # Load competitor data with explicit encoding
        encodings = ['utf-8', 'latin1', 'cp1252']
        competitor_data = {}
        
        for name in ['target', 'costco', 'grocery']:
            for encoding in encodings:
                try:
                    filepath = f'data/raw/{name.capitalize()}_Store_Locations.csv'
                    competitor_data[name] = pd.read_csv(filepath, encoding=encoding)
                    logging.info(f"Successfully loaded {name} locations with {encoding} encoding")
                    break
                except (UnicodeDecodeError, FileNotFoundError):
                    continue
        
        # Calculate competitor densities for available data
        radii = [5, 10, 25]  # kilometers
        
        for name, comp_df in competitor_data.items():
            if comp_df is not None and 'lat' in comp_df.columns and 'lng' in comp_df.columns:
                competitor_coords = np.deg2rad(comp_df[['lat', 'lng']].values)
                store_coords = np.deg2rad(df[['lat', 'lng']].values)
                
                for radius in radii:
                    radius_rad = radius / 6371.0
                    distances = cdist(store_coords, competitor_coords, metric='haversine')
                    df[f'{name}_density_{radius}km'] = (distances <= radius_rad).sum(axis=1)
            else:
                # Create dummy features if coordinates are not available
                for radius in radii:
                    df[f'{name}_density_{radius}km'] = 0
                logging.warning(f"Created dummy density features for {name} stores")
    
    except Exception as e:
        logging.warning(f"Error in competitor density calculation: {str(e)}")
        # Create dummy features for all competitors
        for name in ['target', 'costco', 'grocery']:
            for radius in radii:
                df[f'{name}_density_{radius}km'] = 0
    
    return df

def create_enhanced_features():
    """Main function to create enhanced feature set"""
    logging.info("Starting enhanced feature engineering...")
    
    try:
        # Create output directory
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        
        # Load base features
        df = pd.read_csv('data/processed/location_features.csv')
        original_columns = df.columns.tolist()
        
        # Calculate store density features
        density_features = calculate_store_density(df)
        if not density_features.empty:
            df = pd.concat([df, density_features], axis=1)
        
        # Create interaction terms
        interaction_features = create_interaction_terms(df)
        if not interaction_features.empty:
            df = pd.concat([df, interaction_features], axis=1)
        
        # Enhance real estate features
        df = enhance_real_estate_features(df)
        
        # Calculate competitor density
        df = calculate_competitor_density(df)
        
        # Log new features
        new_features = [col for col in df.columns if col not in original_columns]
        logging.info(f"\nAdded {len(new_features)} new features:")
        for feature in new_features:
            logging.info(f"- {feature}")
        
        # Save enhanced features
        df.to_csv('data/processed/enhanced_features.csv', index=False)
        logging.info(f"\nEnhanced dataset shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logging.error(f"Error in feature engineering: {str(e)}")
        raise

if __name__ == "__main__":
    create_enhanced_features() 
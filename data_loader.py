import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Set up data directory
DATA_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_walmart_locations():
    """Load and clean Walmart store locations data"""
    print("\nLoading Walmart locations...")
    df = pd.read_csv(DATA_DIR / 'Walmart_Store_Locations.csv')
    print("Walmart columns:", df.columns.tolist())
    
    # Clean ZIP codes
    df['ZIP'] = df['zip_code'].astype(str).str.zfill(5)
    
    # Extract relevant columns
    df = df[['ZIP', 'latitude', 'longitude', 'street_address', 'city', 'state']]
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Count stores per ZIP code
    walmart_counts = df.groupby('ZIP').size().reset_index(name='walmart_store_count')
    return walmart_counts

def load_target_locations():
    """Load and clean Target store locations data"""
    print("\nLoading Target locations...")
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252']
        for encoding in encodings:
            try:
                df = pd.read_csv(DATA_DIR / 'target_store_locations.csv', encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
    except Exception as e:
        print(f"Error loading Target locations: {str(e)}")
        return pd.DataFrame(columns=['ZIP', 'target_store_count'])
    
    print("Target columns:", df.columns.tolist())
    
    # Clean ZIP codes
    df['ZIP'] = df['Address.PostalCode'].astype(str).str.zfill(5)
    
    # Extract relevant columns
    df = df[['ZIP', 'Address.Latitude', 'Address.Longitude', 'Address.AddressLine1', 'Address.City', 'Address.Subdivision']]
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Count stores per ZIP code
    target_counts = df.groupby('ZIP').size().reset_index(name='target_store_count')
    return target_counts

def load_zip_data():
    """
    Load ZIP code data from CSV file.
    Returns a DataFrame with ZIP code data.
    """
    print("\nLoading ZIP data...")
    df = pd.read_csv('data/raw/uszips.csv')
    print("ZIP columns:", list(df.columns))
    
    # Clean ZIP codes and county FIPS
    df['zip'] = df['zip'].astype(str).str.zfill(5)
    df['county_fips'] = df['county_fips'].astype(str).str.zfill(5)
    
    # Keep only relevant columns
    columns_to_keep = ['zip', 'lat', 'lng', 'city', 'state_id', 'state_name', 'population', 'density', 'county_fips']
    df = df[columns_to_keep]
    
    return df

def load_wages_population():
    """Load wages and population data from CSV file."""
    df = pd.read_csv('data/raw/wages_population.csv')
    
    # Clean and rename columns
    df = df.rename(columns={
        'Zipcode': 'ZIP',
        'TaxReturnsFiled': 'tax_returns',
        'EstimatedPopulation': 'population',
        'TotalWages': 'total_wages'
    })
    
    # Convert ZIP codes to strings and zero-pad to 5 digits
    df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)
    
    print("\nWages and population data columns:", df.columns.tolist())
    return df

def load_employment_data():
    """Load employment data from CSV file."""
    print("\nLoading employment data...")
    
    # Read the CSV file
    df = pd.read_csv('data/raw/ACSST5Y2022.S2301_data_with_overlays.csv', dtype=str)
    print("Employment data columns:", list(df.columns))
    
    # Extract county FIPS codes from GEO_ID column
    df['county_fips'] = df['GEO_ID'].str.extract(r'US(\d{5})$')[0].astype(str).str.zfill(5)
    
    # Define the columns we want to keep and their new names
    columns_mapping = {
        'S2301_C02_001E': 'labor_force_participation_rate',
        'S2301_C03_001E': 'employment_population_ratio',
        'S2301_C04_001E': 'unemployment_rate'
    }
    
    # Convert employment metrics to float and handle missing values
    for col, new_name in columns_mapping.items():
        df[new_name] = pd.to_numeric(df[col].replace('-', pd.NA), errors='coerce')
    
    # Keep only relevant columns and rows with valid county FIPS codes
    df = df[['county_fips'] + list(columns_mapping.values())]
    df = df[df['county_fips'].notna()]
    
    # Fill missing values with median
    numeric_columns = list(columns_mapping.values())
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    return df

def load_zillow_rentals():
    """Load and clean Zillow rental data"""
    print("\nLoading Zillow rental data...")
    df = pd.read_csv(DATA_DIR / 'ZORI_AllHomesPlusMultifamily_SSA.csv')
    print("Zillow columns:", df.columns.tolist())
    
    # Clean ZIP codes
    df['ZIP'] = df['RegionName'].astype(str).str.zfill(5)
    
    # Get most recent rental price
    # Get all columns that represent dates (start with year)
    price_cols = [col for col in df.columns if col.startswith('20')]
    
    if not price_cols:
        print("Warning: No rental price columns found")
        return pd.DataFrame(columns=['ZIP', 'rental_price'])
    
    # Get the most recent date with data
    latest_prices = df[price_cols].iloc[:, -12:].mean(axis=1)  # Use average of last 12 months
    
    # Create new dataframe with ZIP and rental price
    result_df = pd.DataFrame({
        'ZIP': df['ZIP'],
        'rental_price': latest_prices
    })
    
    # Remove any invalid prices
    result_df = result_df[result_df['rental_price'].notna() & (result_df['rental_price'] > 0)]
    
    print(f"Loaded {len(result_df)} valid rental prices")
    print(f"Rental price statistics:\n{result_df['rental_price'].describe()}")
    
    return result_df

def normalize_features(df, columns_to_normalize):
    """Normalize specified columns using StandardScaler"""
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def merge_all_datasets():
    """Merge all datasets into a single DataFrame."""
    # Load all datasets
    walmart_df = load_walmart_locations()
    target_df = load_target_locations()
    wages_df = load_wages_population()
    zillow_df = load_zillow_rentals()
    zip_df = load_zip_data()
    employment_df = load_employment_data()

    # Print shapes before merging
    print("\nDataset shapes before merging:")
    print(f"Walmart: {walmart_df.shape}")
    print(f"Target: {target_df.shape}")
    print(f"Wages: {wages_df.shape}")
    print(f"Zillow: {zillow_df.shape}")
    print(f"ZIP: {zip_df.shape}")
    print(f"Employment: {employment_df.shape}")

    # Start with ZIP data as the base
    final_df = zip_df.copy()

    # Merge with Walmart locations
    final_df['has_walmart'] = final_df['zip'].isin(walmart_df['ZIP'])

    # Merge with Target locations
    final_df['has_target'] = final_df['zip'].isin(target_df['ZIP'])

    # Merge with wages data
    final_df = final_df.merge(wages_df[['ZIP', 'tax_returns', 'total_wages']], 
                             left_on='zip', 
                             right_on='ZIP', 
                             how='left')
    final_df.drop('ZIP', axis=1, inplace=True)

    # Merge with Zillow rental data
    final_df = final_df.merge(zillow_df[['ZIP', 'rental_price']], 
                             left_on='zip', 
                             right_on='ZIP', 
                             how='left')
    final_df.drop('ZIP', axis=1, inplace=True)

    # Merge with employment data using county_fips
    final_df = final_df.merge(employment_df, on='county_fips', how='left')

    # Fill missing values
    final_df['has_walmart'] = final_df['has_walmart'].fillna(False)
    final_df['has_target'] = final_df['has_target'].fillna(False)
    final_df['rental_price'] = final_df['rental_price'].fillna(final_df['rental_price'].median())
    final_df['tax_returns'] = final_df['tax_returns'].fillna(final_df['tax_returns'].median())
    final_df['total_wages'] = final_df['total_wages'].fillna(final_df['total_wages'].median())
    final_df['labor_force_participation_rate'] = final_df['labor_force_participation_rate'].fillna(final_df['labor_force_participation_rate'].median())
    final_df['employment_population_ratio'] = final_df['employment_population_ratio'].fillna(final_df['employment_population_ratio'].median())
    final_df['unemployment_rate'] = final_df['unemployment_rate'].fillna(final_df['unemployment_rate'].median())

    # Print shape after merging
    print("\nFinal dataset shape:", final_df.shape)
    print("Final dataset columns:", final_df.columns.tolist())

    return final_df

def calculate_haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between points using vectorized operations."""
    R = 6371  # Earth's radius in kilometers
    
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1[:, np.newaxis]  # Shape: (n_zips, n_walmarts)
    dlon = lon2 - lon1[:, np.newaxis]  # Shape: (n_zips, n_walmarts)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1)[:, np.newaxis] * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = R * c  # Shape: (n_zips, n_walmarts)
    
    return distances

def create_location_features(merged_df, walmart_locations):
    """
    Create a feature matrix for location attractiveness based on ZIP codes.
    """
    print("\nCreating location attractiveness features...")

    # Create a new DataFrame with only the necessary columns
    features_df = pd.DataFrame({
        'zip': merged_df['zip'],
        'lat': merged_df['lat'],
        'lng': merged_df['lng'],
        'population': merged_df['population'],
        'population_density': merged_df['density'],
        'total_wages': merged_df['total_wages'],
        'employment_rate': merged_df['employment_population_ratio'],
        'rental_price': merged_df['rental_price'],
        'has_target': merged_df['has_target'],
        'labor_force_participation': merged_df['labor_force_participation_rate'],
        'unemployment_rate': merged_df['unemployment_rate']
    })

    # Print Walmart locations info for debugging
    print("\nWalmart locations shape:", walmart_locations.shape)
    print("Walmart locations columns:", walmart_locations.columns.tolist())
    
    # Clean Walmart locations data
    walmart_locations = walmart_locations.dropna(subset=['latitude', 'longitude'])
    print("Walmart locations shape after cleaning:", walmart_locations.shape)

    # Calculate distances to nearest Walmart stores (vectorized)
    print("\nCalculating distances to nearest Walmart stores (vectorized)...")
    print("Number of ZIP codes:", len(features_df))
    print("Number of ZIP codes with coordinates:", len(features_df.dropna(subset=['lat', 'lng'])))
    print("Number of Walmart stores with coordinates:", len(walmart_locations))

    # Calculate distances using vectorized operations
    distances = calculate_haversine_distance_vectorized(
        features_df['lat'].values,
        features_df['lng'].values,
        walmart_locations['latitude'].values,
        walmart_locations['longitude'].values
    )
    
    print("Shape of distances matrix:", distances.shape)
    min_distances = np.min(distances, axis=1)
    print("Length of min_distances:", len(min_distances))
    
    # Replace infinite distances with a large value (1000 km)
    min_distances = np.where(np.isinf(min_distances), 1000, min_distances)
    features_df['distance_to_nearest_walmart'] = min_distances

    # Calculate competition density (number of competitors within 50km)
    competition_mask = distances <= 50  # True for distances <= 50km
    features_df['walmart_stores_within_50km'] = competition_mask.sum(axis=1)
    
    # Calculate per capita income and related metrics
    features_df['per_capita_income'] = features_df['total_wages'] / features_df['population'].replace(0, np.nan)
    features_df['per_capita_income'] = features_df['per_capita_income'].replace([np.inf, -np.inf], np.nan)
    features_df['per_capita_income'] = features_df['per_capita_income'].fillna(features_df['per_capita_income'].median())

    # Calculate market potential (population * per capita income)
    features_df['market_potential'] = features_df['population'] * features_df['per_capita_income']
    
    # Calculate employment health score
    features_df['employment_health_score'] = (
        features_df['employment_rate'] * 0.4 +
        features_df['labor_force_participation'] * 0.4 -
        features_df['unemployment_rate'] * 0.2
    )

    # Fill missing values with appropriate statistics
    features_df['population_density'] = features_df['population_density'].fillna(0)
    features_df['employment_rate'] = features_df['employment_rate'].fillna(features_df['employment_rate'].median())
    features_df['rental_price'] = features_df['rental_price'].fillna(features_df['rental_price'].median())
    features_df['employment_health_score'] = features_df['employment_health_score'].fillna(features_df['employment_health_score'].median())
    features_df['market_potential'] = features_df['market_potential'].fillna(features_df['market_potential'].median())

    # Save raw values before normalization
    raw_columns = [
        'population_density',
        'per_capita_income',
        'employment_rate',
        'rental_price',
        'distance_to_nearest_walmart',
        'walmart_stores_within_50km',
        'market_potential',
        'employment_health_score'
    ]
    
    for col in raw_columns:
        features_df[f'{col}_raw'] = features_df[col].copy()

    # Print feature statistics before normalization
    print("\nFeature statistics before normalization:")
    print(features_df[raw_columns].describe())

    # Normalize features
    scaler = StandardScaler()
    features_df[raw_columns] = scaler.fit_transform(features_df[raw_columns])

    # Print feature statistics after normalization
    print("\nFeature statistics after normalization:")
    print(features_df[raw_columns].describe())

    print("\nFeature matrix shape:", features_df.shape)
    print("Feature matrix columns:", features_df.columns.tolist())

    # Save features to CSV
    output_dir = os.path.join('data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    features_df.to_csv(os.path.join(output_dir, 'location_features.csv'), index=False)
    print("\nFeatures saved to:", os.path.join(output_dir, 'location_features.csv'))

    return features_df

if __name__ == "__main__":
    merged_df = merge_all_datasets()
    
    # Load full Walmart locations data for distance calculations
    walmart_locations = pd.read_csv(DATA_DIR / 'Walmart_Store_Locations.csv')
    
    # Create location features
    features_df = create_location_features(merged_df, walmart_locations) 
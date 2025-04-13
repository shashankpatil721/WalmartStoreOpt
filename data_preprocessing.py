import pandas as pd
import numpy as np
from pathlib import Path

# Set data paths
DATA_DIR = Path('data/raw')

def load_walmart_locations():
    """Load and clean Walmart store locations data"""
    print("\nLoading Walmart locations...")
    df = pd.read_csv(DATA_DIR / 'Walmart_Store_Locations.csv')
    print("Walmart columns:", df.columns.tolist())
    
    # Rename columns to match our schema
    df = df.rename(columns={
        'zip_code': 'ZIP',
        'street_address': 'STREET_ADDRESS',
        'city': 'CITY',
        'state': 'STATE',
        'latitude': 'LATITUDE',
        'longitude': 'LONGITUDE'
    })
    
    # Clean ZIP codes
    df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)
    
    # Select relevant columns
    relevant_cols = ['ZIP', 'STREET_ADDRESS', 'CITY', 'STATE', 'LATITUDE', 'LONGITUDE']
    df = df[relevant_cols]
    
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
    
    # Rename columns to match our schema
    df = df.rename(columns={
        'Address.PostalCode': 'ZIP',
        'Address.AddressLine1': 'STREET_ADDRESS',
        'Address.City': 'CITY',
        'Address.Subdivision': 'STATE',
        'Address.Latitude': 'LATITUDE',
        'Address.Longitude': 'LONGITUDE'
    })
    
    # Clean ZIP codes
    df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Count stores per ZIP code
    target_counts = df.groupby('ZIP').size().reset_index(name='target_store_count')
    return target_counts

def load_wages_population():
    """Load and clean wages and population data"""
    print("\nLoading wages and population...")
    df = pd.read_csv(DATA_DIR / 'wages_population.csv')
    print("Wages columns:", df.columns.tolist())
    
    # Rename columns to match our schema
    df = df.rename(columns={
        'Zipcode': 'ZIP',
        'City': 'CITY',
        'State': 'STATE',
        'TaxReturnsFiled': 'tax_returns',
        'EstimatedPopulation': 'population',
        'TotalWages': 'total_wages',
        'County': 'county'
    })
    
    # Clean ZIP codes
    df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)
    
    # Handle missing numerical values with median imputation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    return df

def load_zillow_rentals():
    """Load and clean Zillow rental data"""
    print("\nLoading Zillow rentals...")
    df = pd.read_csv(DATA_DIR / 'ZORI_AllHomesPlusMultifamily_SSA.csv')
    print("Zillow columns:", df.columns.tolist())
    
    # Get the most recent date column (last column excluding non-date columns)
    date_columns = [col for col in df.columns if '-' in col]  # Date columns contain '-'
    most_recent_date = sorted(date_columns)[-1]  # Get the last date
    
    # Clean ZIP codes and get the most recent rental price
    df = df.rename(columns={'RegionName': 'ZIP'})
    df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)
    df['rental_price'] = pd.to_numeric(df[most_recent_date], errors='coerce')
    
    # Select relevant columns
    df = df[['ZIP', 'rental_price']]
    
    # Handle missing values with median imputation
    df['rental_price'] = df['rental_price'].fillna(df['rental_price'].median())
    
    return df

def load_zip_data():
    """Load and clean ZIP code database"""
    print("\nLoading ZIP data...")
    df = pd.read_csv(DATA_DIR / 'uszips.csv')
    print("ZIP data columns:", df.columns.tolist())
    
    # Rename columns
    df = df.rename(columns={
        'zip': 'ZIP',
        'postal_code': 'ZIP'  # Alternative name
    })
    
    # Clean ZIP codes
    df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)
    
    # Select relevant columns
    relevant_cols = ['ZIP', 'lat', 'lng', 'city', 'state_id', 'population', 'density', 'county_name']
    df = df[relevant_cols]
    
    return df

def load_employment_data():
    """Load and clean employment data"""
    print("\nLoading employment data...")
    df = pd.read_csv(DATA_DIR / 'ACSST5Y2022.S2301_data_with_overlays.csv')
    print("Employment columns:", df.columns.tolist())
    
    # Rename columns
    df = df.rename(columns={
        'GEO_ID': 'ZIP'
    })
    
    # Clean ZIP codes - extract ZIP from GEO_ID if needed
    df['ZIP'] = df['ZIP'].astype(str).str.extract(r'(\d{5})').fillna(df['ZIP'])
    df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)
    
    # Handle missing numerical values with median imputation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    return df

def merge_all_datasets():
    """Merge all datasets by ZIP code"""
    # Load all datasets
    walmart_df = load_walmart_locations()
    target_df = load_target_locations()
    wages_df = load_wages_population()
    zillow_df = load_zillow_rentals()
    zip_df = load_zip_data()
    employment_df = load_employment_data()
    
    print("\nDataset shapes before merging:")
    print(f"Walmart: {walmart_df.shape}")
    print(f"Target: {target_df.shape}")
    print(f"Wages: {wages_df.shape}")
    print(f"Zillow: {zillow_df.shape}")
    print(f"ZIP: {zip_df.shape}")
    print(f"Employment: {employment_df.shape}")
    
    # Merge all datasets
    merged_df = zip_df.merge(walmart_df, on='ZIP', how='left')\
                     .merge(target_df, on='ZIP', how='left')\
                     .merge(wages_df, on='ZIP', how='left')\
                     .merge(zillow_df, on='ZIP', how='left')\
                     .merge(employment_df, on='ZIP', how='left')
    
    # Fill missing values for store counts with 0
    merged_df['walmart_store_count'] = merged_df['walmart_store_count'].fillna(0)
    merged_df['target_store_count'] = merged_df['target_store_count'].fillna(0)
    
    # Handle any remaining missing values
    numerical_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numerical_cols] = merged_df[numerical_cols].fillna(merged_df[numerical_cols].median())
    
    return merged_df

if __name__ == "__main__":
    # Process and merge all datasets
    final_df = merge_all_datasets()
    
    # Save the merged dataset
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    final_df.to_csv(output_dir / 'merged_dataset.csv', index=False)
    
    print("\nData processing completed. Merged dataset saved to data/processed/merged_dataset.csv")
    print(f"Final dataset shape: {final_df.shape}")
    print("\nSample of the merged dataset:")
    print(final_df.head())
    print("\nColumns in the merged dataset:")
    print(final_df.columns.tolist()) 
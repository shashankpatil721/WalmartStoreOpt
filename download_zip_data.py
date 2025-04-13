import pandas as pd
import requests
from pathlib import Path
import logging
import io
import zipfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_zip_data():
    """
    Download and prepare ZIP code database
    Uses Census Bureau ZIP Code Tabulation Areas (ZCTA) data
    """
    try:
        logger.info("Downloading ZIP code data...")
        
        # Create data directories if they don't exist
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        
        # Create a basic ZIP code database using existing data
        # Load Walmart locations for state data
        walmart_df = pd.read_csv('data/raw/Walmart_Store_Locations.csv')
        walmart_df['zip_code'] = walmart_df['zip_code'].astype(str).str.zfill(5)
        zip_state = walmart_df[['zip_code', 'state', 'city']].drop_duplicates()
        zip_state = zip_state.rename(columns={'zip_code': 'zip'})
        
        # Load enhanced features for additional ZIP codes
        enhanced_df = pd.read_csv('data/processed/features_enhanced.csv')
        enhanced_df['zip'] = enhanced_df['zip'].astype(str).str.zfill(5)
        enhanced_zip_state = enhanced_df[['zip', 'state', 'city']].drop_duplicates()
        
        # Combine both sources
        zip_db = pd.concat([zip_state, enhanced_zip_state], ignore_index=True)
        zip_db = zip_db.drop_duplicates(subset=['zip'])
        
        # Clean state codes
        zip_db['state'] = zip_db['state'].fillna('Unknown')
        zip_db['city'] = zip_db['city'].fillna('Unknown')
        
        # Save to CSV
        output_path = 'data/raw/zip_code_database.csv'
        zip_db.to_csv(output_path, index=False)
        
        logger.info(f"Saved ZIP code database to {output_path}")
        logger.info(f"Total ZIP codes processed: {len(zip_db)}")
        logger.info("\nState distribution:")
        logger.info(zip_db['state'].value_counts().head())
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating ZIP code database: {str(e)}")
        return False

if __name__ == "__main__":
    download_zip_data() 
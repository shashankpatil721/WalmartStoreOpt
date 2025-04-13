import pandas as pd
import requests
from pathlib import Path
import logging
import zipfile
import io
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_zillow_data():
    """
    Download Zillow Home Value Index data
    """
    try:
        logger.info("Downloading Zillow data...")
        
        # Create data directories if they don't exist
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        
        # Zillow ZHVI data URL (Single Family Homes)
        url = "https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
        
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the file
        output_path = 'data/raw/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        logger.info(f"Saved Zillow data to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading Zillow data: {str(e)}")
        return False

def create_sample_competitor_data():
    """
    Create a sample competitor dataset since we don't have direct access to competitor data
    This would normally come from a real data source
    """
    try:
        logger.info("Creating sample competitor data...")
        
        # Load ZIP codes from Zillow data to ensure we use real ZIP codes
        zillow_data = pd.read_csv('data/raw/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
        zip_codes = zillow_data['RegionName'].unique()
        
        # Create sample data for major retailers
        retailers = ['Target', 'Costco', 'Sams_Club', 'Kroger', 'Whole_Foods']
        
        # Generate random store locations
        np.random.seed(42)  # for reproducibility
        
        data = []
        store_id = 1
        
        for retailer in retailers:
            # Select random ZIP codes for each retailer
            n_stores = np.random.randint(500, 2000)  # Random number of stores for each retailer
            selected_zips = np.random.choice(zip_codes, size=n_stores, replace=True)
            
            for zip_code in selected_zips:
                data.append({
                    'store_id': store_id,
                    'retailer': retailer,
                    'zip': zip_code
                })
                store_id += 1
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        output_path = 'data/raw/retail_locations.csv'
        df.to_csv(output_path, index=False)
        
        logger.info(f"Created sample competitor data with {len(df)} store locations")
        logger.info(f"Saved competitor data to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating competitor data: {str(e)}")
        return False

def main():
    """
    Main execution function
    """
    try:
        # Download Zillow data
        zillow_success = download_zillow_data()
        
        if zillow_success:
            # Create sample competitor data
            competitor_success = create_sample_competitor_data()
            
            if competitor_success:
                logger.info("All data downloaded successfully")
            else:
                logger.error("Failed to create competitor data")
        else:
            logger.error("Failed to download Zillow data")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 
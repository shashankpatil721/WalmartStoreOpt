import pandas as pd
import numpy as np
from pathlib import Path
import logging
from shapely.geometry import Point
import geopandas as gpd
from shapely.geometry import Point
import requests
import json
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_state_from_coordinates(lat, lng):
    """Get state from coordinates using Census Bureau's geocoding API."""
    try:
        url = f"https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
        params = {
            "x": lng,
            "y": lat,
            "benchmark": "Public_AR_Current",
            "vintage": "Current_Current",
            "layers": "States",
            "format": "json"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('result', {}).get('geographies', {}).get('States', []):
                state = data['result']['geographies']['States'][0]['STATE']
                return state
        return None
    except Exception as e:
        logger.error(f"Error geocoding coordinates ({lat}, {lng}): {str(e)}")
        return None

def update_unknown_states():
    """Update unknown states using geocoding."""
    logger.info("Starting state update process...")
    
    # Load predictions with unknown states
    predictions_file = 'data/processed/top_optimal_locations.csv'
    if not Path(predictions_file).exists():
        logger.error(f"Predictions file not found: {predictions_file}")
        return
    
    predictions = pd.read_csv(predictions_file)
    unknown_mask = predictions['state'] == 'Unknown'
    unknown_locations = predictions[unknown_mask].copy()
    
    logger.info(f"Found {len(unknown_locations)} locations with unknown states")
    
    # Process in batches to avoid rate limiting
    batch_size = 10
    delay = 1  # seconds between batches
    
    updated_states = []
    total_batches = len(unknown_locations) // batch_size + 1
    
    for i in range(0, len(unknown_locations), batch_size):
        batch = unknown_locations.iloc[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} of {total_batches}")
        
        batch_states = []
        for _, row in batch.iterrows():
            state = get_state_from_coordinates(row['lat'], row['lng'])
            batch_states.append(state)
            time.sleep(0.1)  # Small delay between requests
        
        updated_states.extend(batch_states)
        time.sleep(delay)
    
    # Update states in the original DataFrame
    unknown_locations['state'] = updated_states
    unknown_locations['state'] = unknown_locations['state'].fillna('Unknown')
    
    # Count successful updates
    successful_updates = sum(1 for state in updated_states if state is not None)
    logger.info(f"\nSuccessfully updated {successful_updates} out of {len(unknown_locations)} unknown states")
    
    # Save updated unknown locations
    unknown_file = 'data/processed/geocoded_locations.csv'
    unknown_locations.to_csv(unknown_file, index=False)
    logger.info(f"Saved geocoded locations to: {unknown_file}")
    
    # Update main predictions file
    predictions.loc[unknown_mask, 'state'] = unknown_locations['state']
    predictions.to_csv(predictions_file, index=False)
    logger.info(f"Updated main predictions file: {predictions_file}")
    
    # Print summary of updates
    state_counts = unknown_locations['state'].value_counts()
    logger.info("\nState distribution after geocoding:")
    for state, count in state_counts.items():
        logger.info(f"{state}: {count} locations")

if __name__ == "__main__":
    update_unknown_states() 
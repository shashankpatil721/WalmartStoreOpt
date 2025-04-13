import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_state_data():
    """Analyze and validate state data in predictions."""
    logger.info("Starting state data analysis...")
    
    # Load predicted locations
    predictions_file = 'data/processed/top_optimal_locations.csv'
    if not Path(predictions_file).exists():
        logger.error(f"Predictions file not found: {predictions_file}")
        return
        
    predictions = pd.read_csv(predictions_file)
    
    # Load ZIP code database for reference
    zip_db_file = 'data/raw/zip_code_database.csv'
    if not Path(zip_db_file).exists():
        logger.error(f"ZIP code database not found: {zip_db_file}")
        return
        
    zip_db = pd.read_csv(zip_db_file)
    zip_db['zip'] = zip_db['zip'].astype(str).str.zfill(5)
    
    # Analyze state distribution
    total_locations = len(predictions)
    state_counts = predictions['state'].value_counts()
    unknown_count = state_counts.get('Unknown', 0)
    unknown_percentage = (unknown_count / total_locations) * 100
    
    logger.info("\nState Distribution Analysis:")
    logger.info(f"Total locations analyzed: {total_locations}")
    logger.info(f"Number of unique states: {len(state_counts)}")
    logger.info(f"Unknown state count: {unknown_count} ({unknown_percentage:.2f}%)")
    
    # Print top 10 states by location count
    logger.info("\nTop 10 states by location count:")
    for state, count in state_counts.head(10).items():
        percentage = (count / total_locations) * 100
        logger.info(f"{state}: {count} locations ({percentage:.2f}%)")
    
    # Analyze locations with unknown states
    unknown_locations = predictions[predictions['state'] == 'Unknown']
    if not unknown_locations.empty:
        logger.info("\nAnalyzing locations with unknown states:")
        logger.info(f"Number of unknown state locations: {len(unknown_locations)}")
        
        # Check if these ZIPs exist in the ZIP code database
        unknown_zips = unknown_locations['zip'].unique()
        matching_zips = zip_db[zip_db['zip'].isin(unknown_zips)]
        
        if not matching_zips.empty:
            logger.info(f"\nFound {len(matching_zips)} matching ZIPs in database:")
            for _, row in matching_zips.iterrows():
                logger.info(f"ZIP: {row['zip']}, State: {row['state']}")
        
        # Analyze geographic distribution of unknown locations
        logger.info("\nGeographic distribution of unknown locations:")
        logger.info("\nLatitude range:")
        logger.info(f"Min: {unknown_locations['lat'].min():.4f}")
        logger.info(f"Max: {unknown_locations['lat'].max():.4f}")
        logger.info("\nLongitude range:")
        logger.info(f"Min: {unknown_locations['lng'].min():.4f}")
        logger.info(f"Max: {unknown_locations['lng'].max():.4f}")
        
        # Save unknown locations for further analysis
        unknown_file = 'data/processed/unknown_state_locations.csv'
        unknown_locations.to_csv(unknown_file, index=False)
        logger.info(f"\nUnknown state locations saved to: {unknown_file}")
    
    # Generate recommendations for improvement
    logger.info("\nRecommendations for improvement:")
    if unknown_count > 0:
        logger.info("1. Update ZIP code database with missing locations")
        logger.info("2. Implement geocoding to determine states from coordinates")
        logger.info("3. Cross-reference with USPS database for verification")
        logger.info("4. Consider manual verification for high-priority locations")
    
    # Save analysis results
    analysis_results = {
        'total_locations': int(total_locations),
        'unique_states': int(len(state_counts)),
        'unknown_count': int(unknown_count),
        'unknown_percentage': float(unknown_percentage),
        'state_distribution': {
            state: int(count) for state, count in state_counts.items()
        }
    }
    
    results_file = 'data/processed/state_data_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    logger.info(f"\nAnalysis results saved to: {results_file}")

if __name__ == "__main__":
    analyze_state_data() 
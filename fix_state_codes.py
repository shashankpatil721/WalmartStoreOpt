import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FIPS state codes to state abbreviations mapping
FIPS_TO_STATE = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
    '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
    '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
    '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
    '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
    '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
    '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
    '56': 'WY', '72': 'PR', '78': 'VI'
}

def fix_state_codes():
    """Convert FIPS state codes to state abbreviations."""
    logger.info("Starting state code conversion...")
    
    # Load predictions
    predictions_file = 'data/processed/top_optimal_locations.csv'
    predictions = pd.read_csv(predictions_file)
    
    # Count states before conversion
    logger.info("\nState distribution before conversion:")
    state_counts_before = predictions['state'].value_counts()
    for state, count in state_counts_before.head(10).items():
        logger.info(f"{state}: {count} locations")
    
    # Convert FIPS codes to state abbreviations
    predictions['state'] = predictions['state'].map(lambda x: FIPS_TO_STATE.get(str(x), x))
    
    # Save updated predictions
    predictions.to_csv(predictions_file, index=False)
    
    # Count states after conversion
    logger.info("\nState distribution after conversion:")
    state_counts_after = predictions['state'].value_counts()
    for state, count in state_counts_after.head(10).items():
        logger.info(f"{state}: {count} locations")
    
    logger.info(f"\nUpdated {predictions_file} with state abbreviations")
    
    # Save state distribution for analysis
    state_dist = pd.DataFrame({
        'state': state_counts_after.index,
        'count': state_counts_after.values,
        'percentage': (state_counts_after.values / len(predictions) * 100).round(2)
    })
    state_dist.to_csv('data/processed/state_distribution.csv', index=False)
    logger.info("Saved state distribution to: data/processed/state_distribution.csv")

if __name__ == "__main__":
    fix_state_codes() 
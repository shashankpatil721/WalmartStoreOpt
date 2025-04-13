import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State mapping for special cases
SPECIAL_LOCATIONS = {
    '96799': 'AS',  # American Samoa
    '99693': 'AK',  # Alaska
    '99606': 'AK',  # Alaska
    '99566': 'AK',  # Alaska
    '93042': 'CA',  # Channel Islands, California
    '87313': 'NM',  # Navajo Nation, New Mexico
    '86520': 'AZ',  # Navajo Nation, Arizona
    '87323': 'NM',  # Navajo Nation, New Mexico
    '70082': 'LA',  # Louisiana
    '71674': 'AR',  # Arkansas
    '65744': 'MO',  # Missouri
    '62311': 'IL',  # Illinois
    '57256': 'SD',  # South Dakota
    '25030': 'WV',  # West Virginia
    '39162': 'MS',  # Mississippi
    '66932': 'KS',  # Kansas
    '67452': 'KS',  # Kansas
    '59332': 'MT',  # Montana
    '58466': 'ND',  # North Dakota
    '97920': 'OR'   # Oregon
}

def fix_locations_and_population():
    """Fix unknown locations and adjust population calculations."""
    logger.info("Starting location and population fixes...")
    
    # Load predictions
    predictions_file = 'data/processed/top_optimal_locations.csv'
    if not Path(predictions_file).exists():
        logger.error(f"Predictions file not found: {predictions_file}")
        return
        
    predictions = pd.read_csv(predictions_file)
    
    # Store original state counts
    original_unknown = len(predictions[predictions['state'] == 'Unknown'])
    logger.info(f"Original unknown locations: {original_unknown}")
    
    # Fix state assignments for special cases
    for zip_code, state in SPECIAL_LOCATIONS.items():
        mask = predictions['zip'] == zip_code
        if any(mask):
            predictions.loc[mask, 'state'] = state
            logger.info(f"Updated ZIP {zip_code} to state {state}")
    
    # Adjust population calculations
    logger.info("\nAdjusting population calculations...")
    
    # Fill missing population values with ZIP code averages
    state_pop_avg = predictions.groupby('state')['population'].transform('mean')
    predictions['population'] = predictions['population'].fillna(state_pop_avg)
    
    # Adjust unrealistic per capita income values
    income_median = predictions['per_capita_income'].median()
    income_std = predictions['per_capita_income'].std()
    income_upper_bound = income_median + 3 * income_std
    
    high_income_mask = predictions['per_capita_income'] > income_upper_bound
    predictions.loc[high_income_mask, 'per_capita_income'] = income_median
    
    logger.info(f"Adjusted {sum(high_income_mask)} unrealistic income values")
    
    # Calculate market potential using adjusted population and income
    predictions['market_potential'] = (
        predictions['population'] * 
        predictions['per_capita_income'] * 
        (1 + predictions['employment_rate'] / 100)
    )
    
    # Update prediction scores based on adjusted metrics
    max_market_potential = predictions['market_potential'].max()
    predictions['market_potential_normalized'] = predictions['market_potential'] / max_market_potential
    
    # Recalculate prediction scores with adjusted weights
    predictions['prediction_score'] = (
        0.3 * predictions['market_potential_normalized'] +
        0.2 * (predictions['employment_rate'] / 100) +
        0.2 * (predictions['labor_force_participation'] / 100) +
        0.15 * (1 - predictions['unemployment_rate'] / 100) +
        0.15 * (1 / np.log1p(predictions['distance_to_nearest_walmart']))
    )
    
    # Sort by new prediction scores
    predictions = predictions.sort_values('prediction_score', ascending=False)
    predictions['rank'] = range(1, len(predictions) + 1)
    
    # Save updated predictions
    predictions.to_csv(predictions_file, index=False)
    logger.info(f"\nSaved updated predictions to {predictions_file}")
    
    # Generate summary statistics
    remaining_unknown = len(predictions[predictions['state'] == 'Unknown'])
    logger.info(f"\nRemaining unknown locations: {remaining_unknown}")
    
    logger.info("\nTop 10 states by location count:")
    state_counts = predictions['state'].value_counts()
    for state, count in state_counts.head(10).items():
        percentage = (count / len(predictions)) * 100
        logger.info(f"{state}: {count} locations ({percentage:.2f}%)")
    
    # Save state distribution
    state_dist = pd.DataFrame({
        'state': state_counts.index,
        'count': state_counts.values,
        'percentage': (state_counts.values / len(predictions) * 100).round(2)
    })
    state_dist.to_csv('data/processed/state_distribution.csv', index=False)
    logger.info("\nSaved updated state distribution to data/processed/state_distribution.csv")

if __name__ == "__main__":
    fix_locations_and_population() 
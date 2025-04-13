import pandas as pd
import numpy as np
from train_model import load_and_prepare_data, train_and_evaluate_model

def predict_optimal_locations():
    """Predict and rank optimal locations for new Walmart stores."""
    print("Loading data and training model...")
    
    # Load all data
    df = pd.read_csv('data/processed/location_features.csv')
    enhanced_features = pd.read_csv('data/processed/features_enhanced.csv')
    X, y, feature_columns = load_and_prepare_data()
    
    # Train the model on all data
    best_model, _ = train_and_evaluate_model(X, y, feature_columns)
    
    # Get predictions for all locations
    predictions = best_model.predict_proba(X)[:, 1]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'zip': df['zip'].astype(str).str.zfill(5),
        'lat': df['lat'],
        'lng': df['lng'],
        'prediction_score': predictions,
        'population': df['population'],
        'population_density': df['population_density_raw'],
        'per_capita_income': df['per_capita_income_raw'],
        'employment_rate': df['employment_rate_raw'],
        'unemployment_rate': df['unemployment_rate'],
        'labor_force_participation': df['labor_force_participation'],
        'market_potential': df['market_potential_raw'],
        'rental_price': df['rental_price_raw'],
        'distance_to_nearest_walmart': df['distance_to_nearest_walmart_raw'],
        'has_target': df['has_target']
    })
    
    # Convert ZIP codes to same format in both DataFrames
    enhanced_features['zip'] = enhanced_features['zip'].astype(str).str.zfill(5)
    
    # Merge with enhanced features to get state and city information
    results_df = results_df.merge(
        enhanced_features[['zip', 'state', 'city']],
        on='zip',
        how='left'
    )
    
    # Load ZIP code to state mapping
    zip_state_map = pd.read_csv('data/raw/zip_code_database.csv')[['zip', 'state']]
    zip_state_map['zip'] = zip_state_map['zip'].astype(str).str.zfill(5)
    
    # Fill missing state information using ZIP code database
    missing_state_mask = results_df['state'].isna() | (results_df['state'] == 'Unknown')
    if missing_state_mask.any():
        # Merge with ZIP code database for missing states
        missing_states = results_df[missing_state_mask].merge(
            zip_state_map,
            on='zip',
            how='left',
            suffixes=('_orig', '')
        )
        # Update missing states
        results_df.loc[missing_state_mask, 'state'] = missing_states['state']
        
    # Fill any remaining missing values
    results_df['state'] = results_df['state'].fillna('Unknown')
    results_df['city'] = results_df['city'].fillna('Unknown')
    
    # Initialize parameters for adaptive filtering
    target_count = 4654  # Target number of locations
    min_distance = 30  # Start with 30km minimum distance
    max_distance = 70  # Maximum distance to consider
    step = 5  # Step size for adjusting distance
    current_distance = min_distance
    
    while current_distance <= max_distance:
        # Filter locations based on current distance threshold
        new_locations = results_df[results_df['distance_to_nearest_walmart'] > current_distance].copy()
        
        # Sort by prediction score
        new_locations = new_locations.sort_values('prediction_score', ascending=False)
        
        if len(new_locations) >= target_count:
            print(f"Found {len(new_locations)} candidate locations with {current_distance}km minimum distance")
            break
        
        current_distance -= step
        if current_distance < min_distance:
            # If we can't find enough locations, take the top N regardless of distance
            new_locations = results_df.copy()
            new_locations = new_locations.sort_values('prediction_score', ascending=False)
            print(f"Using all locations sorted by prediction score")
            break
    
    # Select exactly target_count locations
    top_locations = new_locations.head(target_count).copy()
    
    # Add rank column
    top_locations.loc[:, 'rank'] = range(1, len(top_locations) + 1)
    
    # Format numeric columns
    numeric_cols = ['prediction_score', 'population', 'population_density', 'per_capita_income',
                   'employment_rate', 'unemployment_rate', 'labor_force_participation',
                   'market_potential', 'rental_price', 'distance_to_nearest_walmart']
    
    for col in numeric_cols:
        if col in top_locations.columns:
            top_locations.loc[:, col] = top_locations[col].round(2)
    
    # Reorder columns
    columns_order = [
        'rank', 'zip', 'state', 'city', 'lat', 'lng', 'prediction_score',
        'population', 'population_density', 'per_capita_income',
        'employment_rate', 'unemployment_rate', 'labor_force_participation',
        'market_potential', 'rental_price', 'distance_to_nearest_walmart',
        'has_target'
    ]
    top_locations = top_locations[columns_order]
    
    # Save results
    output_file = 'data/processed/top_optimal_locations.csv'
    top_locations.to_csv(output_file, index=False)
    print(f"\nTop {len(top_locations)} optimal locations saved to: {output_file}")
    
    # Print summary statistics
    print(f"\nSummary of Top {len(top_locations)} Locations:")
    
    print("\nAverage Metrics:")
    metrics = ['prediction_score', 'population', 'population_density', 'per_capita_income',
              'employment_rate', 'unemployment_rate', 'labor_force_participation',
              'market_potential', 'rental_price', 'distance_to_nearest_walmart']
    print(top_locations[metrics].mean().round(4))
    
    print("\nMetric Ranges:")
    print(top_locations[metrics].agg(['min', 'max']).round(4))
    
    # Print state distribution
    print("\nState Distribution:")
    state_dist = top_locations.groupby('state')['zip'].count().sort_values(ascending=False)
    print(state_dist)
    
    # Print distance distribution
    print("\nDistance to Nearest Walmart Distribution:")
    distance_bins = [0, 10, 20, 30, 40, 50, 100, float('inf')]
    distance_labels = ['0-10km', '10-20km', '20-30km', '30-40km', '40-50km', '50-100km', '100km+']
    top_locations['distance_category'] = pd.cut(
        top_locations['distance_to_nearest_walmart'],
        bins=distance_bins,
        labels=distance_labels
    )
    print(top_locations['distance_category'].value_counts().sort_index())
    
    # Print top 10 locations
    print("\nTop 10 Most Promising Locations:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(top_locations.head(10))
    
    return top_locations

if __name__ == "__main__":
    optimal_locations = predict_optimal_locations()
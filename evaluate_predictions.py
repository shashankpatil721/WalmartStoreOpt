import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial import cKDTree

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points on Earth."""
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

def find_nearest_store(row, actual_stores):
    """Find the nearest actual Walmart store to a predicted location."""
    distances = actual_stores.apply(
        lambda store: haversine_distance(
            row['lat'], row['lng'],
            store['latitude'], store['longitude']
        ),
        axis=1
    )
    return distances.min()

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized version of haversine distance calculation."""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def calculate_distances(predicted_coords, actual_coords):
    """Calculate distances between all predicted and actual coordinates."""
    n_pred = len(predicted_coords)
    n_actual = len(actual_coords)
    distances = np.zeros((n_pred, n_actual))
    
    for i in range(n_pred):
        distances[i, :] = haversine_distance_vectorized(
            predicted_coords[i, 0],
            predicted_coords[i, 1],
            actual_coords[:, 0],
            actual_coords[:, 1]
        )
    
    return distances

def build_kdtree(df, lat_col='latitude', lng_col='longitude'):
    """Build a KD-tree for fast spatial queries."""
    coords = np.radians(df[[lat_col, lng_col]].values)
    return cKDTree(coords)

def calculate_market_potential_batch(predicted_df, all_zips, batch_size=100):
    """Calculate market potential metrics in batches using KD-tree."""
    radius_km = 50
    R = 6371  # Earth's radius in kilometers
    radius_rad = radius_km / R  # Convert to radians
    
    # Build KD-tree for all_zips
    tree = build_kdtree(all_zips)
    coords = np.radians(all_zips[['latitude', 'longitude']].values)
    
    total_populations = []
    avg_incomes = []
    
    # Process in batches
    for i in range(0, len(predicted_df), batch_size):
        batch = predicted_df.iloc[i:i + batch_size]
        batch_coords = np.radians(batch[['lat', 'lng']].values)
        
        # Query KD-tree for nearby points
        indices_list = tree.query_ball_point(batch_coords, radius_rad)
        
        # Calculate metrics for each point in batch
        for indices in indices_list:
            nearby_zips = all_zips.iloc[indices]
            total_populations.append(nearby_zips['population'].sum())
            avg_incomes.append(nearby_zips['median_income'].mean())
    
    return total_populations, avg_incomes

def evaluate_predictions():
    """Two-stage evaluation of predicted store locations."""
    print("\n=== Two-Stage Model Evaluation ===\n")
    print("Loading data and calculating metrics...")
    
    # Load data
    predicted = pd.read_csv('data/processed/top_optimal_locations.csv')
    actual = pd.read_csv('data/raw/Walmart_Store_Locations.csv')
    all_zips = pd.read_csv('data/processed/features_enhanced.csv')
    
    # Stage 1: Evaluate High-Potential Areas
    print("\n=== Stage 1: High-Potential Area Analysis ===")
    
    # Calculate market potential metrics using optimized batch processing
    print("Calculating market potential metrics...")
    total_populations, avg_incomes = calculate_market_potential_batch(predicted, all_zips)
    predicted['total_population'] = total_populations
    predicted['avg_income'] = avg_incomes
    
    # Market potential metrics
    print("\nMarket Potential Metrics:")
    print(f"Average Population in Market Area: {predicted['total_population'].mean():,.0f}")
    print(f"Median Population in Market Area: {predicted['total_population'].median():,.0f}")
    print(f"Average Income in Market Area: ${predicted['avg_income'].mean():,.2f}")
    print(f"Median Income in Market Area: ${predicted['avg_income'].median():,.2f}")
    
    # Stage 2: Evaluate Strategic Spacing
    print("\n=== Stage 2: Strategic Spacing Analysis ===")
    
    # Calculate distance metrics using vectorized operations
    print("Calculating distances between predicted and actual locations...")
    actual_coords = actual[['latitude', 'longitude']].values
    predicted_coords = predicted[['lat', 'lng']].values
    
    # Calculate all distances
    distances = calculate_distances(predicted_coords, actual_coords)
    predicted['nearest_actual_distance'] = distances.min(axis=1)
    
    # Distance distribution analysis
    distance_bins = [0, 20, 40, 60, 80, 100, float('inf')]
    distance_labels = ['0-20km', '20-40km', '40-60km', '60-80km', '80-100km', '100km+']
    predicted['distance_category'] = pd.cut(predicted['nearest_actual_distance'], 
                                          bins=distance_bins, 
                                          labels=distance_labels)
    
    print("\nDistance Distribution:")
    distance_dist = predicted['distance_category'].value_counts().sort_index()
    for category, count in distance_dist.items():
        print(f"{category}: {count:,} locations ({count/len(predicted)*100:.1f}%)")
    
    # Market coverage analysis using KD-tree
    print("\nCalculating market coverage...")
    tree = build_kdtree(predicted, lat_col='lat', lng_col='lng')
    coords = np.radians(all_zips[['latitude', 'longitude']].values)
    radius_rad = 50 / 6371  # 50km in radians
    
    # Count population within radius of any predicted store
    covered_indices = tree.query_ball_point(coords, radius_rad)
    covered_zips = set([i for sublist in covered_indices for i in sublist])
    covered_population = all_zips.iloc[list(covered_zips)]['population'].sum()
    total_us_population = all_zips['population'].sum()
    coverage_ratio = covered_population / total_us_population
    
    print("\nMarket Coverage Analysis:")
    print(f"Total US Population: {total_us_population:,.0f}")
    print(f"Population within 50km of Predicted Stores: {covered_population:,.0f}")
    print(f"Population Coverage Ratio: {coverage_ratio:.2%}")
    
    # Calculate state-level distribution
    state_dist = predicted['state'].value_counts()
    print("\nTop 10 States by Number of Predicted Locations:")
    for state, count in state_dist.head(10).items():
        print(f"{state}: {count:,} locations ({count/len(predicted)*100:.1f}%)")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'metric': [
            'Average Population in Market Area',
            'Median Population in Market Area',
            'Average Income in Market Area',
            'Median Income in Market Area',
            'Population Coverage Ratio',
            *[f'Locations {dist}' for dist in distance_labels],
            *[f'Locations in {state}' for state in state_dist.head(10).index]
        ],
        'value': [
            predicted['total_population'].mean(),
            predicted['total_population'].median(),
            predicted['avg_income'].mean(),
            predicted['avg_income'].median(),
            coverage_ratio,
            *[distance_dist[label] for label in distance_labels],
            *state_dist.head(10).values
        ]
    })
    
    results_df.to_csv('data/processed/detailed_evaluation_metrics.csv', index=False)
    print("\nDetailed results saved to: data/processed/detailed_evaluation_metrics.csv")
    
    # Create visualizations
    try:
        import matplotlib.pyplot as plt
        
        # Distance distribution plot
        plt.figure(figsize=(12, 6))
        plt.hist(predicted['nearest_actual_distance'], bins=50, color='blue', alpha=0.7)
        plt.axvline(predicted['nearest_actual_distance'].mean(), color='red', linestyle='--', 
                   label=f"Mean ({predicted['nearest_actual_distance'].mean():.1f} km)")
        plt.axvline(predicted['nearest_actual_distance'].median(), color='green', linestyle='--',
                   label=f"Median ({predicted['nearest_actual_distance'].median():.1f} km)")
        plt.xlabel('Distance to Nearest Actual Store (km)')
        plt.ylabel('Number of Predicted Locations')
        plt.title('Distribution of Distances to Nearest Walmart Store')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('data/processed/distance_distribution.png')
        
        # Population vs Income scatter plot
        plt.figure(figsize=(12, 6))
        plt.scatter(predicted['total_population'], predicted['avg_income'], alpha=0.6)
        plt.xlabel('Total Population in Market Area')
        plt.ylabel('Average Income in Market Area ($)')
        plt.title('Market Potential Analysis')
        plt.grid(True, alpha=0.3)
        plt.savefig('data/processed/market_potential.png')
        
        # State distribution plot
        plt.figure(figsize=(12, 6))
        state_dist.head(10).plot(kind='bar')
        plt.title('Top 10 States by Number of Predicted Locations')
        plt.xlabel('State')
        plt.ylabel('Number of Locations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/processed/state_distribution.png')
        
        print("Visualization plots saved to data/processed/")
    except Exception as e:
        print(f"Could not create visualization plots: {str(e)}")

if __name__ == "__main__":
    evaluate_predictions() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from data_loader import calculate_haversine_distance_vectorized
from pathlib import Path

def load_data():
    """Load predicted and actual Walmart locations."""
    # Load predicted locations
    predicted_df = pd.read_csv('data/processed/top_1000_optimal_locations.csv')
    
    # Load actual Walmart locations
    walmart_df = pd.read_csv('data/raw/Walmart_Store_Locations.csv')
    walmart_df = walmart_df.dropna(subset=['latitude', 'longitude'])
    
    return predicted_df, walmart_df

def calculate_distance_metrics(predicted_df, walmart_df):
    """Calculate distance-based metrics between predicted and actual locations."""
    # Calculate distances between each predicted location and all actual Walmart stores
    distances = calculate_haversine_distance_vectorized(
        predicted_df['lat'].values,
        predicted_df['lng'].values,
        walmart_df['latitude'].values,
        walmart_df['longitude'].values
    )
    
    # Calculate minimum distance for each predicted location
    min_distances = np.min(distances, axis=1)
    
    metrics = {
        'mean_distance': np.mean(min_distances),
        'median_distance': np.median(min_distances),
        'min_distance': np.min(min_distances),
        'max_distance': np.max(min_distances),
        'std_distance': np.std(min_distances)
    }
    
    return metrics, min_distances

def calculate_precision_at_k(predicted_df, walmart_df, k_values=[10, 50, 100, 500]):
    """Calculate Precision@k metrics."""
    precision_metrics = {}
    
    # Convert coordinates to radians
    predicted_lats = np.radians(predicted_df['lat'].values)
    predicted_lons = np.radians(predicted_df['lng'].values)
    walmart_lats = np.radians(walmart_df['latitude'].values)
    walmart_lons = np.radians(walmart_df['longitude'].values)
    
    # For each k value
    for k in k_values:
        # Take top k predictions
        top_k_lats = predicted_lats[:k]
        top_k_lons = predicted_lons[:k]
        
        # Count how many predictions have a Walmart within 50km
        hits = 0
        for i in range(len(top_k_lats)):
            # Calculate distances to all Walmart stores
            dlat = walmart_lats - top_k_lats[i]
            dlon = walmart_lons - top_k_lons[i]
            
            a = np.sin(dlat/2)**2 + np.cos(top_k_lats[i]) * np.cos(walmart_lats) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distances = 6371 * c  # Earth's radius in km
            
            # Check if any Walmart is within 50km
            if np.any(distances <= 50):
                hits += 1
        
        precision_metrics[f'precision@{k}'] = hits / k
    
    return precision_metrics

def visualize_locations(predicted_df, walmart_df):
    """Create a map visualization of predicted and actual Walmart locations."""
    # Load US states shapefile
    usa = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot US states
    usa.plot(ax=ax, color='lightgray', edgecolor='white')
    
    # Plot actual Walmart locations
    ax.scatter(walmart_df['longitude'], walmart_df['latitude'], 
              c='blue', s=10, alpha=0.3, label='Actual Walmart')
    
    # Plot predicted locations
    ax.scatter(predicted_df['lng'], predicted_df['lat'], 
              c='red', s=30, alpha=0.6, label='Predicted Location')
    
    # Customize the plot
    ax.set_title('Predicted vs Actual Walmart Locations', fontsize=14)
    ax.legend()
    
    # Set map bounds for continental US
    ax.set_xlim([-125, -65])
    ax.set_ylim([25, 50])
    
    # Save the plot
    plt.savefig('data/processed/location_comparison_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    predicted_df, walmart_df = load_data()
    
    # Calculate distance metrics
    print("\nCalculating distance metrics...")
    distance_metrics, min_distances = calculate_distance_metrics(predicted_df, walmart_df)
    
    # Calculate precision metrics
    print("Calculating precision metrics...")
    precision_metrics = calculate_precision_at_k(predicted_df, walmart_df)
    
    # Print results
    print("\nDistance Metrics:")
    for metric, value in distance_metrics.items():
        print(f"{metric}: {value:.2f} km")
    
    print("\nPrecision Metrics:")
    for metric, value in precision_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create visualization
    print("\nCreating visualization...")
    visualize_locations(predicted_df, walmart_df)
    print("Visualization saved as 'data/processed/location_comparison_map.png'")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'metric': list(distance_metrics.keys()) + list(precision_metrics.keys()),
        'value': list(distance_metrics.values()) + list(precision_metrics.values())
    })
    results_df.to_csv('data/processed/prediction_metrics.csv', index=False)
    print("\nDetailed metrics saved to 'data/processed/prediction_metrics.csv'")

if __name__ == "__main__":
    main() 
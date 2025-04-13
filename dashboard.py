import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from scipy.spatial.distance import cdist

# Set page config
st.set_page_config(
    page_title="Walmart Location Predictor",
    page_icon="🏪",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    # Load predictions and features
    predictions = pd.read_csv('data/model/test_predictions_nationwide.csv')
    features = pd.read_csv('data/processed/features_nationwide.csv')
    
    # Load Walmart locations
    walmart_locations = pd.read_csv('data/raw/Walmart_Store_Locations.csv')
    
    # Clean ZIP codes to ensure matching
    walmart_locations['zip_clean'] = walmart_locations['zip_code'].astype(str).str.extract('(\d{5})')[0]
    predictions['zip_clean'] = predictions['zip'].astype(str).str.extract('(\d{5})')[0]
    
    # First merge predictions with features
    df = predictions.merge(features, on='zip', how='left')
    
    # Create separate DataFrames for actual and predicted locations
    actual_df = walmart_locations.copy()
    actual_df['location_type'] = 'Actual Walmart Location'
    actual_df['lat'] = actual_df['latitude']
    actual_df['lon'] = actual_df['longitude']
    
    # For predicted locations, use only available columns
    available_columns = [
        'zip', 'latitude_x', 'longitude_x', 'state_x', 
        'population_density_x', 'market_potential_x', 
        'distance_to_nearest_walmart_x', 'actual', 
        'predicted', 'probability'
    ]
    
    predicted_df = df[available_columns].copy()
    
    # Rename columns to remove _x suffix
    predicted_df = predicted_df.rename(columns={
        'latitude_x': 'latitude',
        'longitude_x': 'longitude',
        'state_x': 'state',
        'population_density_x': 'population_density',
        'market_potential_x': 'market_potential',
        'distance_to_nearest_walmart_x': 'distance_to_nearest_walmart'
    })
    
    # Format numeric columns for tooltip
    predicted_df['population_density_fmt'] = predicted_df['population_density'].round(0).astype(int).apply(lambda x: f"{x:,}")
    predicted_df['market_potential_fmt'] = predicted_df['market_potential'].round(2).apply(lambda x: f"{x:.2f}")
    predicted_df['distance_fmt'] = predicted_df['distance_to_nearest_walmart'].round(1).apply(lambda x: f"{x:.1f}")
    predicted_df['probability_fmt'] = predicted_df['probability'].apply(lambda x: f"{x:.1%}")
    
    # Add lat/lon columns for mapping
    predicted_df['lat'] = predicted_df['latitude']
    predicted_df['lon'] = predicted_df['longitude']
    
    # Format tooltip information for actual locations
    actual_df['tooltip_text'] = actual_df.apply(
        lambda row: f"<b>Actual Walmart Location</b><br/>"
                   f"<b>ZIP:</b> {row['zip_code']}<br/>"
                   f"<b>State:</b> {row['state']}<br/>"
                   f"<b>City:</b> {row['city']}", 
        axis=1
    )
    
    # Format tooltip information for predicted locations
    predicted_df['tooltip_text'] = predicted_df.apply(
        lambda row: f"<b>Predicted Walmart Location</b><br/>"
                   f"<b>ZIP:</b> {row['zip']}<br/>"
                   f"<b>State:</b> {row['state']}<br/>"
                   f"<b>Population Density:</b> {row['population_density_fmt']}/sq mi<br/>"
                   f"<b>Market Potential:</b> ${row['market_potential_fmt']}<br/>"
                   f"<b>Distance to Nearest Walmart:</b> {row['distance_fmt']} miles<br/>"
                   f"<b>Prediction Confidence:</b> {row['probability_fmt']}", 
        axis=1
    )
    
    return actual_df, predicted_df

def calculate_distance_metrics(actual_locs, predicted_locs):
    """Calculate distance-based accuracy metrics between actual and predicted locations."""
    if len(actual_locs) == 0 or len(predicted_locs) == 0:
        return {
            'mean_distance': float('inf'),
            'median_distance': float('inf'),
            'min_distance': float('inf'),
            'max_distance': float('inf'),
            'within_10_miles': 0,
            'within_20_miles': 0,
            'within_50_miles': 0
        }
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance between two points on Earth."""
        R = 3959  # Earth's radius in miles
        
        # Convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    # Get coordinates
    actual_lats = actual_locs['lat'].values
    actual_lons = actual_locs['lon'].values
    pred_lats = predicted_locs['lat'].values
    pred_lons = predicted_locs['lon'].values
    
    # Calculate distances between all pairs of points
    min_distances = []
    for pred_lat, pred_lon in zip(pred_lats, pred_lons):
        distances = haversine_distance(
            pred_lat, pred_lon,
            actual_lats[:, np.newaxis], actual_lons[:, np.newaxis]
        )
        min_distances.append(np.min(distances))
    
    min_distances = np.array(min_distances)
    
    return {
        'mean_distance': np.mean(min_distances),
        'median_distance': np.median(min_distances),
        'min_distance': np.min(min_distances),
        'max_distance': np.max(min_distances),
        'within_10_miles': np.mean(min_distances <= 10) * 100,
        'within_20_miles': np.mean(min_distances <= 20) * 100,
        'within_50_miles': np.mean(min_distances <= 50) * 100
    }

# Load the data
actual_df, predicted_df = load_data()

# Title and description
st.title('🏪 Walmart Location Predictor')
st.markdown("""
This dashboard shows predicted and actual Walmart store locations across the United States.
The model uses demographic and economic data to predict which ZIP codes are likely to have a Walmart store.
""")

# Sidebar controls
st.sidebar.header('📊 Controls')

# Add probability threshold slider
probability_threshold = st.sidebar.slider(
    'Prediction Probability Threshold',
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help='Adjust this to filter predictions based on confidence level. Higher values mean fewer but more confident predictions.'
)

# Update predictions based on threshold
predicted_df['predicted_filtered'] = (predicted_df['probability'] >= probability_threshold) & (predicted_df['predicted'] == 1)

# Filter data if states are selected
states = sorted(actual_df['state'].unique())
selected_states = st.sidebar.multiselect(
    'Filter by States',
    states,
    default=[]
)

if selected_states:
    actual_locations = actual_df[actual_df['state'].isin(selected_states)].copy()
    predicted_locations = predicted_df[predicted_df['state'].isin(selected_states) & 
                                    (predicted_df['predicted_filtered'] == 1)].copy()
else:
    actual_locations = actual_df.copy()
    predicted_locations = predicted_df[predicted_df['predicted_filtered'] == 1].copy()

predicted_locations['location_type'] = 'Predicted Walmart Location'

# Calculate metrics
total_zips = len(predicted_df)
actual_walmarts = len(actual_locations)
predicted_walmarts = len(predicted_locations)

# Calculate accuracy only if we have both actual and predicted columns
if 'actual' in predicted_df.columns and 'predicted_filtered' in predicted_df.columns:
    accuracy = np.mean(predicted_df['actual'] == predicted_df['predicted_filtered'])
else:
    accuracy = 0.0  # Default accuracy if we can't calculate it

# Calculate distance-based metrics
distance_metrics = calculate_distance_metrics(actual_locations, predicted_locations)

# Display metrics
st.sidebar.header('📈 Basic Metrics')
st.sidebar.metric("Total ZIP Codes", f"{total_zips:,}")
st.sidebar.metric("Actual Walmart Locations", f"{actual_walmarts:,}")
st.sidebar.metric("Predicted Walmart Locations", f"{predicted_walmarts:,}")
st.sidebar.metric("Model Accuracy", f"{accuracy:.1%}")

st.sidebar.header('📏 Distance Metrics')
st.sidebar.metric("Mean Distance to Nearest Walmart", f"{distance_metrics['mean_distance']:.1f} miles")
st.sidebar.metric("Median Distance to Nearest Walmart", f"{distance_metrics['median_distance']:.1f} miles")
st.sidebar.metric("Predictions Within 10 Miles", f"{distance_metrics['within_10_miles']:.1f}%")
st.sidebar.metric("Predictions Within 20 Miles", f"{distance_metrics['within_20_miles']:.1f}%")
st.sidebar.metric("Predictions Within 50 Miles", f"{distance_metrics['within_50_miles']:.1f}%")

# Create layers for the map
actual_layer = pdk.Layer(
    'ScatterplotLayer',
    data=actual_locations,
    get_position=['lon', 'lat'],
    get_color=[0, 255, 0, 140],  # Green for actual locations
    get_radius=3000,  # Smaller radius
    pickable=True,
    opacity=0.6,
    stroked=True,
    filled=True,
    tooltip=True
)

predicted_layer = pdk.Layer(
    'ScatterplotLayer',
    data=predicted_locations,
    get_position=['lon', 'lat'],
    get_color=[0, 0, 255, 140],  # Blue for predicted locations
    get_radius=3000,  # Smaller radius
    pickable=True,
    opacity=0.6,
    stroked=True,
    filled=True,
    tooltip=True
)

# Create the map
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=39.8283,
        longitude=-98.5795,
        zoom=3,
        pitch=0,
    ),
    layers=[actual_layer, predicted_layer],
    tooltip={
        'html': """
            <div style="background-color: {location_type === 'Actual Walmart Location' ? '#006400' : '#4682B4'}; color: white; padding: 10px; border-radius: 5px;">
                {tooltip_text}
            </div>
        """
    }
))

# Update legend and add explanation
st.markdown(f"""
**Legend:**
- 🟢 Actual Walmart Locations (Green): {actual_walmarts:,} stores
- 🔵 Predicted Walmart Locations (Blue): {predicted_walmarts:,} locations (at {probability_threshold:.0%} confidence threshold)

**Distance-Based Accuracy:**
- Average distance to nearest actual Walmart: {distance_metrics['mean_distance']:.1f} miles
- {distance_metrics['within_10_miles']:.1f}% of predictions are within 10 miles of an actual Walmart
- {distance_metrics['within_20_miles']:.1f}% of predictions are within 20 miles of an actual Walmart
- {distance_metrics['within_50_miles']:.1f}% of predictions are within 50 miles of an actual Walmart

**Controls:**
- Use the probability threshold slider to adjust prediction confidence
- Current threshold of {probability_threshold:.0%} results in {predicted_walmarts:,} predicted locations
- Higher threshold = fewer but more confident predictions
- Lower threshold = more but less confident predictions
- Use state filter to focus on specific regions
""")

# Show state-level statistics
if st.checkbox('Show State-Level Statistics'):
    st.subheader('State-Level Statistics')
    
    state_stats = predicted_df.groupby('state').agg({
        'actual': 'sum',
        'predicted_filtered': 'sum',
        'zip': 'count'
    }).reset_index()
    
    state_stats.columns = ['State', 'Actual Walmarts', 'Predicted Walmarts', 'Total ZIP Codes']
    state_stats['Accuracy'] = 1 - abs(state_stats['Actual Walmarts'] - state_stats['Predicted Walmarts']) / state_stats['Total ZIP Codes']
    
    state_stats = state_stats.sort_values('Actual Walmarts', ascending=False)
    
    st.dataframe(state_stats.style.format({
        'Actual Walmarts': '{:,.0f}',
        'Predicted Walmarts': '{:,.0f}',
        'Total ZIP Codes': '{:,.0f}',
        'Accuracy': '{:.1%}'
    }))

# Footer
st.markdown('---')
st.markdown('Data sources: US Census Bureau, Walmart Store Locations') 
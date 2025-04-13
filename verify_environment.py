import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import geopy
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import folium
import geopandas as gpd
import shapely
from shapely.geometry import Point
import joblib
from tqdm import __version__ as tqdm_version
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def verify_installations():
    """Verify all required packages are installed and print their versions"""
    print("\nVerifying Python Environment Setup:")
    print(f"Python version: {sys.version}")
    
    # List of packages to verify
    print("\nPackage Versions:")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"Seaborn: {sns.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"XGBoost: {xgb.__version__}")
    print(f"GeoPy: {geopy.__version__}")
    print(f"Folium: {folium.__version__}")
    print(f"GeoPandas: {gpd.__version__}")
    print(f"Shapely: {shapely.__version__}")
    print(f"Joblib: {joblib.__version__}")
    print(f"tqdm: {tqdm_version}")

def test_basic_functionality():
    """Test basic functionality of key packages"""
    print("\nTesting Basic Functionality:")
    
    # Create sample data
    print("\n1. Testing NumPy and Pandas:")
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'latitude': np.random.uniform(25, 50, n_samples),
        'longitude': np.random.uniform(-130, -60, n_samples),
        'population': np.random.lognormal(10, 1, n_samples),
        'income': np.random.normal(50000, 10000, n_samples),
        'target': np.random.binomial(1, 0.3, n_samples)
    })
    print("✓ Created sample dataset with NumPy and Pandas")
    
    # Test scikit-learn
    print("\n2. Testing Scikit-learn:")
    X = data[['latitude', 'longitude', 'population', 'income']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print("✓ Performed train-test split and scaling")
    
    # Test XGBoost
    print("\n3. Testing XGBoost:")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    print("✓ Successfully trained XGBoost model")
    
    # Test GeoPy
    print("\n4. Testing GeoPy:")
    point1 = (40.7128, -74.0060)  # New York
    point2 = (34.0522, -118.2437)  # Los Angeles
    distance = geodesic(point1, point2).miles
    print(f"✓ Calculated distance between NY and LA: {distance:.2f} miles")
    
    # Test Folium
    print("\n5. Testing Folium:")
    m = folium.Map(location=[40, -98], zoom_start=4)
    print("✓ Created basic map with Folium")
    
    # Test GeoPandas
    print("\n6. Testing GeoPandas:")
    gdf = gpd.GeoDataFrame(
        data,
        geometry=[Point(xy) for xy in zip(data.longitude, data.latitude)]
    )
    print("✓ Created GeoDataFrame from sample data")
    
    # Test Joblib
    print("\n7. Testing Joblib:")
    temp_file = 'temp_model.joblib'
    joblib.dump(xgb_model, temp_file)
    loaded_model = joblib.load(temp_file)
    print("✓ Successfully saved and loaded model with Joblib")
    
    # Cleanup
    import os
    os.remove(temp_file)
    
    print("\nAll functionality tests completed successfully!")

if __name__ == "__main__":
    verify_installations()
    test_basic_functionality() 
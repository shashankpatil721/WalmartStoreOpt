# Walmart Location Predictor

This project predicts optimal locations for new Walmart stores across the United States using machine learning. It analyzes demographic, economic, and geographic data to identify ZIP codes that have similar characteristics to existing Walmart locations.

## Features

- Interactive map visualization of actual and predicted Walmart locations
- Real-time filtering by state and prediction confidence
- Detailed metrics for each predicted location:
  - Population density
  - Market potential
  - Distance to nearest existing Walmart
  - Prediction confidence score
- State-level statistics and accuracy metrics
- Distance-based accuracy measurements

## Getting Started

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```
4. Open http://localhost:8501 in your browser

## Using the Dashboard

- **Map View**: 
  - Green dots: Actual Walmart locations
  - Blue dots: Predicted locations
  - Hover over any point to see detailed information

- **Controls**:
  - Use the probability threshold slider to adjust prediction confidence
  - Filter by states using the dropdown menu
  - Toggle state-level statistics view

- **Metrics**:
  - Basic metrics show total coverage and model accuracy
  - Distance-based metrics show how close predictions are to actual stores
  - State-level statistics provide detailed breakdown by region

## Model Performance

The model's performance can be evaluated through:
- Overall accuracy (matching between predicted and actual locations)
- Distance-based metrics:
  - Percentage of predictions within 10 miles of actual stores
  - Percentage of predictions within 20 miles of actual stores
  - Percentage of predictions within 50 miles of actual stores

## Data Sources

- Walmart store locations
- US Census Bureau demographic data
- Economic indicators
- Geographic data

## Project Structure

```
├── data/
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed features
│   └── model/             # Model predictions
├── scripts/               # Data processing scripts
├── dashboard.py          # Streamlit dashboard
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Contributing

Feel free to submit issues and enhancement requests!

## Project Summary
This project uses machine learning to predict optimal locations for new Walmart stores across the United States. The analysis combines demographic, economic, and geographic data to identify areas with high potential for successful store operations.

## Key Findings

### Model Performance
- **Best Model**: XGBoost classifier achieved exceptional performance with:
  - Accuracy: 98.17% (Error Rate: 1.83%)
  - Precision: 98.05%
  - Recall: 98.26%
  - F1 Score: 98.15%
  - ROC-AUC: 0.9989
  - Mean Distance Error: 517.69 miles
  - IoU (Intersection-over-Union): 0.5304

### Feature Importance
1. Employment Rate (67.13%): Most significant predictor
2. Market Potential (17.21%): Second most important feature
3. Distance to Nearest Walmart (5.28%): Store network effects
4. Labor Force Participation (4.60%): Employment engagement
5. Population (2.44%): Demographic indicator
6. Unemployment Rate (1.42%): Economic health
7. Population Density (1.22%): Urban concentration
8. Per Capita Income (0.70%): Wealth indicator

### Regional Analysis
- **West Region**: Lower prediction errors, strong correlation with population centers
- **Central Region**: Moderate prediction accuracy, influenced by existing store network
- **East Region**: Higher prediction complexity due to dense urban areas
- **Other Regions**: Limited data points, higher uncertainty

### Key Insights
1. Population density is the dominant predictor of Walmart store locations
2. Significant store clustering effects observed
3. Employment metrics contribute ~22% to prediction power
4. Market size more influential than individual income metrics
5. Regional variations in prediction accuracy reflect different market dynamics

## Interactive Dashboard

### Features
1. **Overview**: Key metrics and top predicted locations
2. **Location Map**: Interactive visualization of existing and predicted store locations
3. **Model Performance**: Detailed comparison of model metrics
4. **Regional Analysis**: Breakdown of predictions and errors by region

### Installation & Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

### Data Sources
- `data/processed/top_1000_optimal_locations.csv`: Predicted optimal locations
- `data/raw/Walmart_Store_Locations.csv`: Existing store locations
- `data/processed/model_comparison_results.csv`: Model performance metrics
- `data/processed/region_errors.csv`: Regional error analysis

## Recommendations for New Locations

### Primary Factors to Consider
1. **Population Density**: Target areas with high population density
2. **Market Gaps**: Consider distances from existing stores
3. **Employment Metrics**: Prioritize areas with strong employment indicators
4. **Market Potential**: Focus on regions with high growth potential

### Implementation Strategy
1. **Phase 1**: Target top 100 predicted locations with highest confidence scores
2. **Phase 2**: Evaluate regional distribution for network optimization
3. **Phase 3**: Consider local factors and competition analysis

## Technical Details

### Model Architecture
- **Algorithm**: XGBoost Classifier
- **Hyperparameters**:
  - n_estimators: 600
  - max_depth: 6
  - learning_rate: 0.05
  - min_child_weight: 1
  - gamma: 0.05
  - colsample_bytree: 1.0

### Validation Strategy
- 80-20 train-test split
- Balanced dataset creation
- Cross-validation for hyperparameter tuning
- Comprehensive metrics including:
  - Classification metrics (Accuracy, Precision, Recall, F1)
  - Spatial metrics (Mean Distance Error, IoU)
  - Feature importance analysis

### Model Evaluation Results
- **Classification Performance**:
  - True Negatives: 459
  - False Positives: 9
  - False Negatives: 8
  - True Positives: 452
- **Spatial Performance**:
  - Mean Distance Error: 517.69 miles
  - Standard Deviation of Distance Error: 790.09 miles
  - IoU Score: 0.5304

## Future Improvements
1. Incorporate additional features:
   - Competition analysis
   - Real estate costs
   - Traffic patterns
2. Develop time-series predictions for market evolution
3. Include local regulatory and zoning considerations
4. Enhance regional-specific modeling

## Contact
For questions or suggestions, please open an issue in the repository. 
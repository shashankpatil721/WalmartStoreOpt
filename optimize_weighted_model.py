import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Load and prepare the dataset with enhanced features"""
    df = pd.read_csv('data/processed/enhanced_features.csv')
    
    # Create target variable based on presence of Walmart stores
    y = (df['walmart_stores_within_50km'] > 0).astype(int)
    
    # Select features for training
    feature_columns = [
        # Base features
        'population', 'population_density', 'per_capita_income',
        'employment_rate', 'market_potential', 'employment_health_score',
        'distance_to_nearest_walmart',
        
        # Store density features
        'walmart_density_5km', 'walmart_density_10km', 
        'walmart_density_25km', 'walmart_density_50km',
        
        # Interaction terms
        'population_employment', 'density_income', 'market_employment',
        'income_density', 'market_density', 'employment_density',
        
        # Real estate features
        'rental_price', 'rental_price_volatility', 'rental_price_trend',
        
        # Competitor density features
        'target_density_5km', 'target_density_10km', 'target_density_25km',
        'costco_density_5km', 'costco_density_10km', 'costco_density_25km',
        'grocery_density_5km', 'grocery_density_10km', 'grocery_density_25km'
    ]
    
    # Select only available features
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X, y, df[['zip', 'lat', 'lng']]

def optimize_weighted_model(X_train, y_train):
    """Optimize XGBoost model with enhanced parameters for the larger feature set"""
    # Calculate scale_pos_weight
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    # Define parameter grid with expanded search space
    param_grid = {
        'max_depth': [4, 5, 6],
        'min_child_weight': [1, 3, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 300, 400],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]  # Add regularization parameter
    }
    
    # Create base model with additional parameters
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',  # Faster training for larger feature sets
        enable_categorical=True  # Enable categorical feature support
    )
    
    # Set up cross-validation with more folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    logging.info(f"\nBest parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, location_info):
    """Evaluate model performance with enhanced analysis"""
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    class_report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    # Get feature importance with absolute values
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': np.abs(model.feature_importances_)
    }).sort_values('importance', ascending=False)
    
    # Group features by category
    feature_categories = {
        'Demographics': ['population', 'population_density'],
        'Economic': ['per_capita_income', 'employment_rate', 'market_potential'],
        'Store Density': [col for col in X_test.columns if 'density' in col],
        'Interaction': [col for col in X_test.columns if '_' in col and 'density' not in col],
        'Real Estate': [col for col in X_test.columns if 'rental' in col or 'price' in col],
        'Competition': [col for col in X_test.columns if any(comp in col for comp in ['target', 'costco', 'grocery'])]
    }
    
    # Calculate category importance
    category_importance = {}
    for category, features in feature_categories.items():
        available_features = [f for f in features if f in feature_importance['feature'].values]
        if available_features:
            category_importance[category] = feature_importance[
                feature_importance['feature'].isin(available_features)
            ]['importance'].mean()
    
    # Analyze prediction errors
    error_analysis = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': y_prob,
        'zip': location_info.iloc[y_test.index]['zip'],
        'lat': location_info.iloc[y_test.index]['lat'],
        'lng': location_info.iloc[y_test.index]['lng']
    })
    
    # Calculate error types
    error_analysis['error_type'] = 'correct'
    error_analysis.loc[(y_test == 1) & (y_pred == 0), 'error_type'] = 'false_negative'
    error_analysis.loc[(y_test == 0) & (y_pred == 1), 'error_type'] = 'false_positive'
    
    # Compile results
    results = {
        'metrics': {
            'accuracy': class_report['accuracy'],
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score'],
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        },
        'feature_importance': feature_importance.to_dict('records'),
        'category_importance': category_importance,
        'error_analysis': {
            'false_positives': len(error_analysis[error_analysis['error_type'] == 'false_positive']),
            'false_negatives': len(error_analysis[error_analysis['error_type'] == 'false_negative']),
            'correct_predictions': len(error_analysis[error_analysis['error_type'] == 'correct'])
        }
    }
    
    # Log results
    logging.info("\nModel Evaluation Results:")
    logging.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    logging.info(f"Precision: {results['metrics']['precision']:.4f}")
    logging.info(f"Recall: {results['metrics']['recall']:.4f}")
    logging.info(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    logging.info(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
    logging.info(f"PR AUC: {results['metrics']['pr_auc']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('data/processed/feature_importance.png')
    plt.close()
    
    # Plot category importance
    plt.figure(figsize=(10, 6))
    category_imp_df = pd.DataFrame({
        'category': list(category_importance.keys()),
        'importance': list(category_importance.values())
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='category', data=category_imp_df)
    plt.title('Feature Category Importance')
    plt.tight_layout()
    plt.savefig('data/processed/category_importance.png')
    plt.close()
    
    # Plot error distribution map
    plt.figure(figsize=(15, 10))
    for error_type, color in [('correct', 'blue'), ('false_positive', 'red'), ('false_negative', 'yellow')]:
        mask = error_analysis['error_type'] == error_type
        plt.scatter(
            error_analysis[mask]['lng'],
            error_analysis[mask]['lat'],
            c=color,
            label=error_type,
            alpha=0.5
        )
    plt.title('Prediction Error Distribution')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/processed/error_distribution.png')
    plt.close()
    
    return results, error_analysis

def main():
    """Run optimized weighted model training and evaluation"""
    logging.info("Starting optimized weighted model training...")
    
    # Create output directory if it doesn't exist
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y, location_info = load_data()
    logging.info(f"Dataset shape: {X.shape}")
    logging.info(f"Class distribution: \n{pd.Series(y).value_counts(normalize=True)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train optimized model
    model = optimize_weighted_model(X_train, y_train)
    
    # Evaluate model
    results, error_analysis = evaluate_model(model, X_test, y_test, location_info)
    
    # Save results
    with open('data/processed/optimized_model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    error_analysis.to_csv('data/processed/prediction_errors.csv', index=False)
    
    logging.info("\nOptimization complete! Results have been saved to:")
    logging.info("- data/processed/optimized_model_results.json")
    logging.info("- data/processed/prediction_errors.csv")
    logging.info("- data/processed/feature_importance.png")
    logging.info("- data/processed/category_importance.png")
    logging.info("- data/processed/error_distribution.png")

if __name__ == "__main__":
    main() 
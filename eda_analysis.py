import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
plt.style.use('seaborn-v0_8')  # Using a valid style name

# Load the merged dataset
df = pd.read_csv('data/processed/location_features.csv')

def plot_store_distribution():
    """Plot distribution of Walmart and Target stores"""
    plt.figure(figsize=(15, 5))
    
    # Plot store presence
    plt.subplot(121)
    store_presence = pd.DataFrame({
        'Target': df['has_target'].sum(),
        'Near Walmart': (df['walmart_stores_within_50km'] > 0).sum()
    }, index=['Store Presence'])
    store_presence.plot(kind='bar')
    plt.title('Number of ZIP Codes with Stores')
    plt.ylabel('Count')
    
    # Plot store density
    plt.subplot(122)
    plt.hist(df['walmart_stores_within_50km'], bins=50, alpha=0.5, label='Walmart')
    plt.title('Store Density Distribution (50km radius)')
    plt.xlabel('Number of Stores')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data/processed/store_distribution.png')
    plt.close()

def plot_population_analysis():
    """Analyze population distribution and its relationship with stores"""
    plt.figure(figsize=(15, 5))
    
    # Population distribution
    plt.subplot(121)
    sns.histplot(data=df, x='population', bins=50)
    plt.title('Population Distribution')
    plt.xlabel('Population')
    
    # Population vs Store Presence
    plt.subplot(122)
    sns.boxplot(data=df, x=(df['walmart_stores_within_50km'] > 0).astype(int), y='population')
    plt.title('Population by Walmart Presence')
    plt.xlabel('Has Walmart Within 50km')
    plt.ylabel('Population')
    
    plt.tight_layout()
    plt.savefig('data/processed/population_analysis.png')
    plt.close()

def plot_economic_indicators():
    """Analyze economic indicators"""
    plt.figure(figsize=(15, 10))
    
    # Per capita income distribution
    plt.subplot(221)
    sns.histplot(data=df, x='per_capita_income_raw', bins=50)
    plt.title('Per Capita Income Distribution')
    plt.xlabel('Per Capita Income')
    
    # Employment rate distribution
    plt.subplot(222)
    sns.histplot(data=df, x='employment_rate_raw', bins=50)
    plt.title('Employment Rate Distribution')
    plt.xlabel('Employment Rate')
    
    # Market potential vs Store Presence
    plt.subplot(223)
    sns.boxplot(data=df, x=(df['walmart_stores_within_50km'] > 0).astype(int), y='market_potential_raw')
    plt.title('Market Potential by Walmart Presence')
    plt.xlabel('Has Walmart Within 50km')
    plt.ylabel('Market Potential')
    
    # Employment health score vs Store Presence
    plt.subplot(224)
    sns.boxplot(data=df, x=(df['walmart_stores_within_50km'] > 0).astype(int), y='employment_health_score_raw')
    plt.title('Employment Health Score by Walmart Presence')
    plt.xlabel('Has Walmart Within 50km')
    plt.ylabel('Employment Health Score')
    
    plt.tight_layout()
    plt.savefig('data/processed/economic_indicators.png')
    plt.close()

def calculate_correlations():
    """Calculate and visualize correlations between key variables"""
    # Select relevant numerical columns
    numerical_cols = [
        'population', 'population_density', 'per_capita_income',
        'employment_rate', 'market_potential', 'employment_health_score',
        'distance_to_nearest_walmart', 'walmart_stores_within_50km'
    ]
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Key Variables')
    plt.tight_layout()
    plt.savefig('data/processed/correlation_matrix.png')
    plt.close()
    
    return corr_matrix

def identify_outliers():
    """Identify outliers in key variables"""
    key_variables = [
        'population', 'population_density', 'per_capita_income',
        'employment_rate', 'market_potential', 'employment_health_score'
    ]
    
    outlier_summary = pd.DataFrame()
    for var in key_variables:
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)][var]
        
        outlier_summary[var] = pd.Series({
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100,
            'min': df[var].min(),
            'max': df[var].max(),
            'median': df[var].median()
        })
    
    return outlier_summary

def generate_summary_statistics():
    """Generate summary statistics for key variables"""
    key_variables = [
        'population', 'population_density', 'per_capita_income',
        'employment_rate', 'market_potential', 'employment_health_score',
        'distance_to_nearest_walmart', 'walmart_stores_within_50km'
    ]
    
    summary_stats = df[key_variables].describe()
    return summary_stats

def analyze_class_distribution():
    """Analyze the distribution of Walmart and Target stores"""
    class_dist = pd.DataFrame({
        'Near_Walmart': (df['walmart_stores_within_50km'] > 0).value_counts(),
        'Has_Target': df['has_target'].value_counts()
    })
    class_dist['Near_Walmart_Percentage'] = (class_dist['Near_Walmart'] / len(df)) * 100
    class_dist['Has_Target_Percentage'] = (class_dist['Has_Target'] / len(df)) * 100
    return class_dist

def main():
    """Run all EDA analyses"""
    print("\nStarting EDA Analysis...")
    
    # Create output directory if it doesn't exist
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Analyze class distribution
    class_dist = analyze_class_distribution()
    print("\nClass Distribution:")
    print(class_dist)
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics()
    print("\nSummary Statistics:")
    print(summary_stats)
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_store_distribution()
    plot_population_analysis()
    plot_economic_indicators()
    
    # Calculate correlations
    corr_matrix = calculate_correlations()
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Identify outliers
    outlier_summary = identify_outliers()
    print("\nOutlier Summary:")
    print(outlier_summary)
    
    # Save numerical results
    print("\nSaving results...")
    corr_matrix.to_csv('data/processed/correlation_matrix.csv')
    outlier_summary.to_csv('data/processed/outlier_summary.csv')
    summary_stats.to_csv('data/processed/summary_statistics.csv')
    class_dist.to_csv('data/processed/class_distribution.csv')
    
    print("\nEDA Analysis Complete!")
    print("\nVisualization files have been saved in the data/processed directory:")
    print("- store_distribution.png")
    print("- population_analysis.png")
    print("- economic_indicators.png")
    print("- correlation_matrix.png")
    print("\nDetailed statistics have been saved in:")
    print("- correlation_matrix.csv")
    print("- outlier_summary.csv")
    print("- summary_statistics.csv")
    print("- class_distribution.csv")

if __name__ == "__main__":
    main() 
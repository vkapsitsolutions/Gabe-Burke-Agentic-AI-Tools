"""Visualization functions for charts and plots."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

def create_distribution_plot(data, column_name, title_prefix=""):
    """Create enhanced distribution plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(data, bins=30, edgecolor='black', alpha=0.7, color='#1f77b4')
    ax1.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {data.mean():.2f}')
    ax1.axvline(data.median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {data.median():.2f}')
    ax1.set_title(f'{title_prefix}Distribution of {column_name}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel(column_name)
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # KDE Plot
    data.plot(kind='kde', ax=ax2, color='#1f77b4', linewidth=2)
    ax2.set_title(f'{title_prefix}Density Plot of {column_name}', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel(column_name)
    ax2.set_ylabel('Density')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_boxplot(data, column_name, title_prefix=""):
    """Create enhanced boxplot with annotations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(data, patch_artist=True, vert=True,
                     boxprops=dict(facecolor='lightblue', color='blue'),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='blue', linewidth=1.5),
                     capprops=dict(color='blue', linewidth=1.5))
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    median = data.median()
    
    ax.text(1.15, median, f'Median: {median:.2f}', 
            verticalalignment='center', fontsize=10, color='red')
    ax.text(1.15, Q1, f'Q1: {Q1:.2f}', verticalalignment='center', fontsize=9)
    ax.text(1.15, Q3, f'Q3: {Q3:.2f}', verticalalignment='center', fontsize=9)
    
    ax.set_title(f'{title_prefix}Box Plot of {column_name}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel(column_name)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_trend_analysis(df, date_column, metrics, title_prefix=""):
    """
    Create time series trend analysis with multiple metrics.
    
    Args:
        df: DataFrame with date column
        date_column: Name of date column
        metrics: List of metric column names (e.g., ['Attendees', 'Capacity'])
        title_prefix: Prefix for chart title
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Ensure date column is datetime
    if df[date_column].dtype != 'datetime64[ns]':
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df_sorted = df.sort_values(date_column)
    
    # Plot each metric
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            color = colors[idx % len(colors)]
            ax.plot(df_sorted[date_column], df_sorted[metric], 
                   marker='o', linewidth=2.5, markersize=6, 
                   label=metric, color=color, alpha=0.8)
            
            # Add trend line
            x_numeric = np.arange(len(df_sorted))
            z = np.polyfit(x_numeric, df_sorted[metric].fillna(0), 1)
            p = np.poly1d(z)
            ax.plot(df_sorted[date_column], p(x_numeric), 
                   linestyle='--', linewidth=1.5, alpha=0.5, color=color)
    
    ax.set_title(f'{title_prefix}Trends Over Time', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add subtle background
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig

def create_utilization_trend(df, date_column, attendees_col, capacity_col, title_prefix=""):
    """Create utilization rate trend over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Ensure date column is datetime
    if df[date_column].dtype != 'datetime64[ns]':
        df[date_column] = pd.to_datetime(df[date_column])
    
    df_sorted = df.sort_values(date_column)
    
    # Calculate utilization rate
    df_sorted['Utilization_Rate'] = (df_sorted[attendees_col] / df_sorted[capacity_col] * 100).fillna(0)
    
    # Plot 1: Attendance vs Capacity
    ax1.plot(df_sorted[date_column], df_sorted[attendees_col], 
            marker='o', linewidth=2.5, markersize=6, 
            label='Attendees', color='#2E86AB', alpha=0.8)
    ax1.plot(df_sorted[date_column], df_sorted[capacity_col], 
            marker='s', linewidth=2.5, markersize=6, 
            label='Capacity', color='#A23B72', alpha=0.8)
    
    ax1.fill_between(df_sorted[date_column], 
                     df_sorted[attendees_col], 
                     df_sorted[capacity_col], 
                     alpha=0.2, color='gray', label='Gap')
    
    ax1.set_title(f'{title_prefix}Attendance vs Capacity Over Time', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    # Plot 2: Utilization Rate
    colors = ['red' if x < 50 else 'orange' if x < 80 else 'green' 
              for x in df_sorted['Utilization_Rate']]
    
    ax2.bar(df_sorted[date_column], df_sorted['Utilization_Rate'], 
           color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=80, color='green', linestyle='--', linewidth=2, 
               label='Optimal (80%)', alpha=0.7)
    ax2.axhline(y=50, color='orange', linestyle='--', linewidth=2, 
               label='Low (50%)', alpha=0.7)
    
    ax2.set_title(f'{title_prefix}Utilization Rate Over Time', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Utilization Rate (%)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_facecolor('#f8f9fa')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def create_bar_chart(grouped_data, group_col, value_col, title_prefix=""):
    """Create enhanced bar chart with value labels."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    grouped_data.plot(kind='bar', ax=ax, color='steelblue', 
                      edgecolor='black', width=0.7)
    ax.set_title(f'{title_prefix}{value_col} by {group_col}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(group_col, fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(grouped_data.values):
        ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def create_correlation_heatmap(corr_matrix):
    """Create correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

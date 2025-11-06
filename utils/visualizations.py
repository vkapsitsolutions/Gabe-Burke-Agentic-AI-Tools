"""Visualization functions for charts and plots."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import io
import base64

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def create_capacity_utilization_chart(df, date_column, attendees_col, capacity_col, title_prefix=""):
    """
    Create capacity utilization percentage chart with DAILY intervals on X-axis.
    Y-axis shows occupied percentage = (attendees / capacity) * 100.
    Optimized for clear visual analysis.
    """
    # Create figure with larger size for clarity
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Ensure date column is datetime
    if df[date_column].dtype != 'datetime64[ns]':
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date and reset index
    df_sorted = df.sort_values(date_column).copy().reset_index(drop=True)
    
    # Calculate occupied percentage (utilization)
    df_sorted['Occupied_Percentage'] = (df_sorted[attendees_col] / df_sorted[capacity_col] * 100).fillna(0)
    
    # Ensure we have daily intervals (resample if needed)
    df_sorted = df_sorted.set_index(date_column)
    
    # Resample to daily frequency (forward fill for missing days)
    df_daily = df_sorted[['Occupied_Percentage', attendees_col, capacity_col]].resample('D').mean()
    df_daily = df_daily.fillna(method='ffill')  # Forward fill missing days
    
    # Reset index to get dates back as column
    df_daily = df_daily.reset_index()
    
    # Create the main line plot with larger markers
    ax.plot(df_daily[date_column], df_daily['Occupied_Percentage'], 
            marker='o', linewidth=3, markersize=7, 
            label='Occupied %', color='#1f77b4', alpha=0.9,
            markerfacecolor='#1f77b4', markeredgecolor='white', markeredgewidth=1.5)
    
    # Add reference lines with labels
    ax.axhline(y=100, color='#d62728', linestyle='--', linewidth=2.5, 
               label='100% (Full Capacity)', alpha=0.8, zorder=1)
    ax.axhline(y=80, color='#ff7f0e', linestyle='--', linewidth=2, 
               label='80% (Optimal Target)', alpha=0.7, zorder=1)
    ax.axhline(y=50, color='#bcbd22', linestyle='--', linewidth=2, 
               label='50% (Low Threshold)', alpha=0.7, zorder=1)
    
    # Add colored zones with transparency
    ax.fill_between(df_daily[date_column], 0, 50, alpha=0.15, color='#d62728', 
                     label='Critical Zone (<50%)', zorder=0)
    ax.fill_between(df_daily[date_column], 50, 80, alpha=0.15, color='#ff7f0e', 
                     label='Moderate Zone (50-80%)', zorder=0)
    ax.fill_between(df_daily[date_column], 80, 100, alpha=0.15, color='#2ca02c', 
                     label='Optimal Zone (80-100%)', zorder=0)
    
    # Highlight over-capacity periods
    over_capacity = df_daily['Occupied_Percentage'] > 100
    if over_capacity.any():
        ax.fill_between(df_daily[date_column], 100, df_daily['Occupied_Percentage'], 
                        where=over_capacity, alpha=0.25, color='#d62728', 
                        label='Over Capacity (>100%)', zorder=0)
    
    # Add trend line
    if len(df_daily) > 1:
        x_numeric = np.arange(len(df_daily))
        z = np.polyfit(x_numeric, df_daily['Occupied_Percentage'], 1)
        p = np.poly1d(z)
        trend_line = p(x_numeric)
        ax.plot(df_daily[date_column], trend_line, 
               linestyle=':', linewidth=2.5, alpha=0.7, color='#9467bd', 
               label='Trend Line', zorder=2)
    
    # Add data labels on key points (max, min, first, last)
    max_idx = df_daily['Occupied_Percentage'].idxmax()
    min_idx = df_daily['Occupied_Percentage'].idxmin()
    
    # Annotate maximum
    max_val = df_daily.loc[max_idx, 'Occupied_Percentage']
    max_date = df_daily.loc[max_idx, date_column]
    ax.annotate(f'Peak: {max_val:.1f}%', 
               xy=(max_date, max_val),
               xytext=(10, 20), textcoords='offset points',
               ha='left', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2))
    
    # Annotate minimum
    min_val = df_daily.loc[min_idx, 'Occupied_Percentage']
    min_date = df_daily.loc[min_idx, date_column]
    ax.annotate(f'Low: {min_val:.1f}%', 
               xy=(min_date, min_val),
               xytext=(10, -20), textcoords='offset points',
               ha='left', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='blue', lw=2))
    
    # Formatting for clarity
    ax.set_title(f'{title_prefix}Daily Capacity Occupied Percentage', 
                fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Date (Daily Intervals)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Occupied Percentage (%)', fontsize=14, fontweight='bold', labelpad=10)
    
    # Legend with better positioning
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95, 
             shadow=True, fancybox=True, ncol=2, bbox_to_anchor=(0, 1))
    
    # Grid for better readability
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Set Y-axis limits with some padding
    y_max = max(df_daily['Occupied_Percentage'].max() + 10, 110)
    ax.set_ylim(0, y_max)
    
    # Format X-axis for daily intervals
    ax.xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    
    # Rotate and align x-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add horizontal lines at each major y-tick for better readability
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    
    # Background color for better contrast
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Add minor gridlines
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Add statistics box
    avg_occupied = df_daily['Occupied_Percentage'].mean()
    stats_text = f'Average: {avg_occupied:.1f}%\nDays: {len(df_daily)}'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    return fig, df_daily['Occupied_Percentage']




def fig_to_download_link(fig, filename="chart.png"):
    """Convert matplotlib figure to downloadable bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


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

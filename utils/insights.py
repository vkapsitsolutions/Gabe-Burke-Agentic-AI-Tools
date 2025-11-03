"""Automated insights generation for visualizations."""

import numpy as np
import pandas as pd
from scipy import stats

def generate_distribution_insights(data):
    """Generate insights for distribution plots."""
    insights = []
    
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    skewness = data.skew()
    
    insights.append(f"📊 Mean: {mean_val:.2f} | Median: {median_val:.2f}")
    insights.append(f"📈 Std Dev: {std_val:.2f} | Range: {data.max() - data.min():.2f}")
    
    if skewness > 1:
        insights.append(f"⚠️ Right-skewed (Skewness: {skewness:.2f}) - Most values on left")
    elif skewness < -1:
        insights.append(f"⚠️ Left-skewed (Skewness: {skewness:.2f}) - Most values on right")
    else:
        insights.append(f"✅ Symmetric distribution (Skewness: {skewness:.2f})")
    
    outliers = data[np.abs((data - mean_val) / std_val) > 3]
    if len(outliers) > 0:
        insights.append(f"🔍 {len(outliers)} outliers detected (>3σ from mean)")
    
    return insights

def generate_boxplot_insights(data):
    """Generate insights for box plots."""
    insights = []
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
    
    insights.append(f"📦 Q1: {Q1:.2f} | Q3: {Q3:.2f} | IQR: {IQR:.2f}")
    insights.append(f"🎯 Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
    
    if len(outliers) > len(data) * 0.1:
        insights.append("⚠️ High outlier count - investigate these values")
    else:
        insights.append("✅ Outlier count is normal")
    
    return insights

def generate_comparison_insights(grouped_data):
    """Generate insights for grouped comparisons."""
    insights = []
    
    total = grouped_data.sum()
    top_item = grouped_data.index[0]
    top_value = grouped_data.values[0]
    top_pct = (top_value / total) * 100
    
    insights.append(f"🏆 Top: {top_item} ({top_value:.2f})")
    insights.append(f"📊 Share: {top_pct:.1f}% of total")
    insights.append(f"📈 Average: {grouped_data.mean():.2f}")
    insights.append(f"📉 Min: {grouped_data.min():.2f} | Max: {grouped_data.max():.2f}")
    
    return insights

def generate_trend_insights(df, date_column, metric_columns):
    """Generate insights from trend analysis."""
    insights = []
    suggestions = []
    
    # Ensure date column is datetime
    if df[date_column].dtype != 'datetime64[ns]':
        df[date_column] = pd.to_datetime(df[date_column])
    
    df_sorted = df.sort_values(date_column)
    
    for metric in metric_columns:
        if metric in df.columns:
            data = df_sorted[metric].fillna(0)
            
            # Calculate trend using linear regression
            x = np.arange(len(data))
            if len(data) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
                
                # Determine trend direction
                if slope > 0:
                    direction = "📈 Increasing"
                    trend_type = "upward"
                elif slope < 0:
                    direction = "📉 Decreasing"
                    trend_type = "downward"
                else:
                    direction = "➡️ Stable"
                    trend_type = "stable"
                
                # Calculate change percentage
                if len(data) > 0 and data.iloc[0] != 0:
                    change_pct = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0]) * 100
                else:
                    change_pct = 0
                
                insights.append(f"{direction} trend for {metric} (Change: {change_pct:+.1f}%)")
                
                # Strength of trend
                r_squared = r_value ** 2
                if r_squared > 0.7:
                    insights.append(f"💪 Strong {trend_type} trend (R²: {r_squared:.2f})")
                elif r_squared > 0.3:
                    insights.append(f"📊 Moderate {trend_type} trend (R²: {r_squared:.2f})")
                else:
                    insights.append(f"📉 Weak trend (R²: {r_squared:.2f}) - highly variable")
    
    return insights

def generate_utilization_insights(df, attendees_col, capacity_col):
    """Generate utilization insights and actionable suggestions."""
    insights = []
    suggestions = []
    
    # Calculate utilization rate
    df['Utilization_Rate'] = (df[attendees_col] / df[capacity_col] * 100).fillna(0)
    
    avg_utilization = df['Utilization_Rate'].mean()
    max_utilization = df['Utilization_Rate'].max()
    min_utilization = df['Utilization_Rate'].min()
    
    # Insights
    insights.append(f"📊 Average Utilization: {avg_utilization:.1f}%")
    insights.append(f"⬆️ Peak Utilization: {max_utilization:.1f}%")
    insights.append(f"⬇️ Lowest Utilization: {min_utilization:.1f}%")
    
    # Count days by utilization level
    low_util_days = len(df[df['Utilization_Rate'] < 50])
    optimal_util_days = len(df[(df['Utilization_Rate'] >= 50) & (df['Utilization_Rate'] <= 80)])
    high_util_days = len(df[df['Utilization_Rate'] > 80])
    
    insights.append(f"🔴 Low utilization days (<50%): {low_util_days}")
    insights.append(f"🟢 Optimal utilization days (50-80%): {optimal_util_days}")
    insights.append(f"🟡 High utilization days (>80%): {high_util_days}")
    
    # Generate actionable suggestions
    if avg_utilization < 50:
        suggestions.append("⚠️ Low average utilization - Consider reducing capacity or increasing promotion efforts")
        suggestions.append("💡 Suggestion: Run targeted campaigns to boost attendance")
        suggestions.append("📊 Action: Analyze booking patterns to identify slow periods")
    
    elif avg_utilization > 85:
        suggestions.append("🚨 High utilization - Risk of overcrowding and poor experience")
        suggestions.append("💡 Suggestion: Increase capacity or implement time-slot management")
        suggestions.append("📊 Action: Consider expansion or multiple sessions")
    
    else:
        suggestions.append("✅ Healthy utilization rate - Good balance between capacity and demand")
        suggestions.append("💡 Suggestion: Maintain current operations and monitor trends")
    
    # Trend-based suggestions
    if len(df) > 1:
        recent_util = df['Utilization_Rate'].tail(3).mean()
        earlier_util = df['Utilization_Rate'].head(3).mean()
        
        if recent_util > earlier_util * 1.2:
            suggestions.append("📈 Growing demand detected - Plan for capacity expansion")
        elif recent_util < earlier_util * 0.8:
            suggestions.append("📉 Declining attendance - Review scheduling and marketing strategies")
    
    # Variability analysis
    utilization_std = df['Utilization_Rate'].std()
    if utilization_std > 25:
        suggestions.append("📊 High variability in utilization - Consider dynamic pricing or flexible scheduling")
    
    return insights, suggestions

def generate_comparison_suggestions(df, group_column, metric_column):
    """Generate suggestions based on group comparisons."""
    suggestions = []
    
    grouped = df.groupby(group_column)[metric_column].agg(['sum', 'mean', 'count'])
    
    # Find best and worst performers
    best_performer = grouped['sum'].idxmax()
    worst_performer = grouped['sum'].idxmin()
    
    best_value = grouped.loc[best_performer, 'sum']
    worst_value = grouped.loc[worst_performer, 'sum']
    
    suggestions.append(f"🏆 Best performer: {best_performer} ({best_value:.0f} total)")
    suggestions.append(f"📉 Needs improvement: {worst_performer} ({worst_value:.0f} total)")
    
    # Performance gap
    gap = ((best_value - worst_value) / best_value) * 100
    if gap > 30:
        suggestions.append(f"⚠️ Large performance gap ({gap:.1f}%) - Review operations at underperforming locations")
        suggestions.append("💡 Action: Share best practices from top performers with struggling locations")
    
    return suggestions

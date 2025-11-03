"""Data loading and preprocessing utilities."""

import pandas as pd

def fix_date_columns(df):
    """Convert date columns to proper datetime format for Arrow compatibility."""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except:
                pass
    return df

def load_data(uploaded_file):
    """Load data from uploaded file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        df = fix_date_columns(df)
        return df, None
    except Exception as e:
        return None, str(e)

def get_grouped_data(df, group_column):
    """Get unique groups from a column."""
    if group_column in df.columns:
        return sorted(df[group_column].dropna().unique())
    return []

def filter_by_group(df, group_column, group_value):
    """Filter dataframe by group value."""
    return df[df[group_column] == group_value].copy()

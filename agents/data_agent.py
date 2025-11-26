"""LangChain agent tools for data analysis."""

from langchain.tools import tool
from langchain.agents import create_agent
import pandas as pd
from config import MODEL_NAME

def create_data_analysis_agent(df, filename, model_name=MODEL_NAME):
    """Create LangChain agent with data analysis tools."""
    
    @tool
    def get_dataset_overview() -> str:
        """Get comprehensive dataset overview."""
        return f"""📊 Dataset: {filename}
        
🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns
📋 Columns: {', '.join(df.columns.tolist())}
💾 Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB
⚠️ Missing: {df.isnull().sum().sum()} total null values"""

    @tool
    def get_column_analysis(column_name: str) -> str:
        """Analyze a specific column in detail."""
        if column_name not in df.columns:
            return f"Column not found. Available: {', '.join(df.columns.tolist())}"
        
        col = df[column_name]
        result = f"""📋 Column: {column_name}
🏷️ Type: {col.dtype}
🔢 Values: {len(col)} | Missing: {col.isnull().sum()}
🎯 Unique: {col.nunique()}"""
        
        if col.dtype in ['int64', 'float64']:
            result += f"""
📊 Stats:
  - Mean: {col.mean():.2f}
  - Median: {col.median():.2f}
  - Std: {col.std():.2f}
  - Range: [{col.min():.2f}, {col.max():.2f}]"""
        else:
            top = col.value_counts().head(5)
            result += f"\n🔝 Top 5: {dict(top)}"
        
        return result

    @tool
    def compare_groups(group_column: str, value_column: str, metric: str = "sum") -> str:
        """Compare groups using aggregation. Metrics: sum, mean, count, max, min."""
        if group_column not in df.columns or value_column not in df.columns:
            return "Column(s) not found"
        
        try:
            result = df.groupby(group_column)[value_column].agg(metric).sort_values(ascending=False)
            return f"""📊 Grouped by {group_column}, {metric} of {value_column}:

{result.head(10).to_string()}

📈 Summary:
  - Total: {result.sum():.2f}
  - Average: {result.mean():.2f}
  - Top: {result.index[0]} ({result.values[0]:.2f})"""
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def filter_data(column: str, operator: str, value: str) -> str:
        """Filter data. Operators: equals, >, <, contains."""
        if column not in df.columns:
            return "Column not found"
        
        try:
            if operator == "equals":
                filtered = df[df[column] == value]
            elif operator == ">":
                filtered = df[df[column] > float(value)]
            elif operator == "<":
                filtered = df[df[column] < float(value)]
            elif operator == "contains":
                filtered = df[df[column].astype(str).str.contains(value, case=False, na=False)]
            else:
                return "Invalid operator. Use: equals, >, <, contains"
            
            return f"""🔍 Filter: {column} {operator} {value}
✅ Results: {len(filtered)} rows ({len(filtered)/len(df)*100:.1f}%)

📋 Sample:
{filtered.head(5).to_string()}"""
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def get_statistical_summary() -> str:
        """Get statistical summary of numeric columns."""
        return f"📊 Statistics:\n{df.describe().to_string()}"

    tools = [
        get_dataset_overview,
        get_column_analysis,
        compare_groups,
        filter_data,
        get_statistical_summary
    ]
    
    system_prompt = f"""You are an expert data analyst with access to: {filename}

Dataset: {df.shape[0]} rows × {df.shape[1]} columns
Columns: {', '.join(df.columns.tolist())}

Your abilities:
- Analyze data using available tools
- Remember conversation context
- Provide actionable insights
- Explain findings clearly

Always provide comprehensive, context-aware answers."""
    
    agent = create_agent(
        model=model_name,
        tools=tools,
        system_prompt=system_prompt
    )
    
    return agent


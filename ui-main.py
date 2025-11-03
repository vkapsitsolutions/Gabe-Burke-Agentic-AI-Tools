import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Set style for better-looking graphs
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []  # ChatGPT-like persistent memory
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'insights_cache' not in st.session_state:
    st.session_state.insights_cache = {}

# Helper functions
def fix_date_columns(df):
    """Convert date columns to proper datetime format for Arrow compatibility."""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except:
                pass
    return df

def generate_chart_insights(df, chart_type, column_name, column2=None):
    """Generate automated insights for charts using AI analysis."""
    insights = []
    
    if chart_type == "distribution":
        data = df[column_name].dropna()
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        skewness = data.skew()
        
        insights.append(f"📊 Mean: {mean_val:.2f} | Median: {median_val:.2f}")
        insights.append(f"📈 Standard Deviation: {std_val:.2f}")
        
        if skewness > 1:
            insights.append(f"⚠️ Right-skewed distribution (Skewness: {skewness:.2f})")
            insights.append("Most values are concentrated on the left with a long tail on the right.")
        elif skewness < -1:
            insights.append(f"⚠️ Left-skewed distribution (Skewness: {skewness:.2f})")
            insights.append("Most values are concentrated on the right with a long tail on the left.")
        else:
            insights.append(f"✅ Nearly symmetric distribution (Skewness: {skewness:.2f})")
        
        outliers = data[np.abs((data - mean_val) / std_val) > 3]
        if len(outliers) > 0:
            insights.append(f"🔍 {len(outliers)} potential outliers detected (>3 std from mean)")
    
    elif chart_type == "boxplot":
        data = df[column_name].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
        
        insights.append(f"📦 Q1 (25%): {Q1:.2f} | Q3 (75%): {Q3:.2f}")
        insights.append(f"📏 Interquartile Range (IQR): {IQR:.2f}")
        insights.append(f"🎯 Outliers: {len(outliers)} data points ({len(outliers)/len(data)*100:.1f}%)")
        
        if len(outliers) > len(data) * 0.1:
            insights.append("⚠️ High number of outliers - consider investigating these values")
    
    elif chart_type == "correlation":
        if column2:
            corr = df[column_name].corr(df[column2])
            insights.append(f"🔗 Correlation: {corr:.4f}")
            
            if abs(corr) > 0.7:
                insights.append("💪 Strong correlation - Variables move together significantly")
            elif abs(corr) > 0.3:
                insights.append("📊 Moderate correlation - Some relationship exists")
            else:
                insights.append("📉 Weak correlation - Little to no linear relationship")
            
            if corr > 0:
                insights.append("↗️ Positive relationship: As one increases, the other tends to increase")
            else:
                insights.append("↘️ Negative relationship: As one increases, the other tends to decrease")
    
    return insights

def create_enhanced_distribution_plot(df, column):
    """Create distribution plot with insights."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    data = df[column].dropna()
    
    # Histogram
    ax1.hist(data, bins=30, edgecolor='black', alpha=0.7, color='#1f77b4')
    ax1.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
    ax1.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
    ax1.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # KDE Plot
    data.plot(kind='kde', ax=ax2, color='#1f77b4', linewidth=2)
    ax2.set_title(f'Density Plot of {column}', fontsize=14, fontweight='bold')
    ax2.set_xlabel(column)
    ax2.set_ylabel('Density')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_enhanced_boxplot(df, column):
    """Create enhanced boxplot with statistical annotations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(df[column].dropna(), patch_artist=True, vert=True,
                     boxprops=dict(facecolor='lightblue', color='blue'),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='blue', linewidth=1.5),
                     capprops=dict(color='blue', linewidth=1.5))
    
    # Add statistics
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    median = df[column].median()
    
    ax.text(1.15, median, f'Median: {median:.2f}', verticalalignment='center', fontsize=10, color='red')
    ax.text(1.15, Q1, f'Q1: {Q1:.2f}', verticalalignment='center', fontsize=9)
    ax.text(1.15, Q3, f'Q3: {Q3:.2f}', verticalalignment='center', fontsize=9)
    
    ax.set_title(f'Box Plot Analysis of {column}', fontsize=14, fontweight='bold')
    ax.set_ylabel(column)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

# Function to create tools dynamically
def create_data_tools(df):
    """Create tools that operate on the current dataframe with chat context."""
    
    @tool
    def get_dataframe_info() -> str:
        """Get comprehensive information about the dataset including shape, columns, data types, and missing values."""
        if df is None:
            return "No dataframe loaded."
        
        info = f"""📊 Dataset Information:
        
🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns

📋 Columns: {', '.join(df.columns.tolist())}

🏷️ Data Types:
{df.dtypes.to_string()}

⚠️ Missing Values:
{df.isnull().sum().to_string()}

💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB"""
        return info

    @tool
    def get_dataframe_sample(n: int = 10) -> str:
        """Get sample rows from the dataframe. Specify number of rows (default 10)."""
        if df is None:
            return "No dataframe loaded."
        return f"Sample {n} rows:\n{df.head(n).to_string()}"

    @tool
    def get_statistical_summary() -> str:
        """Get detailed statistical summary of all numeric columns."""
        if df is None:
            return "No dataframe loaded."
        return f"Statistical Summary:\n{df.describe().to_string()}"

    @tool
    def get_column_info(column_name: str) -> str:
        """Get detailed information about a specific column including unique values, data type, and statistics."""
        if df is None:
            return "No dataframe loaded."
        
        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available columns: {', '.join(df.columns.tolist())}"
        
        col = df[column_name]
        info = f"""📋 Column '{column_name}' Details:

🏷️ Data Type: {col.dtype}
🔢 Total Values: {len(col)}
⚠️ Missing Values: {col.isnull().sum()} ({col.isnull().sum()/len(col)*100:.1f}%)
🎯 Unique Values: {col.nunique()}"""
        
        if col.dtype in ['int64', 'float64']:
            info += f"""

📊 Statistics:
- Mean: {col.mean():.2f}
- Median: {col.median():.2f}
- Std Dev: {col.std():.2f}
- Min: {col.min():.2f}
- Max: {col.max():.2f}"""
        else:
            top_values = col.value_counts().head(10)
            info += f"\n\n🔝 Top 10 Values:\n{top_values.to_string()}"
        
        return info

    @tool
    def analyze_correlation(column1: str, column2: str) -> str:
        """Analyze correlation between two numeric columns with interpretation."""
        if df is None:
            return "No dataframe loaded."
        
        if column1 not in df.columns or column2 not in df.columns:
            return "One or both columns not found."
        
        try:
            col1 = pd.to_numeric(df[column1], errors='coerce')
            col2 = pd.to_numeric(df[column2], errors='coerce')
            corr = col1.corr(col2)
            
            interpretation = ""
            if abs(corr) > 0.7:
                strength = "Strong"
            elif abs(corr) > 0.3:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            direction = "positive" if corr > 0 else "negative"
            
            result = f"""🔗 Correlation Analysis:

📊 Correlation Coefficient: {corr:.4f}
💪 Strength: {strength} {direction} correlation

📝 Interpretation:"""
            
            if abs(corr) > 0.7:
                result += f"\nThe variables have a strong {direction} relationship. "
                result += "When one increases, the other tends to " + ("increase" if corr > 0 else "decrease") + " significantly."
            elif abs(corr) > 0.3:
                result += f"\nThe variables have a moderate {direction} relationship. "
                result += "There is some tendency for them to move together."
            else:
                result += "\nThe variables have little to no linear relationship. "
                result += "Changes in one variable don't reliably predict changes in the other."
            
            return result
        except Exception as e:
            return f"Error calculating correlation: {str(e)}"

    @tool
    def filter_and_analyze(column_name: str, condition: str, value: str) -> str:
        """Filter dataframe and provide analysis. Conditions: 'equals', 'greater_than', 'less_than', 'contains'."""
        if df is None:
            return "No dataframe loaded."
        
        if column_name not in df.columns:
            return f"Column '{column_name}' not found."
        
        try:
            if condition == 'equals':
                filtered = df[df[column_name] == value]
            elif condition == 'greater_than':
                filtered = df[df[column_name] > float(value)]
            elif condition == 'less_than':
                filtered = df[df[column_name] < float(value)]
            elif condition == 'contains':
                filtered = df[df[column_name].astype(str).str.contains(value, case=False, na=False)]
            else:
                return "Invalid condition. Use: equals, greater_than, less_than, or contains"
            
            result = f"""🔍 Filter Results:

📊 Condition: {column_name} {condition} {value}
✅ Matched Rows: {len(filtered)} out of {len(df)} ({len(filtered)/len(df)*100:.1f}%)

📋 Sample Results:
{filtered.head(10).to_string()}"""
            
            return result
        except Exception as e:
            return f"Error filtering: {str(e)}"

    @tool
    def group_and_summarize(group_column: str, agg_column: str, agg_function: str = 'sum') -> str:
        """Group data and aggregate with analysis. Functions: 'sum', 'mean', 'count', 'max', 'min'."""
        if df is None:
            return "No dataframe loaded."
        
        if group_column not in df.columns or agg_column not in df.columns:
            return "One or both columns not found."
        
        try:
            result = df.groupby(group_column)[agg_column].agg(agg_function).sort_values(ascending=False)
            
            analysis = f"""📊 Group Analysis:

🔄 Grouped by: {group_column}
📈 Aggregated: {agg_function}({agg_column})
🎯 Total Groups: {len(result)}

📋 Top 10 Results:
{result.head(10).to_string()}

📊 Summary Statistics:
- Total: {result.sum():.2f}
- Average: {result.mean():.2f}
- Max: {result.max():.2f}
- Min: {result.min():.2f}"""
            
            return analysis
        except Exception as e:
            return f"Error in grouping: {str(e)}"
    
    return [
        get_dataframe_info,
        get_dataframe_sample,
        get_statistical_summary,
        get_column_info,
        analyze_correlation,
        filter_and_analyze,
        group_and_summarize
    ]

# Title
st.markdown('<p class="main-header">🤖 AI Data Analysis Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload data, get AI-powered insights, and chat like ChatGPT with persistent memory</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your data file for AI-powered analysis"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            df = fix_date_columns(df)
            
            st.session_state.df = df
            st.session_state.uploaded_file_name = uploaded_file.name
            
            # Reset only agent, keep chat history for context
            if st.session_state.agent is None:
                st.session_state.chat_messages = []
            st.session_state.agent = None
            st.session_state.insights_cache = {}
            
            st.success(f"✅ Loaded successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    st.subheader("ℹ️ About This App")
    st.markdown("""
    Features:
    - 📊 Smart visualizations with AI insights
    - 💬 ChatGPT-like conversation memory
    - 🤖 Powered by LangChain v1.0
    - 🔍 Automated data analysis
    
    Not RAG:
    This uses agentic tools to directly query your data, not retrieval from vector stores.
    """)

# Main content
if st.session_state.df is not None:
    df = st.session_state.df
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Smart Visualizations", "💬 Chat Assistant"])
    
    # Tab 1: Data Overview
    with tab1:
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), width='stretch')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h3>📊 Rows</h3><h2>' + f'{len(df):,}' + '</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>📋 Columns</h3><h2>' + f'{len(df.columns)}' + '</h2></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>⚠️ Missing</h3><h2>' + f'{df.isnull().sum().sum()}' + '</h2></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><h3>💾 Size</h3><h2>' + f'{df.memory_usage(deep=True).sum()/1024:.1f} KB' + '</h2></div>', unsafe_allow_html=True)
        
        st.subheader("🔢 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values,
            'Unique': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, width='stretch', hide_index=True)
        
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe(), width='stretch')
    
    # Tab 2: Smart Visualizations
    with tab2:
        st.subheader("📈 AI-Powered Visualizations with Insights")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_cols) > 0:
            # Distribution Analysis
            st.markdown("### 📊 Distribution Analysis")
            selected_dist = st.selectbox("Select numeric column for distribution", numeric_cols, key='dist')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = create_enhanced_distribution_plot(df, selected_dist)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("#### 🔍 Automated Insights")
                insights = generate_chart_insights(df, "distribution", selected_dist)
                for insight in insights:
                    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            
            st.divider()
            
            # Box Plot Analysis
            st.markdown("### 📦 Outlier Detection Analysis")
            selected_box = st.selectbox("Select column for outlier analysis", numeric_cols, key='box')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = create_enhanced_boxplot(df, selected_box)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("#### 🔍 Automated Insights")
                insights = generate_chart_insights(df, "boxplot", selected_box)
                for insight in insights:
                    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            
            st.divider()
        
        # Categorical Analysis
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            st.markdown("### 📊 Categorical Comparison")
            
            col1, col2 = st.columns(2)
            with col1:
                cat_col = st.selectbox("Select categorical column", categorical_cols)
            with col2:
                num_col = st.selectbox("Select numeric column to analyze", numeric_cols, key='cat_num')
            
            grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(15)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(12, 6))
                grouped.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black', width=0.7)
                ax.set_title(f'{num_col} by {cat_col} (Top 15)', fontsize=14, fontweight='bold')
                ax.set_xlabel(cat_col, fontsize=12)
                ax.set_ylabel(num_col, fontsize=12)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for i, v in enumerate(grouped.values):
                    ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("#### 🔍 Key Insights")
                total = grouped.sum()
                top_category = grouped.index[0]
                top_value = grouped.values[0]
                top_pct = (top_value / total) * 100
                
                st.markdown(f'<div class="insight-box">🏆 <b>Top Category:</b> {top_category}<br>💰 <b>Value:</b> {top_value:.2f}<br>📊 <b>Percentage:</b> {top_pct:.1f}% of total</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-box">📈 <b>Total:</b> {total:.2f}<br>📊 <b>Average:</b> {grouped.mean():.2f}<br>🎯 <b>Categories:</b> {len(grouped)}</div>', unsafe_allow_html=True)
            
            st.divider()
        
        # Correlation Heatmap
        if len(numeric_cols) > 1:
            st.markdown("### 🔥 Correlation Matrix")
            
            corr = df[numeric_cols].corr()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                           square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                           ax=ax, vmin=-1, vmax=1)
                ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("#### 🔍 Key Correlations")
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
                
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                for col1, col2, corr_val in corr_pairs[:5]:
                    strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.3 else "Weak"
                    direction = "positive" if corr_val > 0 else "negative"
                    st.markdown(f'<div class="insight-box">📊 <b>{col1}</b> ↔️ <b>{col2}</b><br>🔗 {corr_val:.3f} ({strength} {direction})</div>', unsafe_allow_html=True)
    
    # Tab 3: ChatGPT-like Chat Interface
    with tab3:
        st.subheader("💬 AI Data Assistant (ChatGPT-style)")
        st.caption("🧠 Persistent conversation memory - I remember our entire conversation!")
        
        # Initialize agent with chat history support
        if st.session_state.agent is None:
            with st.spinner("🤖 Initializing AI agent with memory..."):
                try:
                    tools = create_data_tools(df)
                    
                    system_prompt = f"""You are an expert data analyst assistant with access to a dataset. You have persistent memory of our conversation.

Dataset Details:
- Filename: {st.session_state.uploaded_file_name}
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}

Your Role:
- Provide detailed, insightful analysis
- Use tools to query the data
- Remember previous conversation context
- Give actionable recommendations
- Explain findings clearly

Always provide comprehensive answers with context from our conversation history."""
                    
                    st.session_state.agent = create_agent(
                        model="gpt-4o-mini",
                        tools=tools,
                        system_prompt=system_prompt
                    )
                    st.success("✅ AI Agent ready with conversation memory!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display chat history (ChatGPT-style)
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask anything about your data... I remember our conversation! 💬"):
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get AI response with full conversation history
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Build message history for LangChain
                        messages = []
                        for msg in st.session_state.chat_messages:
                            if msg["role"] == "user":
                                messages.append(HumanMessage(content=msg["content"]))
                            else:
                                messages.append(AIMessage(content=msg["content"]))
                        
                        # Invoke agent with full history
                        response = st.session_state.agent.invoke({"messages": messages})
                        
                        # Extract answer
                        final_message = response["messages"][-1]
                        answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
                        
                        st.write(answer)
                        
                        # Save to history
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
        
        # Chat controls
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()
        with col2:
            if st.button("💾 Export Chat"):
                chat_export = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_messages])
                st.download_button("Download", chat_export, "chat_history.txt", "text/plain")
        
        with st.expander("💡 Example Questions"):
            st.markdown("""
            Basic Questions:
            - What's in this dataset?
            - Show me the first 10 rows
            - What are the data types of each column?
            
            Analysis Questions:
            - What's the correlation between [column1] and [column2]?
            - Show me statistics for [column_name]
            - What are the top 5 values in [column]?
            
            Follow-up Questions (I remember context!):
            - Tell me more about that
            - What about the other columns?
            - Can you explain that in simpler terms?
            
            Complex Questions:
            - Filter data where [column] > [value] and analyze it
            - Group by [category] and sum [numeric_column]
            - Find outliers in [column]
            """)

else:
    # Welcome screen
    st.info("👈 Upload a CSV or Excel file to get started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Smart Visualizations
        - Distribution plots with AI insights
        - Outlier detection
        - Correlation analysis
        - Automated interpretations
        """)
    
    with col2:
        st.markdown("""
        ### 💬 ChatGPT-like Chat
        - Persistent conversation memory
        - Context-aware responses
        - Natural language queries
        - Export chat history
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 Not RAG - It's Better!
        - Direct data querying with tools
        - No vector embeddings needed
        - Real-time analysis
        - LangChain v1.0 agents
        """)

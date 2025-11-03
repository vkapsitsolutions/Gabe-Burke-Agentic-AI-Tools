"""Main Streamlit application with modular structure."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, AIMessage

# Import custom modules
from config import *
from utils.styling import apply_custom_css
from utils.data_loader import load_data, get_grouped_data, filter_by_group
from utils.visualizations import (
    create_distribution_plot, create_boxplot, 
    create_bar_chart, create_correlation_heatmap,
    create_utilization_trend
)
from utils.insights import (
    generate_distribution_insights, generate_boxplot_insights,
    generate_comparison_insights, generate_comparison_suggestions,
    generate_utilization_insights, generate_trend_insights
)
from agents.data_agent import create_data_analysis_agent

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_custom_css()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'group_column' not in st.session_state:
    st.session_state.group_column = None

# Title
st.markdown(f'<p class="main-header">{APP_TITLE}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{APP_SUBTITLE}</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        df, error = load_data(uploaded_file)
        
        if error:
            st.error(f"Error: {error}")
        else:
            st.session_state.df = df
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.agent = None
            
            st.success("Loaded! Below select grouping column for analysis.")
            
            # Group column selector
            st.divider()
            st.subheader("Group Analysis")

            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                group_col = st.selectbox(
                    "Select grouping column",
                    ['None'] + categorical_cols,
                    help="Analyze data by groups (e.g., Building 1, Building 2)"
                )
                
                if group_col != 'None':
                    st.session_state.group_column = group_col
                    groups = get_grouped_data(df, group_col)
                    st.info(f"Found {len(groups)} unique groups: {', '.join(map(str, groups[:5]))}" + 
                           (f"... +{len(groups)-5} more" if len(groups) > 5 else ""))
                else:
                    st.session_state.group_column = None
    
    # st.divider()
    # st.markdown("""
    # ✨ Features:
    # - Group analysis 
    # - ChatGPT-like interface
    # - AI-powered insights
    # - Modular architecture
    # """)

# Main content
if st.session_state.df is not None:
    df = st.session_state.df
    group_col = st.session_state.group_column
    
    tab1, tab2, tab3 = st.tabs([
        "Data Overview", 
        "Visualizations" + (" (Grouped)" if group_col else ""), 
        "Chat Assistant"
    ])
    
    # TAB 1: Data Overview
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), width='stretch')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><h2>{len(df):,}</h2><p>Rows</p></div>', 
                       unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h2>{len(df.columns)}</h2><p>Columns</p></div>', 
                       unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h2>{df.isnull().sum().sum()}</h2><p>Missing</p></div>', 
                       unsafe_allow_html=True)
        with col4:
            memory_kb = df.memory_usage(deep=True).sum() / 1024
            st.markdown(f'<div class="metric-card"><h2>{memory_kb:.1f}</h2><p>KB</p></div>', 
                       unsafe_allow_html=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), width='stretch')

    # TAB 2: Visualizations with Trend Analysis
    with tab2:
        if group_col:
            st.markdown(f'<div class="group-header">Analyzing by: {group_col}</div>', 
                       unsafe_allow_html=True)
            
            groups = get_grouped_data(df, group_col)
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Detect date column
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            date_column = date_cols[0] if date_cols else None
            
            if numeric_cols and date_column:
                # Metric selection with validation
                col1, col2 = st.columns(2)
                with col1:
                    primary_metric = st.selectbox("Primary metric (e.g., Attendees)", 
                                                 numeric_cols, key='primary')
                with col2:
                    # Filter out primary metric from secondary options
                    secondary_options = [col for col in numeric_cols if col != primary_metric]
                    if secondary_options:
                        secondary_metric = st.selectbox("Secondary metric (e.g., Capacity)", 
                                                       secondary_options, key='secondary')
                    else:
                        secondary_metric = None
                        st.warning("Not enough numeric columns for comparison")
                
                # Check if we have both metrics
                if secondary_metric is None:
                    st.error("Please ensure you have at least 2 numeric columns in your dataset")
                else:
                    st.divider()
                    
                    # Overall comparison first
                    st.markdown("### Overall Comparison Across All Groups")
                    
                    comparison = df.groupby(group_col)[primary_metric].sum().sort_values(ascending=False)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = create_bar_chart(comparison, group_col, primary_metric, 
                                              title_prefix="Total ")
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("#### Key Findings")
                        insights = generate_comparison_insights(comparison)
                        for insight in insights:
                            st.markdown(f'<div class="insight-box">{insight}</div>', 
                                      unsafe_allow_html=True)
                        
                        st.markdown("#### Recommendations")
                        suggestions = generate_comparison_suggestions(df, group_col, primary_metric)
                        for suggestion in suggestions:
                            st.markdown(f'<div class="insight-box">{suggestion}</div>', 
                                      unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Individual group analysis with trends
                    st.markdown("### Detailed Trend Analysis by Group")
                    
                    group_tabs = st.tabs([str(g) for g in groups])
                    
                    for idx, group_value in enumerate(groups):
                        with group_tabs[idx]:
                            group_df = filter_by_group(df, group_col, group_value)
                            
                            st.markdown(f"## 🏢 {group_value} - Comprehensive Analysis")
                            st.caption(f"Analyzing {len(group_df)} data points")
                            
                            # Check if group has data
                            if len(group_df) == 0:
                                st.warning(f" No data available for {group_value}")
                                continue
                            
                            # TREND ANALYSIS
                            st.markdown("### Attendance vs Capacity Trends Over Time")
                            
                            try:
                                fig = create_utilization_trend(
                                    group_df, date_column, primary_metric, secondary_metric,
                                    title_prefix=f"{group_value} - "
                                )
                                st.pyplot(fig)
                                plt.close()
                            except Exception as e:
                                st.error(f"Error creating trend chart: {str(e)}")
                            
                            # Trend insights and suggestions
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### 🔍 Trend Insights")
                                try:
                                    trend_insights = generate_trend_insights(
                                        group_df, date_column, [primary_metric, secondary_metric]
                                    )
                                    for insight in trend_insights:
                                        st.markdown(f'<div class="insight-box">{insight}</div>', 
                                                  unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error generating insights: {str(e)}")
                            
                            with col2:
                                st.markdown("#### 💡 Actionable Suggestions")
                                try:
                                    util_insights, util_suggestions = generate_utilization_insights(
                                        group_df, primary_metric, secondary_metric
                                    )
                                    for suggestion in util_suggestions:
                                        st.markdown(f'<div class="insight-box">{suggestion}</div>', 
                                                  unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error generating suggestions: {str(e)}")
                            
                            st.divider()
                            
                            # Statistical distribution
                            st.markdown(f"### 📊 Statistical Distribution - {primary_metric}")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                try:
                                    data = group_df[primary_metric].dropna()
                                    if len(data) > 0:
                                        fig = create_distribution_plot(data, primary_metric, 
                                                                      title_prefix=f"{group_value} - ")
                                        st.pyplot(fig)
                                        plt.close()
                                    else:
                                        st.warning(f"No data available for {primary_metric}")
                                except Exception as e:
                                    st.error(f"Error creating distribution plot: {str(e)}")
                            
                            with col2:
                                st.markdown("#### 🔍 Distribution Insights")
                                try:
                                    data = group_df[primary_metric].dropna()
                                    if len(data) > 0:
                                        insights = generate_distribution_insights(data)
                                        for insight in insights:
                                            st.markdown(f'<div class="insight-box">{insight}</div>', 
                                                      unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error generating insights: {str(e)}")
                            
                            st.divider()
                            
                            # Summary statistics - FIX FOR DUPLICATE COLUMNS
                            st.markdown("### Summary Statistics")
                            
                            try:
                                # Create unique list of columns to analyze
                                columns_to_analyze = [primary_metric]
                                if secondary_metric and secondary_metric != primary_metric:
                                    columns_to_analyze.append(secondary_metric)
                                
                                # Get statistics
                                summary_df = group_df[columns_to_analyze].describe()
                                
                                # Display with custom formatting
                                st.dataframe(summary_df.style.format("{:.2f}"), width='stretch')
                                
                                # Additional context
                                st.caption(f"Statistics calculated for {len(group_df)} data points")
                                
                            except Exception as e:
                                st.error(f"Error displaying summary statistics: {str(e)}")
                                st.write("Debug info:", columns_to_analyze)
            
            else:
                if not numeric_cols:
                    st.warning("No numeric columns found for analysis")
                if not date_column:
                    st.warning("No date column found. Add a column with 'date' in the name for time series analysis")
        
        else:
            st.info("Select a grouping column in the sidebar to see grouped trend analysis")
            
            # Show basic visualization without grouping
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numeric_cols:
                selected = st.selectbox("Select column to analyze", numeric_cols)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    data = df[selected].dropna()
                    fig = create_distribution_plot(data, selected)
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("#### 🔍 Insights")
                    insights = generate_distribution_insights(data)
                    for insight in insights:
                        st.markdown(f'<div class="insight-box">{insight}</div>', 
                                  unsafe_allow_html=True)

    # TAB 3: Chat Assistant
    with tab3:
        st.subheader("Assistant")
        st.caption("Ask anything about your data!")
        
        # Initialize agent
        if st.session_state.agent is None:
            with st.spinner("Initializing..."):
                try:
                    st.session_state.agent = create_data_analysis_agent(
                        df, 
                        st.session_state.uploaded_file_name,
                        MODEL_NAME
                    )
                    # st.success("✅ Ready!")
                except Exception as e:
                    st.error(f" Error: {str(e)}")
        
        # Chat history container
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="chat-history">', unsafe_allow_html=True)
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input 
        prompt = st.chat_input(CHAT_INPUT_PLACEHOLDER)
        
        if prompt:
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Get response
            with st.spinner():
                try:
                    messages = []
                    for msg in st.session_state.chat_messages:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))
                    
                    response = st.session_state.agent.invoke({"messages": messages})
                    answer = response["messages"][-1].content
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f" Error: {str(e)}")
        
        # Controls
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()
        with col2:
            if st.session_state.chat_messages:
                chat_text = "\n\n".join([
                    f"{m['role'].upper()}: {m['content']}" 
                    for m in st.session_state.chat_messages
                ])
                st.download_button("Export Chat", chat_text, "chat.txt")

else:
    st.info("<- Upload a file to begin")
    
    col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     st.markdown("""
    #     ### 📊 Grouped Analysis
    #     - Analyze by Building 1, 2, 3...
    #     - Compare across groups
    #     - Individual insights
    #     """)
    
    # with col2:
    #     st.markdown("""
    #     ### 💬 Fixed Chat UI
    #     - Input always at bottom
    #     - Scrollable history
    #     - Persistent memory
    #     """)
    
    # with col3:
    #     st.markdown("""
    #     ### 🏗️ Modular Code
    #     - Organized structure
    #     - Easy to maintain
    #     - Reusable components
    #     """)

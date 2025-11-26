"""Main Streamlit application with modular structure."""

import streamlit as st
import pandas as pd 
import io
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Import custom modules
from config import (
    APP_TITLE, APP_SUBTITLE, 
    CHAT_INPUT_PLACEHOLDER, MODEL_NAME
)
from utils.styling import apply_custom_css
from utils.data_loader import load_data, get_grouped_data, filter_by_group
from utils.visualizations import (
    create_distribution_plot, create_boxplot, 
    create_bar_chart, create_correlation_heatmap,
    create_utilization_trend, fig_to_download_link,
    create_capacity_utilization_chart
)
from utils.insights import (
    generate_distribution_insights, generate_boxplot_insights,
    generate_comparison_insights, generate_comparison_suggestions,
    generate_utilization_insights, generate_trend_insights,
    generate_utilization_percentage_insights
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
        
        # Get numeric columns for totals calculation
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><h2>{len(df):,}</h2><p>Total Rows</p></div>', 
                    unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h2>{len(df.columns)}</h2><p>Columns</p></div>', 
                    unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h2>{df.isnull().sum().sum()}</h2><p>Missing Values</p></div>', 
                    unsafe_allow_html=True)
        with col4:
            memory_kb = df.memory_usage(deep=True).sum() / 1024
            st.markdown(f'<div class="metric-card"><h2>{memory_kb:.1f}</h2><p>KB</p></div>', 
                    unsafe_allow_html=True)
        
        # Show totals for numeric columns
        if numeric_cols:
            st.subheader("Data Totals by Column")
            totals_data = {}
            for col in numeric_cols:
                totals_data[col] = [
                    f"{df[col].sum():,.0f}",
                    f"{df[col].mean():.2f}",
                    f"{df[col].median():.2f}",
                    f"{df[col].min():.0f}",
                    f"{df[col].max():.0f}"
                ]
            
            totals_df = pd.DataFrame(
                totals_data,
                index=['Total', 'Mean', 'Median', 'Min', 'Max']
            )
            st.dataframe(totals_df, width='stretch')
        
        # st.subheader("Statistical Summary")
        # st.dataframe(df.describe(), width='stretch')


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
                        
                        # ADD DOWNLOAD BUTTON
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            label="Download Chart (PNG)",
                            data=buf.getvalue(),
                            file_name=f"comparison_{group_col}_{primary_metric}.png",
                            mime="image/png"
                        )
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
                            
                            # Calculate actual totals for this group
                            total_attendees = group_df[primary_metric].sum()
                            total_capacity = group_df[secondary_metric].sum()
                            avg_attendees = group_df[primary_metric].mean()
                            avg_capacity = group_df[secondary_metric].mean()
                            
                            st.markdown(f"## 🏢 {group_value} - Comprehensive Analysis")
                            
                            # Display totals in metric cards
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Attendees", f"{total_attendees:,.0f}")
                            with col2:
                                st.metric("Total Capacity", f"{total_capacity:,.0f}")
                            with col3:
                                st.metric("Avg Attendees", f"{avg_attendees:.1f}")
                            with col4:
                                st.metric("Avg Capacity", f"{avg_capacity:.1f}")
                            
                            st.caption(f"Analyzing {len(group_df)} data points")
                            
                            # Check if group has data
                            if len(group_df) == 0:
                                st.warning(f"No data available for {group_value}")
                                continue
                            
                            # TREND ANALYSIS
                            st.markdown("### Attendance vs Capacity Trends Over Time")
                            
                            try:
                                fig = create_utilization_trend(
                                    group_df, date_column, primary_metric, secondary_metric,
                                    title_prefix=f"{group_value} - "
                                )
                                st.pyplot(fig)
                                
                                # ADD DOWNLOAD BUTTON
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                buf.seek(0)
                                st.download_button(
                                    label="Download Trend Chart (PNG)",
                                    data=buf.getvalue(),
                                    file_name=f"trend_{group_value}_{primary_metric}_vs_{secondary_metric}.png",
                                    mime="image/png",
                                    key=f"download_trend_{idx}"
                                )
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

                            # NEW: CAPACITY UTILIZATION PERCENTAGE CHART
                            st.markdown("### 📊 Capacity Utilization Percentage Over Time")
                            st.caption("Shows what percentage of capacity is being used at each time period")

                            try:
                                fig, utilization_data = create_capacity_utilization_chart(
                                    group_df, date_column, primary_metric, secondary_metric,
                                    title_prefix=f"{group_value} - "
                                )
                                st.pyplot(fig)
                                
                                # ADD DOWNLOAD BUTTON
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                buf.seek(0)
                                st.download_button(
                                    label="Download Utilization % Chart (PNG)",
                                    data=buf.getvalue(),
                                    file_name=f"utilization_percentage_{group_value}.png",
                                    mime="image/png",
                                    key=f"download_util_pct_{idx}"
                                )
                                plt.close()
                                
                                # Insights and suggestions
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("#### 🔍 Utilization Insights")
                                    util_pct_insights, util_pct_suggestions = generate_utilization_percentage_insights(utilization_data)
                                    for insight in util_pct_insights:
                                        st.markdown(f'<div class="insight-box">{insight}</div>', 
                                                unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("#### 💡 Recommendations")
                                    for suggestion in util_pct_suggestions:
                                        st.markdown(f'<div class="insight-box">{suggestion}</div>', 
                                                unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Error creating utilization percentage chart: {str(e)}")
                            
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
                                        
                                        # ADD DOWNLOAD BUTTON
                                        buf = io.BytesIO()
                                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                        buf.seek(0)
                                        st.download_button(
                                            label="Download Distribution Chart (PNG)",
                                            data=buf.getvalue(),
                                            file_name=f"distribution_{group_value}_{primary_metric}.png",
                                            mime="image/png",
                                            key=f"download_dist_{idx}"
                                        )
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
                            
                            # Summary statistics
                            st.markdown("### Summary Statistics")
                            
                            try:
                                # Create unique list of columns to analyze
                                columns_to_analyze = [primary_metric]
                                if secondary_metric and secondary_metric != primary_metric:
                                    columns_to_analyze.append(secondary_metric)
                                
                                # Get statistics with actual totals
                                summary_df = group_df[columns_to_analyze].describe()
                                
                                # Add total row
                                totals = group_df[columns_to_analyze].sum()
                                summary_df.loc['total'] = totals
                                
                                # Reorder to put total at top
                                summary_df = summary_df.reindex(['total', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
                                
                                # Display with custom formatting
                                st.dataframe(summary_df.style.format("{:.2f}"), width='stretch')
                                
                                # Additional context
                                st.caption(f"Total {primary_metric}: {totals[primary_metric]:,.0f} | Total {secondary_metric}: {totals[secondary_metric]:,.0f}")
                                
                            except Exception as e:
                                st.error(f"Error displaying summary statistics: {str(e)}")


    # TAB 3: Chat Assistant
    with tab3:
        st.subheader("Assistant")
        st.caption("Ask anything about your data!")
        
        # Initialize cache
        if 'response_cache' not in st.session_state:
            st.session_state.response_cache = {}
        
        # Initialize agent once
        if st.session_state.agent is None:
            with st.spinner("Initializing AI assistant..."):
                try:
                    st.session_state.agent = create_data_analysis_agent(
                        df, 
                        st.session_state.uploaded_file_name,
                        MODEL_NAME
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display all previous chat messages
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        
        
        # Chat input - ONLY ONE, at the very end
        user_input = st.chat_input(CHAT_INPUT_PLACEHOLDER)
        
        # Process input AFTER it's entered
        if user_input:
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            # Get response
            cache_key = user_input.lower().strip()
            
            if cache_key in st.session_state.response_cache:
                answer = st.session_state.response_cache[cache_key]
            else:
                # Build messages
                messages = []
                for msg in st.session_state.chat_messages:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
                
                # Get AI response (spinner outside chat message)
                with st.spinner("🤖 Analyzing..."):
                    response = st.session_state.agent.invoke({"messages": messages})
                    answer = response["messages"][-1].content
                
                st.session_state.response_cache[cache_key] = answer
            
            # Save assistant response
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer
            })

            # # Controls at bottom (BEFORE chat input)
            # col1, col2 = st.columns([1, 5])
            # with col1:
            #     if st.button("🗑️ Clear", use_container_width=True):
            #         st.session_state.chat_messages = []
            #         st.session_state.response_cache = {}
            #         st.rerun()
            
            # Rerun to display new messages
            st.rerun()


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

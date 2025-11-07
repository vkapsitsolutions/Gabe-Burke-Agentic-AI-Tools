"""Main Streamlit application with modular structure."""

import streamlit as st
import pandas as pd 
import io
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, AIMessage

from config import (
    APP_TITLE, APP_SUBTITLE, 
    CHAT_INPUT_PLACEHOLDER, MODEL_NAME
)
from utils.styling import apply_custom_css
from utils.data_loader import load_data, get_grouped_data, filter_by_group
from utils.visualizations import (
    create_distribution_plot, create_boxplot, 
    create_bar_chart, create_utilization_trend,
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

apply_custom_css()

# Initialize session state
for key in ['df', 'chat_messages', 'agent', 'uploaded_file_name', 'group_column', 'dataset_context']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'chat_messages' else []

# Dynamic labels based on dataset
def get_labels():
    """Get contextual labels based on dataset type."""
    if st.session_state.dataset_context:
        return st.session_state.dataset_context
    return {
        'entity_label': 'Entity',
        'primary_metric_label': 'Primary Metric',
        'target_metric_label': 'Target Metric'
    }

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
            st.error(error)
        else:
            st.session_state.df = df
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.agent = None
            
            labels = get_labels()
            entity_label = labels.get('entity_label', 'Entity')
            
            st.success("Data loaded successfully")
            
            st.divider()
            st.subheader("Group Analysis")

            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                group_col = st.selectbox(
                    f"Select {entity_label} column for grouping",
                    ['None'] + categorical_cols,
                    help=f"Analyze data by different {entity_label.lower()}s"
                )
                
                if group_col != 'None':
                    st.session_state.group_column = group_col
                    groups = get_grouped_data(df, group_col)
                    st.info(f"Found {len(groups)} groups: {', '.join(map(str, groups[:5]))}" + 
                           (f"... +{len(groups)-5} more" if len(groups) > 5 else ""))
                else:
                    st.session_state.group_column = None

# Main content
if st.session_state.df is not None:
    df = st.session_state.df
    group_col = st.session_state.group_column
    labels = get_labels()
    
    primary_label = labels.get('primary_metric_label', 'Primary Metric')
    target_label = labels.get('target_metric_label', 'Target Metric')
    entity_label = labels.get('entity_label', 'Entity')
    
    tab1, tab2, tab3 = st.tabs([
        "Data Overview", 
        f"Analysis{' (Grouped)' if group_col else ''}", 
        "AI Assistant"
    ])
    
    # TAB 1: Data Overview
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), width='stretch')
        
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
        
        if numeric_cols:
            st.subheader("Data Summary by Column")
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
    
    # TAB 2: Analysis
    with tab2:
        if group_col:
            st.markdown(f'<div class="group-header">Analyzing by {group_col}</div>', 
                    unsafe_allow_html=True)
            
            groups = get_grouped_data(df, group_col)
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            date_column = date_cols[0] if date_cols else None
            
            if numeric_cols and date_column:
                col1, col2 = st.columns(2)
                with col1:
                    primary_metric = st.selectbox(f"Select {primary_label}", 
                                                numeric_cols, key='primary')
                with col2:
                    secondary_options = [col for col in numeric_cols if col != primary_metric]
                    if secondary_options:
                        secondary_metric = st.selectbox(f"Select {target_label}", 
                                                    secondary_options, key='secondary')
                    else:
                        secondary_metric = None
                        st.warning("Need at least 2 numeric columns")
                
                if secondary_metric:
                    st.divider()
                    
                    st.markdown(f"### Overall Comparison: {primary_metric} by {group_col}")
                    
                    comparison = df.groupby(group_col)[primary_metric].sum().sort_values(ascending=False)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = create_bar_chart(comparison, group_col, primary_metric, title_prefix="Total ")
                        st.pyplot(fig)
                        
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            label="Download Chart",
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
                    
                    st.markdown(f"### Detailed Analysis by {entity_label}")
                    
                    group_tabs = st.tabs([str(g) for g in groups])
                    
                    for idx, group_value in enumerate(groups):
                        with group_tabs[idx]:
                            group_df = filter_by_group(df, group_col, group_value)
                            
                            total_primary = group_df[primary_metric].sum()
                            total_secondary = group_df[secondary_metric].sum()
                            avg_primary = group_df[primary_metric].mean()
                            avg_secondary = group_df[secondary_metric].mean()
                            
                            st.markdown(f"## {group_value} - Analysis")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(f"Total {primary_metric}", f"{total_primary:,.0f}")
                            with col2:
                                st.metric(f"Total {secondary_metric}", f"{total_secondary:,.0f}")
                            with col3:
                                st.metric(f"Avg {primary_metric}", f"{avg_primary:.1f}")
                            with col4:
                                st.metric(f"Avg {secondary_metric}", f"{avg_secondary:.1f}")
                            
                            st.caption(f"Analyzing {len(group_df)} data points")
                            
                            if len(group_df) == 0:
                                st.warning(f"No data for {group_value}")
                                continue
                            
                            # Trend Analysis
                            st.markdown(f"### {primary_metric} vs {secondary_metric} Over Time")
                            
                            try:
                                fig = create_utilization_trend(
                                    group_df, date_column, primary_metric, secondary_metric,
                                    title_prefix=f"{group_value} - "
                                )
                                st.pyplot(fig)
                                
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                buf.seek(0)
                                st.download_button(
                                    label="Download Trend Chart",
                                    data=buf.getvalue(),
                                    file_name=f"trend_{group_value}.png",
                                    mime="image/png",
                                    key=f"download_trend_{idx}"
                                )
                                plt.close()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Trend Insights")
                                try:
                                    trend_insights = generate_trend_insights(
                                        group_df, date_column, [primary_metric, secondary_metric]
                                    )
                                    for insight in trend_insights:
                                        st.markdown(f'<div class="insight-box">{insight}</div>', 
                                                unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                            
                            with col2:
                                st.markdown("#### Recommendations")
                                try:
                                    util_insights, util_suggestions = generate_utilization_insights(
                                        group_df, primary_metric, secondary_metric
                                    )
                                    for suggestion in util_suggestions:
                                        st.markdown(f'<div class="insight-box">{suggestion}</div>', 
                                                unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                            
                            st.divider()
                            
                            # Utilization Percentage
                            st.markdown("### Utilization Percentage Over Time")
                            
                            try:
                                fig, utilization_data = create_capacity_utilization_chart(
                                    group_df, date_column, primary_metric, secondary_metric,
                                    title_prefix=f"{group_value} - "
                                )
                                st.pyplot(fig)
                                
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                buf.seek(0)
                                st.download_button(
                                    label="Download Utilization Chart",
                                    data=buf.getvalue(),
                                    file_name=f"utilization_{group_value}.png",
                                    mime="image/png",
                                    key=f"download_util_{idx}"
                                )
                                plt.close()
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("#### Utilization Insights")
                                    util_pct_insights, _ = generate_utilization_percentage_insights(utilization_data)
                                    for insight in util_pct_insights:
                                        st.markdown(f'<div class="insight-box">{insight}</div>', 
                                                unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("#### Actions")
                                    _, util_pct_suggestions = generate_utilization_percentage_insights(utilization_data)
                                    for suggestion in util_pct_suggestions:
                                        st.markdown(f'<div class="insight-box">{suggestion}</div>', 
                                                unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
        else:
            st.info(f"Select a {entity_label} column in the sidebar for grouped analysis")
    
    # TAB 3: AI Assistant
    with tab3:
        st.subheader("AI Assistant")
        st.caption("Ask questions about your data")
        
        if st.session_state.agent is None:
            with st.spinner("Initializing AI..."):
                try:
                    st.session_state.agent = create_data_analysis_agent(
                        df, st.session_state.uploaded_file_name, MODEL_NAME
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        chat_container = st.container()
        
        with chat_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        
        prompt = st.chat_input(CHAT_INPUT_PLACEHOLDER)
        
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Thinking..."):
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
                    st.error(f"Error: {str(e)}")
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("Clear"):
                st.session_state.chat_messages = []
                st.rerun()
        with col2:
            if st.session_state.chat_messages:
                chat_text = "\n\n".join([
                    f"{m['role'].upper()}: {m['content']}" 
                    for m in st.session_state.chat_messages
                ])
                st.download_button("Export", chat_text, "chat.txt")

else:
    st.info("Upload a file to begin analysis")


"""CSS styling for the Streamlit app."""

import streamlit as st

def apply_custom_css():
    """Apply custom CSS for better UI."""
    st.markdown("""
    <style>
        /* Main styling */
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
        
        /* Insight boxes */
        .insight-box {
            background-color: #e8f4f8;
            border-left: 5px solid #1f77b4;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
        
        /* Metric cards */
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Chat container - FIXED AT BOTTOM */
        .stChatFloatingInputContainer {
            bottom: 20px;
            background-color: white;
            padding: 10px;
            border-top: 2px solid #e0e0e0;
        }
        
        /* Chat messages container with scroll */
        .stChatMessageContainer {
            padding-bottom: 100px;
        }
        
        /* Group analysis headers */
        .group-header {
            background: linear-gradient(90deg, #1f77b4 0%, #4a9eff 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            font-size: 1.3rem;
            font-weight: bold;
        }
        
        /* Divider */
        hr {
            margin: 2rem 0;
            border: none;
            border-top: 2px solid #e0e0e0;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            background-color: #f0f2f6;
            border-radius: 5px;
        }
        
        /* Scrollable chat history */
        .chat-history {
            max-height: 500px;
            overflow-y: auto;
            padding-right: 10px;
            margin-bottom: 20px;
        }
        
        /* Better scrollbar */
        .chat-history::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-history::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        .chat-history::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }
        
        .chat-history::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
    """, unsafe_allow_html=True)

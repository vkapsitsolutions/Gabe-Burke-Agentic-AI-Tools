"""Configuration settings for the application."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

# App Configuration
APP_TITLE = "AI Data Analysis Agent"
APP_SUBTITLE = "Upload data, get AI-powered insights with grouped analysis"

# Visualization Settings
PLOT_STYLE = "whitegrid"
FIGURE_DPI = 100
COLOR_PALETTE = "Set2"

# Chat Settings
MAX_CHAT_HISTORY = 50
CHAT_INPUT_PLACEHOLDER = "Ask anything about your data..."

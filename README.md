# Agentic AI Data Insights Dashboard

## Overview

The **Agentic AI Data Insights Dashboard** is a modular Streamlit-based analytics platform designed for intelligent, automated data exploration.
It leverages **agentic AI capabilities** through LangChain agents to interpret data, generate insights, and respond to natural language queries — enabling advanced, interactive analytics without manual scripting.

This system integrates **Streamlit**, **Pandas**, **Matplotlib**, and **LangChain**, packaged within a scalable and maintainable architecture that supports both data visualization and AI-assisted reasoning.

---

## Key Features

* Upload and process CSV or Excel datasets.
* Automated detection of numerical, categorical, and temporal columns.
* Group-level comparison and utilization analysis.
* AI-driven insights and actionable recommendations.
* Visual analytics, including:

  * Bar charts
  * Trend and utilization analysis
  * Statistical distributions
  * Boxplots and correlation heatmaps
* Conversational data assistant powered by LangChain and LLMs.
* Downloadable charts and AI-generated reports.
* Modular, agentic, and production-ready architecture.

---

## Project Structure

```
GABE-BURKE-AGENTIC-AI-TOOLS/
│
├── .env                        # Environment variables (API keys, model settings)
├── .gitignore                  # Git ignore rules
├── .python-version             # Python version configuration
│
├── app/
│   └── app.py                  # Main Streamlit application entry point
│
├── agents/                     # LangChain and AI agent modules
│   └── data_agent.py
│
├── utils/                      # Supporting utility modules
│   ├── styling.py              # Custom CSS and layout management
│   ├── data_loader.py          # Data upload and preprocessing utilities
│   ├── visualizations.py       # Chart and plotting functions
│   └── insights.py             # Insight generation and analysis logic
│
├── data/                       # Sample or user-uploaded datasets
├── notebook/                   # Jupyter notebooks for experimentation
├── service/                    # Backend service components (if applicable)
├── test-code/                  # Testing and development scripts
│
├── config.py                   # Configuration constants and model definitions
├── start_program.sh            # Shell script to start the application
├── sample.json                 # Sample data or configuration example
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata and dependency management
├── Dockerfile                  # Docker configuration for container deployment
├── feedback.txt                # Internal notes or client feedback
├── uv.lock                     # Lock file for dependency versions
└── README.md                   # Project documentation
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/agentic-ai-data-insights.git
cd agentic-ai-data-insights
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## `requirements.txt`

```txt
# Core Framework
streamlit>=1.39.0

# Data Processing
pandas>=2.0.0
numpy>=1.26.0
openpyxl>=3.1.0
pyarrow>=17.0.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Statistical Analysis
scipy>=1.11.0

# AI/ML - LangChain v1.0
langchain
langchain-openai
langchain-core

# OpenAI
openai>=1.0.0

# Environment Management
python-dotenv>=1.0.0
```

---

## Running the Application

After installation, start the Streamlit application:

```bash
streamlit run app/app.py
```

or, if using the startup script:

```bash
bash start_program.sh
```

Then open your browser and navigate to:

```
http://localhost:8501
```

---

## Application Workflow

### 1. Data Upload

Upload a CSV or Excel file from the sidebar.
The application automatically detects column types (numeric, categorical, and date).

### 2. Data Overview

Displays:

* Dataset preview
* Summary metrics (rows, columns, missing values, memory usage)
* Aggregations and descriptive statistics for numeric fields

### 3. Visualizations

Automatically generates:

* Group-based comparisons
* Trend and utilization analysis
* Distribution and correlation plots
  Each visualization includes optional AI-generated insights and downloadable charts.

### 4. Agentic Chat Assistant

The built-in LangChain-powered agent enables natural language interaction with the dataset.
Users can query, summarize, or request insights conversationally.
Chat sessions persist across interactions and can be exported or cleared.

---

## Configuration and Customization

* **App configuration:** Modify titles, text, and constants in `config.py`.
* **Styling and UI:** Customize layout and colors in `utils/styling.py`.
* **Insight logic:** Extend data reasoning rules in `utils/insights.py`.
* **Visualization templates:** Adjust Matplotlib configurations in `utils/visualizations.py`.
* **Agent model:** Configure the model name and provider in `MODEL_NAME` within `config.py` or `.env`.

---

## Deployment Options

### Docker

A preconfigured `Dockerfile` is included. Build and run the image with:

```bash
docker build -t agentic-ai-dashboard .
docker run -p 8501:8501 agentic-ai-dashboard
```

### Manual / Local

Run directly via Streamlit as outlined above.

---

#### Thanks :)
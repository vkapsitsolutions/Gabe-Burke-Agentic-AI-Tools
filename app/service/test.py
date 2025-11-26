import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.schema import HumanMessage

load_dotenv()

# -------------------------------
# Tool Definition
# -------------------------------
@tool
def analyze_building_data(input_file: str) -> str:
    """Analyze building attendance Excel data and generate monthly summaries and charts."""
    data = pd.read_excel(input_file)
    columns = data.columns.tolist()
    first_col = columns[0]
    date_col = next((col for col in columns if 'date' in col.lower()), None)
    numeric_cols = [col for col in columns if col.lower() in ['attendees', 'capacity']]

    if not date_col:
        return "No date column found."
    if len(numeric_cols) < 2:
        return "The file must contain 'Attendees' and 'Capacity' columns."

    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data['Month'] = data[date_col].dt.strftime('%b-%Y')

    summary = (
        data.groupby([first_col, 'Month'])[numeric_cols]
        .sum()
        .reset_index()
        .sort_values(['Month', first_col])
    )

    output_dir = "app/output_buildings"
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "building_monthly_summary.xlsx")
    summary.to_excel(summary_file, index=False)

    for building, group in summary.groupby(first_col):
        plt.figure(figsize=(10, 6))
        x = np.arange(len(group['Month']))
        bar_width = 0.35
        plt.bar(x - bar_width/2, group[numeric_cols[1]], width=bar_width, label=numeric_cols[1], alpha=0.7)
        plt.bar(x + bar_width/2, group[numeric_cols[0]], width=bar_width, label=numeric_cols[0], alpha=0.9)
        plt.title(f"{building} - Monthly Attendance vs Capacity", fontsize=14)
        plt.xlabel("Month")
        plt.ylabel("Count")
        plt.xticks(x, group['Month'], rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        chart_path = os.path.join(output_dir, f"{building.replace(' ', '_')}_bar_chart.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()

    return f"Summary and charts created successfully in {output_dir}"

# -------------------------------
# Agent Setup
# -------------------------------
model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.1,
    max_tokens=1024
)

agent = create_agent(
    model=model,
    tools=[analyze_building_data],
    system_prompt="You are a helpful assistant that can analyze Excel attendance data."
)

# -------------------------------
# Run Agent
# -------------------------------
user_message = HumanMessage(content="Analyze the building data from app/data/Office Attedance Data.xlsx")
result = agent.invoke(user_message)

print(result)

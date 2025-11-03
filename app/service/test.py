# import pandas as pd
# import os


# input_file = 'app/data/Office Attedance Data.xlsx'


# data = pd.read_excel(input_file)


# columns = data.columns.tolist()
# print(f"Columns in the DataFrame: {columns}")


# first_col = columns[0]
# print(f"Using first column: {first_col}")


# unique_values = data[first_col].dropna().unique()
# print(f"Unique values in '{first_col}': {unique_values}")


# output_dir = "app/output_buildings"
# os.makedirs(output_dir, exist_ok=True)


# for val in unique_values:
#     subset = data[data[first_col] == val]
#     clean_name = str(val).replace(" ", "_").replace("/", "_")
#     output_path = os.path.join(output_dir, f"{clean_name}.xlsx")


    
#     subset.to_excel(output_path, index=False)
#     print(f" File created for '{val}': {output_path}")

# print("\n All files created successfully!")



# import pandas as pd
# import os

# # Input Excel file path
# input_file = 'app/data/Office Attedance Data.xlsx'

# # Read the Excel file
# data = pd.read_excel(input_file)

# # Get all column names
# columns = data.columns.tolist()
# print(f"Columns in the DataFrame: {columns}")

# # Use the first column dynamically
# first_col = columns[0]
# print(f"Using first column: {first_col}")

# # Check if 'Attendees' and 'Capacity' columns exist
# numeric_cols = [col for col in columns if col.lower() in ['attendees', 'capacity']]
# if len(numeric_cols) < 2:
#     raise ValueError("The file must contain 'Attendees' and 'Capacity' columns.")

# # Group by the first column and calculate totals
# summary = (
#     data.groupby(first_col)[numeric_cols]
#     .sum()
#     .reset_index()
# )

# # Create output folder
# output_dir = "app/output_buildings"
# os.makedirs(output_dir, exist_ok=True)

# # Save summary to one Excel file
# output_file = os.path.join(output_dir, "building_totals.xlsx")
# summary.to_excel(output_file, index=False)

# print(f"Summary Excel file created: {output_file}")
# print(summary)



# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# input_file = 'app/data/Office Attedance Data.xlsx'

# data = pd.read_excel(input_file)

# columns = data.columns.tolist()
# print(f"Columns: {columns}")

# first_col = columns[0]
# print(f"Using first column: {first_col}")

# numeric_cols = [col for col in columns if col.lower() in ['attendees', 'capacity']]
# if len(numeric_cols) < 2:
#     raise ValueError("The file must contain 'Attendees' and 'Capacity' columns.")



# summary = (
#     data.groupby(first_col)[numeric_cols]
#     .sum()
#     .reset_index()
# )


# output_dir = "app/output_buildings"
# os.makedirs(output_dir, exist_ok=True)

# summary_file = os.path.join(output_dir, "building_totals.xlsx")
# summary.to_excel(summary_file, index=False)
# print(f"Summary Excel created: {summary_file}")


# plt.figure(figsize=(10, 6))
# x = summary[first_col]
# bar_width = 0.35

# plt.bar(x, summary[numeric_cols[1]], width=bar_width, label=numeric_cols[1], alpha=0.6)
# plt.bar(x, summary[numeric_cols[0]], width=bar_width, label=numeric_cols[0], alpha=0.9)

# plt.xlabel(first_col)
# plt.ylabel("Count")
# plt.title("Building-wise Capacity vs Attendees")
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()

# chart_path = os.path.join(output_dir, "building_chart.png")
# plt.savefig(chart_path, dpi=300)
# plt.close()

# print(f"Chart saved: {chart_path}")
# print(summary)







# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # -------------------------------
# # 🔹 1. Read the Excel file
# # -------------------------------
# input_file = 'app/data/Office Attedance Data.xlsx'
# data = pd.read_excel(input_file)

# # -------------------------------
# # 🔹 2. Detect columns dynamically
# # -------------------------------
# columns = data.columns.tolist()
# print(f"Columns detected: {columns}")

# first_col = columns[0]  # e.g., 'Building Name'
# print(f"Using first column: {first_col}")

# # Find date column dynamically (any column containing 'date')
# date_col = next((col for col in columns if 'date' in col.lower()), None)
# if not date_col:
#     raise ValueError(" No 'Date' column found — please ensure there’s a date column.")

# # Detect numeric columns (Attendees and Capacity)
# numeric_cols = [col for col in columns if col.lower() in ['attendees', 'capacity']]
# if len(numeric_cols) < 2:
#     raise ValueError(" The file must contain 'Attendees' and 'Capacity' columns.")

# # -------------------------------
# # 🔹 3. Prepare data
# # -------------------------------
# # Convert date column to datetime
# data[date_col] = pd.to_datetime(data[date_col], errors='coerce')

# # Extract Month-Year (e.g. "Jan-2024")
# data['Month'] = data[date_col].dt.strftime('%b-%Y')

# # -------------------------------
# # 🔹 4. Group by Building and Month
# # -------------------------------
# summary = (
#     data.groupby([first_col, 'Month'])[numeric_cols]
#     .sum()
#     .reset_index()
#     .sort_values(['Month', first_col])
# )

# # -------------------------------
# # 🔹 5. Create output folder
# # -------------------------------
# output_dir = "app/output_buildings"
# os.makedirs(output_dir, exist_ok=True)

# # Save summary table
# output_file = os.path.join(output_dir, "building_monthly_summary.xlsx")
# summary.to_excel(output_file, index=False)
# print(f"Summary Excel created: {output_file}")

# # -------------------------------
# # 🔹 6. Generate Month-wise Charts per Building
# # -------------------------------
# for building, group in summary.groupby(first_col):
#     plt.figure(figsize=(10, 6))
    
#     x = group['Month']
#     plt.plot(x, group[numeric_cols[1]], marker='o', label=numeric_cols[1], linewidth=2, alpha=0.7)
#     plt.plot(x, group[numeric_cols[0]], marker='s', label=numeric_cols[0], linewidth=2, alpha=0.9)
    
#     plt.title(f"{building} — Monthly Attendance vs Capacity", fontsize=14, weight='bold')
#     plt.xlabel("Month")
#     plt.ylabel("Count")
#     plt.xticks(rotation=45)
#     plt.grid(alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
    
#     chart_path = os.path.join(output_dir, f"{building.replace(' ', '_')}_chart.png")
#     plt.savefig(chart_path, dpi=300)
#     plt.close()
    
#     print(f" Chart saved for {building}: {chart_path}")

# print("\n All monthly summaries and charts created successfully!")























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

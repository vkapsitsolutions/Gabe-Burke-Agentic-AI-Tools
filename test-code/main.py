import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_building_data(input_file):
    # Read Excel file
    data = pd.read_excel(input_file)

    # Get all column names
    columns = data.columns.tolist()
    print(f"Detected columns: {columns}")

    # Identify first column (e.g., Building Name)
    first_col = columns[0]
    print(f"Using first column: {first_col}")

    # Detect date column dynamically
    date_col = next((col for col in columns if 'date' in col.lower()), None)
    if not date_col:
        raise ValueError("No date column found in the file.")

    # Detect numeric columns (Attendees and Capacity)
    numeric_cols = [col for col in columns if col.lower() in ['attendees', 'capacity']]
    if len(numeric_cols) < 2:
        raise ValueError("The file must contain 'Attendees' and 'Capacity' columns.")

    # Convert date column to datetime
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')

    # Extract Month-Year from the date column
    data['Month'] = data[date_col].dt.strftime('%b-%Y')

    # Group by Building and Month, then sum totals
    summary = (
        data.groupby([first_col, 'Month'])[numeric_cols]
        .sum()
        .reset_index()
        .sort_values(['Month', first_col])
    )

    # Create output directory
    output_dir = "app/output_buildings"
    os.makedirs(output_dir, exist_ok=True)

    # Save summary table to Excel
    summary_file = os.path.join(output_dir, "building_monthly_summary.xlsx")
    summary.to_excel(summary_file, index=False)
    print(f"Summary Excel saved: {summary_file}")

    # Generate bar chart for each building
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

        print(f"Bar chart saved for {building}: {chart_path}")

    print("All summaries and bar charts created successfully.")


if __name__ == "__main__":
    input_file = "app/data/Office Attedance Data.xlsx"
    analyze_building_data(input_file)

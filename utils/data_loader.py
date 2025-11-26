# """AI-powered data loading with intelligent column detection using OpenAI."""

import pandas as pd
import streamlit as st
from openai import OpenAI
import os
import json
from typing import Optional, Tuple, Dict
from config import OPENAI_API_KEY

# Initialize OpenAI client (using latest SDK v1.0+)
client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_data
def ai_analyze_file_structure(preview_data: list, max_rows: int = 10) -> Dict:
    """
    Use OpenAI to analyze file structure and identify:
    1. Which row contains the actual headers
    2. Which columns map to required fields
    
    Args:
        preview_data: List of lists containing first N rows of the file
        max_rows: Maximum rows to analyze
    
    Returns:
        Dictionary with header row and column mappings
    """
    try:
        # Prepare data for analysis
        rows_str = json.dumps(preview_data[:max_rows], indent=2, default=str)
        
        system_prompt = """You are an expert data analyst specializing in Excel/CSV file structure analysis.
Your task is to intelligently detect headers and map columns regardless of file format variations.

Key capabilities:
- Detect title rows vs actual header rows
- Handle merged cells (Unnamed columns)
- Understand descriptive column names
- Infer column purpose from both names AND data
- Work with any language or format"""

        user_prompt = f"""Analyze this spreadsheet structure and provide precise mapping.

**Raw Data (first {max_rows} rows, 0-indexed):**
{rows_str}

**Your Task:**
1. Identify which row (0-indexed) contains the ACTUAL column headers (not title/blank rows)
2. For each header, determine if it represents:
   - Building/Location/Facility name
   - Date/Time information
   - Number of attendees/people/visitors
   - Capacity/seats/maximum occupancy

**Important Rules:**
- Look at BOTH column names AND sample data values
- Merged cells often show as "Unnamed: N"
- Long descriptions may contain the actual column name at the end
- Infer from data patterns (dates look like dates, numbers for counts)
- Be flexible - column names vary widely

**Return Format:**
Provide a JSON object with this EXACT structure:
{{
    "header_row_index": 0,
    "column_mappings": {{
        "building_name": "exact column name or null",
        "date": "exact column name or null",
        "attendees": "exact column name or null",
        "capacity": "exact column name or null"
    }},
    "confidence_score": 0.95,
    "analysis_notes": "Brief explanation of your decisions"
}}

**Critical:** Return ONLY the JSON object, no markdown, no code blocks, no extra text."""

        # Call OpenAI with structured output approach
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Latest model with better JSON handling
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent results
            response_format={"type": "json_object"}  # Force JSON response
        )
        
        result = response.choices[0].message.content.strip()
        analysis = json.loads(result)
        
        return analysis
        
    except json.JSONDecodeError as e:
        st.error(f"AI returned invalid JSON: {str(e)}")
        return None
    except Exception as e:
        st.error(f"AI analysis failed: {str(e)}")
        return None

def load_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load and process uploaded file using AI-powered intelligence.
    
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        # Step 1: Read preview of file (no header assumption)
        with st.spinner("Agent analyzing file structure..."):
            if uploaded_file.name.endswith('.csv'):
                preview_df = pd.read_csv(uploaded_file, nrows=10, header=None)
                uploaded_file.seek(0)  # Reset file pointer
            else:
                preview_df = pd.read_excel(uploaded_file, nrows=10, header=None)
                uploaded_file.seek(0)  # Reset file pointer
            
            # Convert to list of lists for AI analysis
            preview_data = preview_df.values.tolist()
            
            # Step 2: AI analyzes structure
            # st.info("🧠 AI detecting headers and columns...")
            analysis = ai_analyze_file_structure(preview_data)
            
            if not analysis:
                return None, "❌ AI analysis failed. Please check your OpenAI API key and try again."
            
            # Display AI's findings
            # confidence = analysis.get('confidence_score', 0)
            # st.success(f"✅ AI Analysis Complete (Confidence: {confidence:.0%})")
            
            # with st.expander("🔍 View AI Analysis Details"):
            #     st.json(analysis)
            
            # Step 3: Read file with detected header row
            header_row = analysis.get('header_row_index', 0)
            
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=header_row)
            else:
                df = pd.read_excel(uploaded_file, header=header_row)
            
            # Step 4: Apply AI-detected column mappings
            mappings = analysis.get('column_mappings', {})
            rename_dict = {}
            
            if mappings.get('building_name'):
                rename_dict[mappings['building_name']] = 'Building Name'
            if mappings.get('date'):
                rename_dict[mappings['date']] = 'Date'
            if mappings.get('attendees'):
                rename_dict[mappings['attendees']] = 'Attendees'
            if mappings.get('capacity'):
                rename_dict[mappings['capacity']] = 'Capacity'
            
            # Rename columns
            df = df.rename(columns=rename_dict)
            
            # Step 5: Data cleaning
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Remove duplicate header rows (where data matches column names)
            if len(df) > 0:
                mask = ~df.apply(
                    lambda row: any(
                        str(val).lower().strip() == str(col).lower().strip()
                        for val, col in zip(row, df.columns)
                        if pd.notna(val)
                    ),
                    axis=1
                )
                df = df[mask]
            
            # Reset index
            df = df.reset_index(drop=True)
            
            # Step 6: Validate required columns
            required = ['Building Name', 'Date', 'Attendees', 'Capacity']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                available_cols = ', '.join(df.columns)
                notes = analysis.get('analysis_notes', 'N/A')
                
                return None, (
                    f"❌ AI couldn't identify all required columns\n\n"
                    f"**Missing:** {', '.join(missing)}\n\n"
                    f"**Found:** {available_cols}\n\n"
                    f"**AI Notes:** {notes}\n\n"
                    f"💡 **Tip:** Ensure your file has clear labels for building, date, attendees, and capacity."
                )
            
            # Step 7: Convert data types
            # Date column
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Numeric columns
            df['Attendees'] = pd.to_numeric(df['Attendees'], errors='coerce')
            df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
            
            # Step 8: Remove rows with missing critical data
            before_clean = len(df)
            df = df.dropna(subset=['Building Name', 'Date', 'Attendees', 'Capacity'], how='any')
            after_clean = len(df)
            
            if after_clean == 0:
                return None, "❌ No valid data rows found after cleaning. Please check your file format."
            
            # Show cleaning summary
            if before_clean > after_clean:
                st.warning(f"⚠️ Removed {before_clean - after_clean} rows with missing data")
            
            # st.success(f"✅ Successfully loaded {len(df)} rows of clean data!")
            
            return df, None
            
    except Exception as e:
        return None, f"❌ Error loading file: {str(e)}\n\nPlease ensure your file is a valid CSV or Excel file."

def get_grouped_data(df, group_column):
    """Get unique groups from a column."""
    if group_column in df.columns:
        return sorted(df[group_column].dropna().unique())
    return []

def filter_by_group(df, group_column, group_value):
    """Filter dataframe by group value."""
    return df[df[group_column] == group_value].copy()

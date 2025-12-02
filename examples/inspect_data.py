
import pandas as pd
import sys

def inspect_data():
    try:
        df = pd.read_csv("input/applications_enriched_with_context_mboa.csv")
        print("Columns:", df.columns)
        
        # Check column mapping as per main.py
        if "schooljaar_afgeleid" in df.columns:
            year_col = "schooljaar_afgeleid"
        elif "Collegejaar" in df.columns:
            year_col = "Collegejaar"
        else:
            print("Could not find year column")
            return

        if "week_of_year" in df.columns:
            week_col = "week_of_year"
        elif "Weeknummer" in df.columns:
            week_col = "Weeknummer"
        else:
            print("Could not find week column")
            return

        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df[week_col] = pd.to_numeric(df[week_col], errors='coerce')

        print("\nMax week per year:")
        summary = df.groupby(year_col)[week_col].max().sort_index()
        print(summary)
        
        print("\nRow counts per year:")
        print(df[year_col].value_counts().sort_index())

    except Exception as e:
        print(f"Error reading data: {e}")

if __name__ == "__main__":
    inspect_data()

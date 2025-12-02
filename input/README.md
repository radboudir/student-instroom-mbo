# Input Data Directory

This directory should contain the required CSV data files for the MBO student enrollment prediction system.

## Required Files

### 1. `applications_enriched_with_context_mboa.csv`
Individual student application data with the following required columns:
- `schooljaar_afgeleid` - Academic year
- `opleidingcode` - Program code
- `leertrajectmbo` - Learning trajectory (e.g., BOL, BBL)
- `week_of_year` - Week number (1-52)
- `status_proper_case` - Application status (Enrolled, Received, Offered, Withdrawn, Rejected, Submitted)
- `bsnhash` - Student identifier (hashed BSN)
- `Opleidingsnaam` - Program name (optional but recommended)

Additional context columns for prediction features:
- `Opleiding`, `Type vooropleiding`, `Nationaliteit`, `EER`, `Geslacht`
- `Geverifieerd adres land`, `School eerste vooropleiding`, `Land code eerste vooropleiding`

### 2. `inschrijvingen_summary_mboa.csv`
Enrollment summary data with columns:
- `schooljaar_afgeleid` - Academic year
- `opleidingcode` - Program code  
- `leertrajectmbo` - Learning trajectory
- `aantal` - Number of enrolled students

**Note:** This file can use semicolon (`;`), comma (`,`), or tab (`\t`) as delimiter. The system will auto-detect.

## Data Privacy

⚠️ **Important**: These files contain sensitive student data and should **never** be committed to version control.

The `.gitignore` file is configured to exclude all CSV files and contents of this directory (except this README).

## Setup

1. Place your data files in this directory
2. Ensure file names match exactly as specified above
3. Verify column names match the requirements
4. Run the prediction system with: `uv run main.py -y 2024 -w 8 -wf`

## Data Format Notes

- Week numbers follow the academic year calendar (weeks 39-52, then 1-38)
- Dates and numeric fields will be automatically converted
- Missing values in categorical columns will be filled with "Onbekend" (Unknown)

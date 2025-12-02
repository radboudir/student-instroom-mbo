#!/usr/bin/env python
"""
Test SARIMA for FINAL ENROLLMENT prediction based on PROBABILITY SUMS
Goal: Use cumulative probability sums (from XGBoost) to predict final enrollment
"""

import sys
from pathlib import Path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

def load_and_prepare_data():
    """Load MBO data"""
    print("Loading data...")
    df = pd.read_csv("input/applications_enriched_with_context_mboa.csv")
    
    df = df.rename(columns={
        "schooljaar_afgeleid": "Collegejaar",
        "opleidingcode": "Opleidingscode",
        "leertrajectmbo": "Leertraject",
        "week_of_year": "Weeknummer",
        "status_proper_case": "Inschrijfstatus",
        "bsnhash": "Sleutel"
    })
    
    df["Collegejaar"] = pd.to_numeric(df["Collegejaar"], errors='coerce')
    df["Weeknummer"] = pd.to_numeric(df["Weeknummer"], errors='coerce')
    df["Opleidingscode"] = pd.to_numeric(df["Opleidingscode"], errors='coerce')
    df = df.dropna(subset=["Collegejaar", "Weeknummer", "Opleidingscode", "Leertraject"])
    
    # Map status to enrollment probability (0 or 1 for historical data)
    status_map = {
        "Enrolled": 1,
        "Received": 0,
        "Offered": 0,
        "Withdrawn": 0,
        "Rejected": 0,
        "Submitted": 0,
    }
    df["enrollment_prob"] = df["Inschrijfstatus"].map(status_map).fillna(0)
    
    print(f"Loaded {len(df)} rows")
    return df


def find_programme_with_multi_year_data(df, min_years=3, min_weeks_per_year=25):
    """Find programmes with data across multiple years"""
    print(f"\nFinding programmes with {min_years}+ years of data...")
    
    # Group by programme*leertraject and year
    grouped = df.groupby(["Opleidingscode", "Leertraject", "Collegejaar"]).agg({
        "Weeknummer": "nunique",
        "Sleutel": "count"
    }).reset_index()
    
    grouped.columns = ["Opleidingscode", "Leertraject", "Collegejaar", "n_weeks", "n_applications"]
    
    # Filter years with enough weeks
    grouped = grouped[grouped["n_weeks"] >= min_weeks_per_year]
    
    # Count years per programme
    year_counts = grouped.groupby(["Opleidingscode", "Leertraject"]).size().reset_index(name="n_years")
    
    # Filter programmes with enough years
    candidates = year_counts[year_counts["n_years"] >= min_years].sort_values("n_years", ascending=False)
    
    print(f"Found {len(candidates)} programmes with {min_years}+ years")
    
    if len(candidates) > 0:
        # Get details for top candidate
        top = candidates.iloc[0]
        print(f"\nSelected: Opleiding={top['Opleidingscode']}, Leertraject={top['Leertraject']}")
        
        # Show year details
        details = grouped[
            (grouped["Opleidingscode"] == top["Opleidingscode"]) &
            (grouped["Leertraject"] == top["Leertraject"])
        ]
        print("\nYear details:")
        print(details[["Collegejaar", "n_weeks", "n_applications"]])
        
        return top["Opleidingscode"], top["Leertraject"]
    
    return None, None


def create_multi_year_time_series(df, opleidingscode, leertraject):
    """
    Create time series of weekly cumulative probability sums across multiple years
    Returns: DataFrame with columns [year, week_idx, cumulative_prob]
    """
    print(f"\nCreating multi-year time series...")
    
    # Filter to programme
    prog_data = df[
        (df["Opleidingscode"] == opleidingscode) &
        (df["Leertraject"] == leertraject)
    ].copy()
    
    years = sorted(prog_data["Collegejaar"].unique())
    print(f"Years available: {years}")
    
    all_series = []
    
    for year in years:
        year_data = prog_data[prog_data["Collegejaar"] == year].copy()
        
        # Group by week and sum probabilities
        weekly_probs = year_data.groupby("Weeknummer")["enrollment_prob"].sum().reset_index()
        weekly_probs.columns = ["Weeknummer", "prob_sum"]
        
        # Create full academic year (39-52, 1-38)
        all_weeks = list(range(39, 53)) + list(range(1, 39))
        ts_df = pd.DataFrame({"Weeknummer": all_weeks})
        ts_df = ts_df.merge(weekly_probs, on="Weeknummer", how="left")
        ts_df["prob_sum"] = ts_df["prob_sum"].fillna(0)
        
        # Make cumulative
        ts_df["cumulative_prob"] = ts_df["prob_sum"].cumsum()
        
        # Add week index (0-51)
        ts_df["week_idx"] = range(len(ts_df))
        ts_df["year"] = year
        
        # Only keep weeks with data
        max_week = weekly_probs["Weeknummer"].max()
        if max_week >= 39:
            ts_df = ts_df[ts_df["Weeknummer"] <= max_week]
        else:
            ts_df = ts_df[(ts_df["Weeknummer"] >= 39) | (ts_df["Weeknummer"] <= max_week)]
        
        final_enrollment = ts_df["cumulative_prob"].iloc[-1]
        print(f"  Year {year:.0f}: {len(ts_df)} weeks, final enrollment = {final_enrollment:.1f}")
        
        all_series.append(ts_df)
    
    return all_series, years


def test_sarima_with_probabilities(all_series, years, test_year_idx, predict_from_week_idx):
    """
    Test SARIMA using historical probability sums to predict final enrollment
    
    Args:
        all_series: List of DataFrames, one per year
        years: List of years
        test_year_idx: Index of year to use as test (e.g., -1 for last year)
        predict_from_week_idx: Week index to predict from (0-based)
    """
    test_year = years[test_year_idx]
    train_years = years[:test_year_idx]
    
    print(f"\n{'='*70}")
    print(f"SARIMA Test: Predict final enrollment for year {test_year:.0f}")
    print(f"Training on years: {train_years}")
    print(f"Predicting from week index: {predict_from_week_idx}")
    print(f"{'='*70}")
    
    # Get historical final enrollments
    historical_finals = [series["cumulative_prob"].iloc[-1] for series in all_series[:test_year_idx]]
    print(f"\nHistorical final enrollments: {[f'{x:.1f}' for x in historical_finals]}")
    
    # Get test year data
    test_series = all_series[test_year_idx]
    actual_final = test_series["cumulative_prob"].iloc[-1]
    current_cumulative = test_series["cumulative_prob"].iloc[predict_from_week_idx]
    current_week = test_series["Weeknummer"].iloc[predict_from_week_idx]
    
    print(f"\nTest year {test_year:.0f}:")
    print(f"  Current week: {current_week:.0f}")
    print(f"  Current cumulative prob: {current_cumulative:.1f}")
    print(f"  Actual final enrollment: {actual_final:.1f}")
    
    # Build training time series from historical data
    # Use the cumulative values at the SAME week index from historical years
    train_values = []
    for series in all_series[:test_year_idx]:
        if predict_from_week_idx < len(series):
            train_values.append(series["cumulative_prob"].iloc[predict_from_week_idx])
        else:
            # If historical year doesn't have this week, use final value
            train_values.append(series["cumulative_prob"].iloc[-1])
    
    print(f"\nTraining values (cumulative prob at week {predict_from_week_idx}): {[f'{x:.1f}' for x in train_values]}")
    
    if len(train_values) < 2:
        print("ERROR: Need at least 2 years of training data")
        return None
    
    # Try different models
    configs = [
        ((0, 0, 0), (0, 0, 0, 0), "Mean"),  # Simple mean
        ((0, 1, 1), (0, 0, 0, 0), "ARIMA(0,1,1)"),
        ((1, 1, 1), (0, 0, 0, 0), "ARIMA(1,1,1)"),
        ((1, 0, 1), (0, 0, 0, 0), "ARIMA(1,0,1)"),
    ]
    
    results = []
    
    for order, seasonal_order, name in configs:
        try:
            if name == "Mean":
                # Simple baseline: mean of historical values
                predicted_final = np.mean(train_values)
            else:
                model = sm.tsa.SARIMAX(
                    train_values,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                fitted = model.fit(disp=False, maxiter=100)
                
                # Forecast 1 step ahead (the test year)
                forecast = fitted.forecast(steps=1)
                predicted_final = float(forecast[0])
            
            error = predicted_final - actual_final
            pct_error = (error / actual_final) * 100 if actual_final > 0 else 0
            
            print(f"\n{name}:")
            print(f"  Predicted final: {predicted_final:.1f}")
            print(f"  Actual final:    {actual_final:.1f}")
            print(f"  Error:           {error:.1f} ({pct_error:.1f}%)")
            
            results.append({
                "model": name,
                "predicted_final": predicted_final,
                "error": abs(error),
                "pct_error": abs(pct_error)
            })
                
        except Exception as e:
            print(f"\n{name}: FAILED - {e}")
            continue
    
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY:")
        print(f"{'='*70}")
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        best = results_df.loc[results_df["error"].idxmin()]
        print(f"\nBEST: {best['model']} - Error: {best['error']:.1f} ({best['pct_error']:.1f}%)")
        
        return best
    
    return None


def test_within_year_prediction(series_df, predict_from_week_idx):
    """
    Test predicting final enrollment from a point within the same year
    This simulates: "We're at week X, what will the final enrollment be?"
    """
    year = series_df["year"].iloc[0]
    current_week = series_df["Weeknummer"].iloc[predict_from_week_idx]
    current_cumulative = series_df["cumulative_prob"].iloc[predict_from_week_idx]
    actual_final = series_df["cumulative_prob"].iloc[-1]
    
    print(f"\n{'='*70}")
    print(f"Within-Year Test for {year:.0f}")
    print(f"Predict final enrollment from week {current_week:.0f}")
    print(f"{'='*70}")
    print(f"Current cumulative prob: {current_cumulative:.1f}")
    print(f"Actual final enrollment: {actual_final:.1f}")
    print(f"Remaining to enroll: {actual_final - current_cumulative:.1f}")
    
    # Simple extrapolation methods
    results = []
    
    # Method 1: Linear extrapolation
    weeks_elapsed = predict_from_week_idx + 1
    total_weeks = len(series_df)
    if weeks_elapsed > 0:
        rate = current_cumulative / weeks_elapsed
        predicted_linear = rate * total_weeks
        error = abs(predicted_linear - actual_final)
        pct_error = (error / actual_final) * 100 if actual_final > 0 else 0
        
        print(f"\nLinear Extrapolation:")
        print(f"  Predicted final: {predicted_linear:.1f}")
        print(f"  Error: {error:.1f} ({pct_error:.1f}%)")
        
        results.append({
            "model": "Linear",
            "predicted_final": predicted_linear,
            "error": error,
            "pct_error": pct_error
        })
    
    # Method 2: Exponential smoothing on weekly increments
    weekly_increments = series_df["prob_sum"].iloc[:predict_from_week_idx + 1].values
    if len(weekly_increments) >= 3:
        try:
            # Fit exponential smoothing
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            model = SimpleExpSmoothing(weekly_increments)
            fitted = model.fit()
            
            # Forecast remaining weeks
            remaining_weeks = total_weeks - predict_from_week_idx - 1
            forecast_increments = fitted.forecast(steps=remaining_weeks)
            
            predicted_exp = current_cumulative + forecast_increments.sum()
            error = abs(predicted_exp - actual_final)
            pct_error = (error / actual_final) * 100 if actual_final > 0 else 0
            
            print(f"\nExponential Smoothing:")
            print(f"  Predicted final: {predicted_exp:.1f}")
            print(f"  Error: {error:.1f} ({pct_error:.1f}%)")
            
            results.append({
                "model": "Exp Smoothing",
                "predicted_final": predicted_exp,
                "error": error,
                "pct_error": pct_error
            })
        except Exception as e:
            print(f"\nExponential Smoothing: FAILED - {e}")
    
    if results:
        results_df = pd.DataFrame(results)
        print(f"\n{'='*70}")
        print("SUMMARY:")
        print(results_df.to_string(index=False))
        best = results_df.loc[results_df["error"].idxmin()]
        print(f"\nBEST: {best['model']} - Error: {best['error']:.1f} ({best['pct_error']:.1f}%)")
        return best
    
    return None


def main():
    df = load_and_prepare_data()
    
    # Find programme with multi-year data
    opleidingscode, leertraject = find_programme_with_multi_year_data(df, min_years=2, min_weeks_per_year=25)
    
    if opleidingscode is None:
        print("\nNo suitable programme found!")
        return
    
    # Create multi-year time series
    all_series, years = create_multi_year_time_series(df, opleidingscode, leertraject)
    
    if len(years) < 1:
        print("\nNeed at least 1 year of data!")
        return
    
    # Test within-year prediction for 2023 (most complete year)
    print("\n" + "="*70)
    print("TESTING WITHIN-YEAR PREDICTION FOR 2023")
    print("="*70)
    
    series_2023 = all_series[0]  # First year
    
    print("\n" + "="*70)
    print("TEST 1: Predict from week 10")
    test_within_year_prediction(series_2023, predict_from_week_idx=10)
    
    print("\n" + "="*70)
    print("TEST 2: Predict from week 15")
    test_within_year_prediction(series_2023, predict_from_week_idx=15)
    
    print("\n" + "="*70)
    print("TEST 3: Predict from week 20")
    test_within_year_prediction(series_2023, predict_from_week_idx=20)
    
    # If we have 2 years, test cross-year prediction
    if len(years) >= 2:
        print("\n\n" + "="*70)
        print("TESTING CROSS-YEAR PREDICTION (2023 → 2024)")
        print("="*70)
        test_sarima_with_probabilities(all_series, years, test_year_idx=-1, predict_from_week_idx=10)


if __name__ == "__main__":
    main()


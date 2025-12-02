"""
SARIMA Individual Predictor
Predicts final enrollment based on historical trends and current probability sums
"""

import numpy as np
import pandas as pd
import logging
from scripts.utils.helper import get_weeks_list

logger = logging.getLogger(__name__)


class SARIMAIndividualPredictor:
    """
    Predicts final enrollment for programme*leertraject combinations
    using historical final enrollments and simple time series methods.
    
    Works with as few as 2 years of historical data.
    """
    
    def __init__(self, predict_year, individual_start_year=2018, verbose=False):
        """
        Args:
            predict_year: Year to predict for
            individual_start_year: First year to include in historical data
            verbose: Whether to print debug information
        """
        self.predict_year = predict_year
        self.individual_start_year = individual_start_year
        self.verbose = verbose
        
    def fit(self, df_wide, latest_data):
        """
        Prepare historical data for predictions
        
        Args:
            df_wide: Wide format DataFrame with cumulative probabilities per week
            latest_data: DataFrame with actual final enrollments (Aantal_studenten)
        """
        # Store historical final enrollments
        self.historical_data = latest_data[
            (latest_data["Collegejaar"] < self.predict_year) &
            (latest_data["Collegejaar"] >= self.individual_start_year)
        ].copy()
        
        # Store wide format data for current year cumulative values
        self.df_wide = df_wide
        
        if self.verbose:
            logger.info(f"SARIMA fitted with {len(self.historical_data)} historical records")
            logger.info(f"Years: {sorted(self.historical_data['Collegejaar'].unique())}")
        
        return self
    
    def predict(self, prediction_df, predict_week):
        """
        Predict final enrollment for each programme*leertraject
        
        Args:
            prediction_df: DataFrame with programmes to predict for
            predict_week: Current week number (e.g., 28)
            
        Returns:
            DataFrame with SARIMA_individual column added
        """
        df_results = prediction_df.copy()
        df_results["SARIMA_individual"] = np.nan
        
        # Get unique programme*leertraject combinations
        combos = df_results[["Opleidingscode", "Leertraject"]].drop_duplicates()
        
        if self.verbose:
            logger.info(f"Predicting for {len(combos)} programme/leertraject combinations")
            logger.info(f"Current week: {predict_week}")
        
        for idx, combo in combos.iterrows():
            opleidingscode, leertraject = combo["Opleidingscode"], combo["Leertraject"]
            
            # Get current year cumulative value at predict_week
            current_year_data = self.df_wide[
                (self.df_wide["Opleidingscode"] == opleidingscode) &
                (self.df_wide["Leertraject"] == leertraject) &
                (self.df_wide["Collegejaar"] == self.predict_year)
            ]
            
            # Get historical data for this combination
            historical_data = self.df_wide[
                (self.df_wide["Opleidingscode"] == opleidingscode) &
                (self.df_wide["Leertraject"] == leertraject) &
                (self.df_wide["Collegejaar"] < self.predict_year) &
                (self.df_wide["Collegejaar"] >= self.individual_start_year)
            ]
            
            if len(current_year_data) == 0:
                # No current year data
                prediction = np.nan
            elif len(historical_data) == 0:
                # No historical data - can't make prediction
                prediction = np.nan
            else:
                # Make prediction using weekly patterns
                prediction = self._predict_with_weekly_pattern(
                    historical_data, 
                    current_year_data.iloc[0],
                    predict_week,
                    historical_actuals=self.historical_data
                )
            
            # Assign prediction
            mask = (
                (df_results["Opleidingscode"] == opleidingscode) &
                (df_results["Leertraject"] == leertraject)
            )
            df_results.loc[mask, "SARIMA_individual"] = prediction
            
            if self.verbose and not np.isnan(prediction):
                logger.info(
                    f"  {opleidingscode}/{leertraject}: "
                    f"Predicted={prediction:.1f}"
                )
        
        # Count successful predictions
        successful = df_results["SARIMA_individual"].notna().sum()
        if self.verbose:
            logger.info(f"SARIMA predictions: {successful}/{len(df_results)} successful")
        
        return df_results
    
    def _predict_with_weekly_pattern(self, historical_data, current_year_row, predict_week, historical_actuals=None):
        """
        Predict final enrollment using historical weekly patterns
        
        Args:
            historical_data: DataFrame with historical years' weekly cumulative data
            current_year_row: Series with current year's weekly cumulative data
            predict_week: Current week number
            historical_actuals: DataFrame with actual final enrollments for historical years
            
        Returns:
            Predicted final enrollment (float)
        """
        # Get week column name for predict_week
        week_col = str(predict_week)
        
        # Check if we have data for this week
        if week_col not in current_year_row.index or pd.isna(current_year_row[week_col]):
            return np.nan
        
        current_cumulative = float(current_year_row[week_col])
        
        # Get all week columns (they're strings like '39', '40', ..., '1', '2', ...)
        all_cols = list(current_year_row.index)
        week_cols = [col for col in all_cols if col.isdigit()]
        
        if len(week_cols) == 0:
            return np.nan
        
        # Sort week columns in academic year order (39-52, 1-38)
        def week_sort_key(w):
            w_int = int(w)
            if w_int >= 39:
                return w_int - 39  # 39->0, 40->1, ..., 52->13
            else:
                return w_int + 14  # 1->14, 2->15, ..., 38->51
        
        week_cols_sorted = sorted(week_cols, key=week_sort_key)
        
        # Get historical patterns: ratio of final to current week
        ratios = []
        for _, hist_row in historical_data.iterrows():
            if week_col in hist_row.index:
                current_val = hist_row[week_col]
                
                # Get actual final enrollment for this year/programme
                year = hist_row['Collegejaar']
                
                if historical_actuals is not None:
                    actual_row = historical_actuals[
                        (historical_actuals['Collegejaar'] == year) &
                        (historical_actuals['Opleidingscode'] == hist_row['Opleidingscode']) &
                        (historical_actuals['Leertraject'] == hist_row['Leertraject'])
                    ]
                    
                    if len(actual_row) > 0:
                        final_val = float(actual_row.iloc[0]['Aantal_studenten'])
                    else:
                        # Fallback to cumulative sum if actual not found
                        final_week_col = week_cols_sorted[-1]
                        if final_week_col in hist_row.index:
                            final_val = hist_row[final_week_col]
                        else:
                            continue
                else:
                    # Fallback to cumulative sum if actuals not provided
                    final_week_col = week_cols_sorted[-1]
                    if final_week_col in hist_row.index:
                        final_val = hist_row[final_week_col]
                    else:
                        continue
                
                if pd.notna(current_val) and pd.notna(final_val) and current_val > 0:
                    ratio = final_val / current_val
                    ratios.append(ratio)
        
        if len(ratios) == 0:
            # No valid historical ratios - use simple extrapolation
            # Assume linear growth based on weeks elapsed
            weeks_elapsed = len(get_weeks_list(predict_week))
            total_weeks = len(week_cols_sorted)
            
            if weeks_elapsed > 0 and weeks_elapsed < total_weeks:
                extrapolation_ratio = total_weeks / weeks_elapsed
                prediction = current_cumulative * extrapolation_ratio
            else:
                prediction = current_cumulative
        else:
            # Use historical ratios to predict
            if len(ratios) == 1:
                # Only one historical year - use that ratio
                prediction_ratio = ratios[0]
            else:
                # Multiple years - use weighted average (more weight to recent)
                weights = np.exp(np.arange(len(ratios)) * 0.5)  # Exponential weights
                weights = weights / weights.sum()
                prediction_ratio = np.average(ratios, weights=weights)
            
            prediction = current_cumulative * prediction_ratio
        
        # Ensure non-negative
        prediction = max(prediction, 0)
        
        if self.verbose:
            logger.info(
                f"    Week {predict_week}: current={current_cumulative:.1f}, "
                f"ratios={[f'{r:.2f}' for r in ratios]}, predicted={prediction:.1f}"
            )
        
        return prediction


def predict_with_sarima(df_wide, prediction_df, latest_data, predict_year, predict_week,
                        individual_start_year=2018, verbose=False):
    """
    Convenience function to create predictor and make predictions
    
    Args:
        df_wide: Wide format DataFrame with cumulative probabilities
        prediction_df: DataFrame with programmes to predict for
        latest_data: DataFrame with historical final enrollments
        predict_year: Year to predict for
        predict_week: Current week number
        individual_start_year: First year to include in historical data
        verbose: Whether to print debug information
        
    Returns:
        DataFrame with SARIMA_individual column added
    """
    predictor = SARIMAIndividualPredictor(
        predict_year=predict_year,
        individual_start_year=individual_start_year,
        verbose=verbose
    )
    
    predictor.fit(df_wide, latest_data)
    
    return predictor.predict(prediction_df, predict_week)

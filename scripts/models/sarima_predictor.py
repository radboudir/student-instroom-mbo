# sarima_predictor.py - SARIMA time series predictions for MBO data

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Too few observations to estimate starting parameters*")
warnings.filterwarnings("ignore", category=UserWarning)


class SARIMAPredictor:
    """SARIMA-based predictions for student enrollment"""
    
    def __init__(self, data_individual):
        self.data_individual = data_individual
        
    def predict_with_sarima(self, row, predict_week: int):
        """
        Predict enrollment using SARIMA for a single programme
        
        Args:
            row: DataFrame row with programme info
            predict_week: Week number to predict for
            
        Returns:
            Forecasted value or np.nan if prediction fails
        """
        opleiding = row["Croho groepeernaam"]
        data_prog = self.data_individual[
            self.data_individual["Croho groepeernaam"] == opleiding
        ].copy()
        
        if data_prog.empty:
            return np.nan
        
        # Status mapping - using probabilities from XGBoost
        # If we have enrollment probabilities, use those; otherwise map statuses
        if "Inschrijfstatus" in data_prog.columns:
            # Check if it's already numeric (probabilities)
            if pd.api.types.is_numeric_dtype(data_prog["Inschrijfstatus"]):
                status_col = "Inschrijfstatus"
            else:
                # Map text statuses to numeric
                status_map = {
                    "Enrolled": 1,
                    "Rejected": 1,
                    "Withdrawn": 1,
                    "Received": 0,
                    "Offered": 0,
                    "Submitted": 0,
                }
                data_prog["status_numeric"] = data_prog["Inschrijfstatus"].map(status_map)
                status_col = "status_numeric"
        else:
            return np.nan
        
        # Pivot to wide format (weeks as columns)
        try:
            data_wide = data_prog.pivot_table(
                index=["Croho groepeernaam"],
                columns="Weeknummer",
                values=status_col,
                aggfunc="sum",
                fill_value=0
            )
        except Exception:
            return np.nan
        
        # Get time series
        ts = data_wide.values.flatten()
        
        # Need at least 52 weeks of data for seasonal SARIMA
        if len(ts) < 52 or predict_week is None:
            return np.nan
        
        try:
            # SARIMA(1,1,1)(1,1,1,52) - seasonal period of 52 weeks
            model = sm.tsa.statespace.SARIMAX(
                ts,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=0, maxiter=100)
            
            # Calculate forecast steps
            if predict_week >= 39:
                pred_len = predict_week - 38
            else:
                pred_len = 38 - predict_week
            
            # Forecast
            forecast = results.forecast(steps=pred_len)
            
            # Return the final forecasted value
            return forecast.iloc[-1] if hasattr(forecast, 'iloc') else forecast[-1]
            
        except Exception as e:
            # SARIMA can fail for various reasons (convergence, etc.)
            return np.nan
    
    def predict_all_programmes(self, prediction_df, predict_week: int, parallel=True):
        """
        Run SARIMA predictions for all programmes
        
        Args:
            prediction_df: DataFrame with programmes to predict
            predict_week: Week number to predict for
            parallel: Whether to use parallel processing
            
        Returns:
            DataFrame with SARIMA_individual column added
        """
        df = prediction_df.copy()
        
        logger.info(f"Running SARIMA predictions for {len(df)} programmes...")
        
        if parallel:
            # Parallel processing
            nr_CPU_cores = min(joblib.cpu_count(), 4)  # Limit to 4 cores
            logger.info(f"Using {nr_CPU_cores} CPU cores for parallel processing")
            
            predicted_chunks = joblib.Parallel(n_jobs=nr_CPU_cores)(
                joblib.delayed(self.predict_with_sarima)(row, predict_week)
                for _, row in df.iterrows()
            )
            df["SARIMA_individual"] = predicted_chunks
        else:
            # Sequential processing (for debugging)
            df["SARIMA_individual"] = df.apply(
                lambda row: self.predict_with_sarima(row, predict_week),
                axis=1
            )
        
        # Count successful predictions
        successful = df["SARIMA_individual"].notna().sum()
        logger.info(f"SARIMA predictions: {successful}/{len(df)} successful")
        
        return df


def main():
    """Test SARIMA predictor"""
    import yaml
    from cli import parse_args
    
    args = parse_args()
    
    # Load configuration
    with open("configuration.yaml", "r") as f:
        configuration = yaml.safe_load(f)
    
    # Load data
    logger.info("Loading individual data...")
    individual_data = pd.read_csv("input/applications_enriched_with_context_mboa.csv")
    
    # Preprocess
    column_mapping = {
        "schooljaar_afgeleid": "Collegejaar",
        "opleidingcode": "Croho groepeernaam",
        "leertrajectmbo": "Examentype",
        "week_of_year": "Weeknummer",
        "status_proper_case": "Inschrijfstatus",
    }
    individual_data = individual_data.rename(columns=column_mapping)
    individual_data["Weeknummer"] = pd.to_numeric(individual_data["Weeknummer"], errors='coerce')
    individual_data["Collegejaar"] = pd.to_numeric(individual_data["Collegejaar"], errors='coerce')
    
    # Create predictor
    predictor = SARIMAPredictor(individual_data)
    
    # Load enrollment summary for prediction targets
    latest_data = pd.read_csv("input/inschrijvingen_summary_mboa.csv", sep=";")
    latest_data = latest_data.rename(columns={
        "schooljaar_afgeleid": "Collegejaar",
        "opleidingcode": "Croho groepeernaam",
        "leertrajectmbo": "Examentype",
    })
    latest_data["Collegejaar"] = pd.to_numeric(latest_data["Collegejaar"], errors='coerce')
    
    # Predict for each year/week
    for year in args.years:
        for week in args.weeks:
            logger.info(f"\n{'='*60}")
            logger.info(f"SARIMA Predictions for year {year}, week {week}")
            logger.info(f"{'='*60}")
            
            # Get programmes for this year
            prediction_df = latest_data[latest_data["Collegejaar"] == year].copy()
            
            # Run SARIMA predictions
            results = predictor.predict_all_programmes(
                prediction_df,
                predict_week=week,
                parallel=True
            )
            
            # Show summary
            sarima_preds = results[results["SARIMA_individual"].notna()]
            if not sarima_preds.empty:
                logger.info(f"\nSARIMA Summary:")
                logger.info(f"  Total predicted: {sarima_preds['SARIMA_individual'].sum():.1f}")
                logger.info(f"  Mean per programme: {sarima_preds['SARIMA_individual'].mean():.1f}")
                logger.info(f"  Range: {sarima_preds['SARIMA_individual'].min():.1f} to {sarima_preds['SARIMA_individual'].max():.1f}")
                
                # Save if requested
                if args.write_file:
                    output_path = f"output/sarima_predictions_{year}_week{week}.csv"
                    results.to_csv(output_path, index=False)
                    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

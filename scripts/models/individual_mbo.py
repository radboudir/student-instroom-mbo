# individual_mbo_complete.py - Complete MBO version with Individual_ratio, Individual_mean, and SARIMA_individual

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd
import yaml
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from scripts.prediction_methods.bayesian_models import BayesianRatioRegressor, BayesianClusterRegressor
from scripts.prediction_methods.sarima_individual_predictor import predict_with_sarima
from scripts.utils.helper import get_weeks_list
from cli import parse_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
GROUP_COLS = ["Collegejaar", "Opleidingscode", "Leertraject"]
CATEGORICAL_COLS = ["Opleiding", "Type vooropleiding", "Nationaliteit", "EER", "Geslacht", 
                    "Geverifieerd adres land", "School eerste vooropleiding", "Land code eerste vooropleiding"]
NUMERIC_COLS = ["Sleutel_count", "is_numerus_fixus", "Afstand", "Deadlineweek"]
WEEK_COL = ["Weeknummer"]
TARGET_COL = ['Inschrijfstatus']


def preprocess_mbo_data(df):
    """Preprocess MBO data"""
    column_mapping = {
        "schooljaar_afgeleid": "Collegejaar",
        "opleidingcode": "Opleidingscode",
        "leertrajectmbo": "Leertraject",
        "week_of_year": "Weeknummer",
        "status_proper_case": "Inschrijfstatus",
        "bsnhash": "Sleutel"
    }
    df = df.rename(columns=column_mapping)
    
    # Add Opleidingsnaam if not present (preserve if it exists)
    if "Opleidingsnaam" not in df.columns:
        df["Opleidingsnaam"] = "Onbekend"
    
    df["Datum intrekking vooraanmelding"] = np.nan
    df["is_numerus_fixus"] = 0
    df["Afstand"] = np.nan
    df["Deadlineweek"] = False
    df["Sleutel_count"] = df.groupby(["Collegejaar", "Sleutel"])["Sleutel"].transform("count")
    
    df["Weeknummer"] = pd.to_numeric(df["Weeknummer"], errors='coerce')
    df["Collegejaar"] = pd.to_numeric(df["Collegejaar"], errors='coerce')
    df["Opleidingscode"] = pd.to_numeric(df["Opleidingscode"], errors='coerce')
    df = df[df["Inschrijfstatus"].notna()]
    
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = "Onbekend"
        else:
            df[col] = df[col].fillna("Onbekend")
    
    required_cols = GROUP_COLS + ["Opleidingsnaam"] + CATEGORICAL_COLS + NUMERIC_COLS + WEEK_COL + TARGET_COL + ["Datum intrekking vooraanmelding"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 if col in NUMERIC_COLS else "Onbekend"
    
    return df[required_cols]


def predict_probabilities(df, predict_year, individual_start_year=2018):
    """Predict enrollment probabilities using XGBoost"""
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    train_mask = (df["Collegejaar"] < predict_year) & (df["Collegejaar"] >= individual_start_year)
    test_mask = df["Collegejaar"] == predict_year
    
    train = df[train_mask].copy()
    test = df[test_mask].copy()
    
    status_map = {
        "Enrolled": 1,
        "Received": 0,
        "Offered": 0,
        "Withdrawn": 0,
        "Rejected": 0,
        "Submitted": 0,
    }
    
    train[TARGET_COL[0]] = train[TARGET_COL[0]].map(status_map)
    test[TARGET_COL[0]] = test[TARGET_COL[0]].map(status_map)
    
    train = train[train[TARGET_COL[0]].notna()]
    test = test[test[TARGET_COL[0]].notna()]
    
    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")
    
    if train.empty or test.empty:
        logger.warning("Train or test set is empty!")
        return df
    
    X_train = train.drop(columns=[TARGET_COL[0]])
    y_train = train[TARGET_COL[0]]
    X_test = test.drop(columns=[TARGET_COL[0]])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_COLS),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
             CATEGORICAL_COLS + GROUP_COLS + WEEK_COL),
        ],
        remainder="drop",
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(objective="binary:logistic", learning_rate=0.001, 
                               eval_metric="auc", random_state=0, verbosity=0))
    ])
    
    logger.info("Training XGBoost model...")
    pipeline.fit(X_train, y_train)
    
    logger.info("Making probability predictions...")
    probas = pipeline.predict_proba(X_test)[:, 1]
    
    df.loc[test.index, TARGET_COL[0]] = probas
    
    return df


def transform_to_wide(df, data_latest, individual_start_year=2018):
    """Transform data from long to wide format for Bayesian models"""
    df = df[df["Collegejaar"] >= individual_start_year].copy()
    df[TARGET_COL[0]] = pd.to_numeric(df[TARGET_COL[0]], errors="coerce")
    
    # Convert Weeknummer to int to avoid float column names
    df["Weeknummer"] = df["Weeknummer"].fillna(0).astype(int)
    
    pivot = df.pivot_table(
        index=GROUP_COLS,
        columns="Weeknummer",
        values=TARGET_COL[0],
        aggfunc="sum",
        fill_value=0,
    ).sort_index(axis=1).reset_index()
    
    # Week columns are now integers
    all_cols = list(pivot.columns)
    week_cols = [col for col in all_cols if isinstance(col, int) and col > 0]
    
    # Sort week columns in academic year order (39..52, 1..38)
    def academic_week_key(w):
        if w >= 39:
            return w - 39
        else:
            return w + 14 # 53 - 39 = 14
            
    week_cols_sorted = sorted(week_cols, key=academic_week_key)
    week_cols_str = [str(col) for col in week_cols_sorted]
    
    # Rename week columns to strings
    rename_dict = {col: str(col) for col in week_cols}
    pivot = pivot.rename(columns=rename_dict)
    
    # Reorder columns to match academic year
    pivot = pivot[GROUP_COLS + week_cols_str]
    
    if week_cols_str:
        pivot[week_cols_str] = pivot[week_cols_str].cumsum(axis=1)
    
    merged = pivot.merge(
        data_latest[GROUP_COLS + ['Aantal_studenten']],
        on=GROUP_COLS,
        how='left'
    )
    
    return merged.drop_duplicates()


def filter_data(data, opleidingscode, leertraject, predict_year, covid_year=2020):
    """Filter data for training and target"""
    data = data[data.Collegejaar != covid_year]
    
    train = data[data.Collegejaar < predict_year]
    target = data[
        (data.Collegejaar == predict_year) &
        (data["Opleidingscode"] == opleidingscode) &
        (data.Leertraject == leertraject)
    ]
    
    train_prog = train[
        (train["Opleidingscode"] == opleidingscode) &
        (train.Leertraject == leertraject)
    ]
    
    train_total = train[
        (train.Leertraject == leertraject)
    ]
    
    return train_prog, train_total, target


def predict_student_inflow(df_wide, prediction_df, predict_year, predict_week, print_output=False, verbose=False):
    """Predict student inflow using Bayesian models"""
    df_results = prediction_df.copy()
    df_results["Individual_ratio"] = np.nan
    df_results["Individual_mean"] = np.nan
    
    ratio_predictor = BayesianRatioRegressor(predict_week, verbose=verbose)
    cluster_predictor = BayesianClusterRegressor(predict_week, verbose=verbose)
    
    combos = df_results[["Opleidingscode", "Leertraject"]].drop_duplicates()
    
    logger.info(f"Predicting inflow for {len(combos)} programme/group combinations...")
    
    for idx, combo in combos.iterrows():
        opleidingscode, leertraject = combo
        
        train_prog, train_total, target = filter_data(
            df_wide, opleidingscode, leertraject, predict_year
        )
        
        # Ratio prediction
        try:
            if 'Aantal_studenten' in train_prog.columns and len(train_prog) > 0:
                ratio_predictor.fit(
                    train_prog.drop(columns='Aantal_studenten'), 
                    train_prog['Aantal_studenten'].copy()
                )
                ratio_pred = ratio_predictor.predict(target)[0] if len(target) > 0 else 0
            else:
                ratio_pred = 0
        except (ValueError, IndexError, TypeError, UnboundLocalError) as e:
            if verbose:
                logger.error(f"Error in Ratio prediction for {opleidingscode}, {leertraject}: {e}")
            ratio_pred = 0
        
        # Mean prediction
        try:
            if len(train_prog) > 0 and len(train_total) > 0:
                cluster_predictor.fit(train_prog, train_total, target)
                mean_pred = cluster_predictor.predict()[0]
            else:
                mean_pred = 0
        except (ValueError, IndexError, TypeError, UnboundLocalError):
            mean_pred = 0
        
        mask = (
            (df_results["Opleidingscode"] == opleidingscode) &
            (df_results["Leertraject"] == leertraject)
        )
        df_results.loc[mask, "Individual_ratio"] = ratio_pred
        df_results.loc[mask, "Individual_mean"] = mean_pred
        
        if print_output and (ratio_pred > 0 or mean_pred > 0):
            print(f"Predictions for {opleidingscode}, {leertraject}:")
            print(f"  Individual_ratio: {ratio_pred:.1f}")
            print(f"  Individual_mean: {mean_pred:.1f}")
    
    return df_results


def main():
    args = parse_args()
    
    with open("configuration.yaml", "r") as f:
        configuration = yaml.safe_load(f)
    
    # Load individual data
    logger.info("Loading individual data...")
    individual_data = pd.read_csv("input/applications_enriched_with_context_mboa.csv")
    logger.info(f"Raw data shape: {individual_data.shape}")
    
    # Load latest enrollment data
    logger.info("Loading enrollment summary data...")
    latest_data = pd.read_csv("input/inschrijvingen_summary_mboa.csv", sep=";")
    
    # Clean latest_data
    latest_data = latest_data.rename(columns={
        "schooljaar_afgeleid": "Collegejaar",
        "opleidingcode": "Opleidingscode",
        "leertrajectmbo": "Leertraject",
        "aantal": "Aantal_studenten"
    })
    
    # Merge Opleidingsnaam from individual_data if available
    if "Opleidingsnaam" in individual_data.columns:
        opleidings_mapping = individual_data[["opleidingcode", "Opleidingsnaam"]].drop_duplicates()
        opleidings_mapping = opleidings_mapping.rename(columns={"opleidingcode": "Opleidingscode"})
        latest_data = latest_data.merge(opleidings_mapping, on="Opleidingscode", how="left")
    
    latest_data["Collegejaar"] = pd.to_numeric(latest_data["Collegejaar"], errors='coerce')
    
    logger.info(f"Enrollment summary shape: {latest_data.shape}")
    
    # Preprocess
    logger.info("Preprocessing...")
    processed_data = preprocess_mbo_data(individual_data)
    logger.info(f"Processed data shape: {processed_data.shape}")
    
    individual_start_year = configuration.get("individual_start_year", 2018)
    
    # Predict for each year/week
    for year in args.years:
        for week in args.weeks:
            logger.info(f"\n{'='*60}")
            logger.info(f"Predicting for year {year}, week {week}")
            logger.info(f"{'='*60}")
            
            # Filter data to simulate current week
            # We need to keep:
            # 1. All historical data (Collegejaar < year)
            # 2. Current year data UP TO the current week
            allowed_weeks = get_weeks_list(week)
            
            mask_historical = processed_data["Collegejaar"] < year
            mask_current = (processed_data["Collegejaar"] == year) & (processed_data["Weeknummer"].isin(allowed_weeks))
            
            data_for_prediction = processed_data[mask_historical | mask_current].copy()
            
            logger.info(f"Filtered data for week {week}: {len(data_for_prediction)} rows (Original: {len(processed_data)})")
            
            # Step 1: Predict individual probabilities
            result = predict_probabilities(data_for_prediction, year, individual_start_year)
            
            # Step 2: Transform to wide format for Bayesian models
            logger.info("Transforming data to wide format...")
            df_wide = transform_to_wide(result, latest_data, individual_start_year)
            logger.info(f"Wide format shape: {df_wide.shape}")
            
            # Step 3: Create prediction dataframe
            prediction_df = latest_data[latest_data["Collegejaar"] == year].copy()
            if "Weeknummer" not in prediction_df.columns:
                prediction_df["Weeknummer"] = week
            
            logger.info(f"Prediction dataframe shape: {prediction_df.shape}")
            
            # Step 4: Predict student inflow with Bayesian models
            final_predictions = predict_student_inflow(
                df_wide, prediction_df, year, week,
                print_output=args.print,
                verbose=args.verbose if hasattr(args, 'verbose') else False
            )
            
            # Step 5: Add SARIMA predictions
            logger.info("Adding SARIMA predictions...")
            final_predictions = predict_with_sarima(
                df_wide=df_wide,
                prediction_df=final_predictions,
                latest_data=latest_data,
                predict_year=year,
                predict_week=week,
                individual_start_year=individual_start_year,
                verbose=args.verbose if hasattr(args, 'verbose') else False
            )
            
            # Show summary
            ratio_preds = final_predictions[final_predictions["Individual_ratio"].notna()]
            sarima_preds = final_predictions[final_predictions["SARIMA_individual"].notna()]
            
            logger.info(f"\n=== Prediction Summary ===")
            logger.info(f"Total Individual_ratio: {ratio_preds['Individual_ratio'].sum():.1f}")
            logger.info(f"Total Individual_mean: {ratio_preds['Individual_mean'].sum():.1f}")
            logger.info(f"Total SARIMA_individual: {sarima_preds['SARIMA_individual'].sum():.1f}")
            logger.info(f"Number of programme groups: {len(final_predictions)}")
            
            # Save results if requested
            if args.write_file:
                output_path = f"output/predictions_mbo_{year}_week{week}.xlsx"
                final_predictions.to_excel(output_path, index=False, engine="xlsxwriter")
                logger.info(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()

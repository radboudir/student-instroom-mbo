# bayesian_models.py

# --- Standard library ---
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.api as sm
from statsmodels.tsa.forecasting.theta import ThetaModel
from scipy.stats import norm
from typing import Optional, Dict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Project modules ---
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from scripts.utils.helper import get_weeks_list, get_current_len

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
#  Base Class: Bayesian Kalman-Smoothed Regressors
# ============================================================

class BaseBayesianRegressor(BaseEstimator, RegressorMixin):
    """
    Base class for models using:
    - Kalman-filtered local trend Poisson prior
    - RTS smoothing
    - Precision-weighted prior/posterior combination

    Subclasses implement domain-specific posterior likelihood.
    """

    LOG_OFFSET = 0.5
    EPS = 1e-9
    MAD_OUTLIER_MULT_WEEKLY = 1.5
    MAD_OUTLIER_MULT_YEAR = 1
    ABS_THRESHOLD = 25       
    REL_THRESHOLD = 0.7      

    # General colnames
    PROGRAMME_COL = 'Opleidingscode'
    EXAMTYPE_COL = 'Leertraject'
    TARGET_COL = 'Aantal_studenten'

    # ------------------------------------------------------------
    def __init__(self, predict_week: int,  nf_programmes: dict = None, verbose: bool = True):
        self.verbose = verbose
        self.predict_week = predict_week
        self.nf_programmes = nf_programmes

    # ============================================================
    #  1) PRIOR: Sarima/Theta Trend 
    # ============================================================

    def local_trend_prior(self, counts: np.ndarray):
        """
        Fit a trend using Theta + Arima
        """
        counts = np.asarray(counts, dtype=float)


        self.last_year_value = counts[-1]
        
        if self.verbose:
            print(f'Students in earlier years: {counts}')

        n = len(counts)

        if n < 3:
            # Too few data points OR small counts detected → degenerate prior
            
            # Calculate mean_lambda from counts, or use a default if counts is empty
            lam = float(np.mean(counts)) if counts.size else 0.0
            
            # Original case for n < 3
            mean_rate = lam
            var_rate = np.nanvar(counts)

            alpha = mean_rate**2 / max(var_rate, self.EPS) if var_rate > 0 else 1.0 # Default to 1.0 if variance is 0
            beta = mean_rate / max(var_rate, self.EPS) if var_rate > 0 else 1.0 / max(lam, self.EPS)

            return {
                "alpha": alpha,
                "beta": beta,
                "mean_lambda": mean_rate,
                "var_lambda": var_rate
            }

        # Only run time series forecasting if we didn't trigger the degenerate prior
        time_series_result = self.forecaste_sarima_theta_ensemble(counts)

        # The rest of the original logic for processing the time_series_result follows...
        if time_series_result is not None:
            mean_rate = time_series_result['forecast']
            var_rate = time_series_result['variance']

            # Make sure mean is not below 0
            mean_rate = max(mean_rate, 0)

            # Note: If the time series *itself* produces a very high variance (var_rate), 
            alpha = mean_rate**2 / max(var_rate, self.EPS)
            beta = mean_rate / max(var_rate, self.EPS)

            return {
                "alpha": alpha,
                "beta": beta,
                "mean_lambda": mean_rate,
                "var_lambda": var_rate
            }
        
        # If the ensemble forecast failed, return the high-variance prior manually
        mean_rate = float(np.mean(counts)) if counts.size else 0.0
        var_rate = 10000.0
        alpha = mean_rate**2 / max(var_rate, self.EPS)
        beta = mean_rate / max(var_rate, self.EPS)
        
        return {
            "alpha": alpha,
            "beta": beta,
            "mean_lambda": mean_rate,
            "var_lambda": var_rate
        }
    
    # ------------------------------------------------------------
    def fit_sarimax(self, y: np.ndarray, horizon: int = 1) -> Optional[Dict[str, np.ndarray]]:
        """
        Fit a SARIMAX(0,1,1) model with no trend and return forecasts and their variances.
        """
        y_clean = y[~np.isnan(y)].astype(float)
        if y_clean.size < 1:
            if self.verbose:
                print("Insufficient data points to fit SARIMAX.")
            return None

        try:
            model = sm.tsa.SARIMAX(
                y_clean,
                order=(0, 1, 1),
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=True
            )
            result = model.fit(disp=False)
            forecast_obj = result.get_forecast(steps=horizon)

            # Convert to numpy arrays (even if horizon=1)
            forecast_array = np.asarray(forecast_obj.predicted_mean)
            variance_array = np.asarray(forecast_obj.var_pred_mean)

            return {
                "forecast": forecast_array,
                "variance": variance_array
            }

        except Exception as e:
            if self.verbose:
                print(f"SARIMAX fitting failed: {e}")
            return None

    def fit_theta(self, y: np.ndarray, horizon: int = 1) -> Optional[Dict[str, np.ndarray]]:
        """
        Fit a Theta model (period=1) and return forecasts and variance derived from prediction intervals.
        """
        y_clean = y[~np.isnan(y)].astype(float)
        if y_clean.size < 1:
            if self.verbose:
                print("Insufficient data points to fit Theta model.")
            return None

        try:
            model = ThetaModel(y_clean, period=1)
            result = model.fit()

            forecast_values = result.forecast(horizon)
            alpha = 0.05
            prediction_intervals = result.prediction_intervals(steps=horizon, alpha=alpha)

            # Correct: use .iloc for positional indexing
            z_score = norm.ppf(1 - alpha / 2)
            std_dev = (prediction_intervals.iloc[:, 1] - prediction_intervals.iloc[:, 0]) / (2 * z_score)
            variance_values = np.asarray(std_dev)

            return {
                "forecast": forecast_values,
                "variance": variance_values
            }

        except Exception as e:
            if self.verbose:
                print(f"Theta fitting failed: {e}")
            return None
        
    
    def forecaste_sarima_theta_ensemble(self, y: np.ndarray, horizon: int = 1) -> Optional[Dict[str, np.ndarray]]:
        """
        Fit both SARIMAX and Theta models and return an ensemble forecast
        and ensemble variance (averaged from both models).
        """
        sarimax_result = self.fit_sarimax(y, horizon)
        theta_result = self.fit_theta(y, horizon)

        # If both models fail, return None
        if sarimax_result is None and theta_result is None:
            if self.verbose:
                print("Both SARIMAX and Theta model fitting failed.")
            return None

        # If only one model succeeds, return that model's output
        if sarimax_result is None:
            return theta_result
        if theta_result is None:
            return sarimax_result

        # Both models succeeded: compute ensemble as the mean of forecasts and variances
        ensemble_forecast = (sarimax_result["forecast"] + theta_result["forecast"]) / 2
        ensemble_variance = (sarimax_result["variance"] + theta_result["variance"]) / 2

        return {
            "forecast": ensemble_forecast.item(),
            "variance": ensemble_variance.item()
        }

    # ============================================================
    #  Week weights
    # ============================================================
    
    def _compute_predictive_weights(self, X, y, predict_week, threshold=0.8, sharpness=5.0):
        """
        Compute weekly predictive weights based on Pearson R² scores.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor matrix with weekly columns.
        y : array-like
            Target values.
        predict_week : int
            Week to predict.
        threshold : float, optional
            Minimum R² to select informative weeks, by default 0.8.
        sharpness : float, optional
            Sharpness parameter for softmax weighting, by default 5.0.

        Returns
        -------
        np.ndarray
            Normalized weights for each week (length = number of weeks), reversed so earliest week first.
        """
        def _get_r2(model, col, y_true):
            model.fit(col, y_true)
            y_pred = model.predict(col)
            r2 = r2_score(y, y_pred)
            return max(r2,0)
        
        # --- Prepare data ---
        valid_weeks = list(map(str, reversed(get_weeks_list(predict_week))))
        # Select only the columns that actually exist in X to avoid KeyErrors
        available_weeks = [w for w in valid_weeks if w in X.columns]
        if self.verbose:
            missing = [w for w in valid_weeks if w not in X.columns]
            if missing:
                print(f"Warning: missing week columns for predictive weights: {missing}")
            print(f"Using week columns for predictive weights: {available_weeks}")
        if not available_weeks:
            # If no week columns are present, fall back to a single weight
            if self.verbose:
                print("No available week columns found. Returning default weight [1.0].")
            return np.array([1.0])
        X_mat = X[available_weeks].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        n_weeks = len(available_weeks)

        # --- Compute per-week R² scores ---
        self.informative_r2_ = 0.0
        r2_scores = np.zeros(n_weeks)
        model = LinearRegression(positive=True, fit_intercept=False)
        i = n_weeks - 1  # Initialize i to last index in case loop doesn't run
        if y.size > 1 and y.std() > 0:
            for i, w in enumerate(valid_weeks):
                col = X_mat[:, i]

                # Skip constant columns or target
                if col.std() > 0:
                    r2 = _get_r2(model, col.reshape(-1,1), y)
                    r2_scores[i] = max(r2, 0)

                cols = X_mat[:, :i+1]
                self.informative_r2_ = _get_r2(model, cols, y)  

                # Stop early if threshold reached
                if self.informative_r2_ >= threshold:
                    if self.verbose:
                        print(
                        f"R² = {self.informative_r2_:.2f} up to week {w}. "
                        "All weeks up to this week are selected."
                    )
                    # --- Compute softmax weights ---
                    weighted_scores = softmax(r2_scores * sharpness)

                    # Return weights in chronological order (earliest week first)
                    return np.flip(weighted_scores)
                
        if self.verbose:
            print(f"No week found that was above the threshold. Using all weeks")
            print(f" R² = {self.informative_r2_:.2f}. ")


        # --- Compute softmax weights ---
        if np.any(r2_scores > 0):
            weighted_scores = softmax(r2_scores * sharpness)
        else:
            weighted_scores = np.ones(n_weeks) / n_weeks

        # Update informative_r2_ with all weeks if loop completed
        if y.size > 1 and y.std() > 0 and i >= 0:
            cols = X_mat[:, :i+1]
            self.informative_r2_ = _get_r2(model, cols, y) 

        # Return weights in chronological order (earliest week first)
        return np.flip(weighted_scores)

    # ============================================================
    #  Posterior Combination
    # ============================================================

    def _combine_prior_posterior(self, posterior, lambda_decay, posterior_trust):
        """Precision-weighted combination of prior + posterior."""
        posterior = np.asarray(posterior)

        if posterior.size == 0 or np.allclose(posterior, 0):
            if self.verbose:
                print("Posterior empty → using prior only.")
            return np.array([round(self.prior_mean_)])
        
        n = len(posterior)

        # indices from oldest → newest
        idx = np.arange(n)            # [0, 1, 2, ..., n-1]


        # we want highest weight for the LAST entry
        exponent = (n-1) - idx       

        w_value = np.exp(-lambda_decay * exponent)
        w_value /= w_value.sum()        # normalize

        # weighted mean
        m_post = float(np.sum(w_value * posterior))

        # weighted variance 
        v_post = float(np.sum(w_value * (posterior - m_post)**2) + self.EPS)

        m_prior = self.prior_mean_
        v_prior = self.prior_var_

        if hasattr(self, "informative_r2_"):
            v_prior = v_prior / (1 - (self.informative_r2_ + self.EPS))

            if self.informative_r2_ == 0:
                v_post = 10000000 # arbitrary big number, don't use it at all
            else:
                v_post = v_post / (self.informative_r2_ + self.EPS)

        
        if self.verbose:
            print("=== Prior ===")
            print(f"Mean lambda:     {m_prior:.2f}")
            print(f"Variance lambda: {v_prior:.2f}")
            print("=========================")
            print("=== Posterior ===")
            print(f"Mean: {m_post:.2f}")
            print(f"Var:  {v_post:.2f}")
            print("=================")

        # 1. Calculate raw precisions
        prec_prior = 1.0 / v_prior
        prec_post  = 1.0 / v_post

        # 2. Boost the posterior precision
        prec_post_weighted = prec_post * posterior_trust

        # 3. Combine
        inv = prec_prior + prec_post_weighted
        v_final = 1.0 / inv
        
        m_final = v_final * (m_prior * prec_prior + m_post * prec_post_weighted)

        return np.array([round(m_final)])


# ============================================================
#  Ratio-Based Bayesian Regression
# ============================================================

class BayesianRatioRegressor(BaseBayesianRegressor):
    """
    Bayesian regressor using weekly ratios (X_week / y).
    """

    MAD_OUTLIER_MULT_WEEKLY = 1.5
    MAD_OUTLIER_MULT_YEAR = 1.0

    # ------------------------------------------------------------
    def fit(self, X, y):
        if self.verbose:
            print("=====================================")
            print("Ratio Model Fit")
            print("=====================================")
        
        # Replace nan with 0
        y = np.nan_to_num(y, nan=0)

        prior = self.local_trend_prior(y)
        self.prior_mean_ = prior["mean_lambda"]
        self.prior_var_ = prior["var_lambda"] + self.EPS
        self.prior_info_ = prior

        # Weekly weight selection
        self.week_weights_ = self._compute_predictive_weights(
            X, y, self.predict_week
        )
        
        # Override informative_r2_: The ratio model structure handles the relationship
        # explicitly, so we don't want low raw correlation (R2) to suppress the evidence.
        self.informative_r2_ = 0.9

        # Store years
        self.years_ = np.array(X.Collegejaar)

        # Ratio matrix (weekly cumulative predictors / y)
        self.ratios_ = self._compute_weekly_ratios(X, y)
        self.predict_week_ = self.predict_week
        self.is_fitted_ = True

        return self

    # ------------------------------------------------------------
    def predict(self, X):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit() before predict().")
        
        evidence = self._likelihood_ratio_model(
            X, self.ratios_, self.week_weights_, self.predict_week_
        )
        return self._combine_prior_posterior(evidence, lambda_decay = 0.5, posterior_trust=2.0) # Higher trust in recent (weekly) data
 
    # ============================================================
    #  Ratio Model Helpers
    # ============================================================

    # ------------------------------------------------------------
    @staticmethod
    def _compute_weekly_ratios(X, y):
        y = np.asarray(y).reshape(-1, 1)
        valid_weeks = list(map(str, get_weeks_list(38)))
        available = [w for w in valid_weeks if w in X.columns]
        if len(available) < len(valid_weeks):
            # Prefer to use only available weeks and warn the user
            print(f"Warning: only using week columns that exist for ratios: {available}")
        if not available:
            # No week columns — return an empty ratio matrix
            return np.zeros((len(y), 0))
        ratios = X[available].astype(float).to_numpy() / np.maximum(y, 1e-9)
        return ratios

    # ------------------------------------------------------------
    def _likelihood_ratio_model(self, X_test, ratios, w_week, predict_week):
        """
        Predict final enrollment using current week's cumulative value and historical ratios.
        
        New approach: Use only the current week's cumulative probability and apply
        historical ratio (final / current_week) to predict final enrollment.
        """
        # Get current week column
        week_col = str(predict_week)
        
        if week_col not in X_test.columns:
            # No data for this week - return empty evidence
            return np.array([])
        
        # Get current year's cumulative probability at predict_week
        current_cumulative = X_test[week_col].astype(float).values
        
        # Get all week columns to find the final week
        all_cols = list(X_test.columns)
        week_cols = [col for col in all_cols if col.isdigit()]
        
        if len(week_cols) == 0:
            return np.array([])
        
        # Sort week columns in academic year order
        def week_sort_key(w):
            w_int = int(w)
            if w_int >= 39:
                return w_int - 39
            else:
                return w_int + 14
        
        week_cols_sorted = sorted(week_cols, key=week_sort_key)
        final_week_col = week_cols_sorted[-1]
        
        # Calculate historical ratios: final / current_week
        historical_ratios = []
        for i, ratio_row in enumerate(ratios):
            # ratios has shape (n_historical_years, n_weeks)
            # We need to find the ratio at predict_week
            weeks_list = list(map(str, get_weeks_list(38)))
            
            if week_col in weeks_list:
                week_idx = weeks_list.index(week_col)
                
                if week_idx < len(ratio_row):
                    # ratio_row[week_idx] is cumulative_at_week / final
                    # We want final / cumulative_at_week, so invert it
                    if ratio_row[week_idx] > 0:
                        historical_ratios.append(1.0 / ratio_row[week_idx])
        
        # Initialize evidence
        evidence = np.array([])
        
        if len(historical_ratios) == 0:
            # No valid historical ratios - use simple extrapolation
            weeks_elapsed = len(get_weeks_list(predict_week))
            total_weeks = len(week_cols_sorted)
            
            if weeks_elapsed > 0 and weeks_elapsed < total_weeks:
                extrapolation_ratio = total_weeks / weeks_elapsed
                evidence = current_cumulative * extrapolation_ratio
            else:
                evidence = current_cumulative
        else:
            # Use historical ratios to predict
            if len(historical_ratios) == 1:
                prediction_ratio = historical_ratios[0]
            else:
                # Multiple years - use weighted average (more weight to recent)
                weights = np.exp(np.arange(len(historical_ratios)) * 0.5)
                weights = weights / weights.sum()
                prediction_ratio = np.average(historical_ratios, weights=weights)
            
            evidence = current_cumulative * prediction_ratio
        
        # Apply threshold filtering
        # RELAXED: If we only have one prediction (current week based), we trust it more
        # unless it's completely wild (e.g. > 3x last year or < 0.1x last year)
        
        threshold = np.maximum(self.last_year_value * self.REL_THRESHOLD, self.ABS_THRESHOLD)
        minimum_value = max(self.last_year_value - threshold, 0)
        maximum_value = self.last_year_value + threshold
        
        # Check if evidence is within bounds
        # If it's a single value (which it is now), and it's outside bounds,
        # we might want to clamp it instead of removing it, or just return it if it's reasonable.
        
        # For now, let's clamp it to be safe, but maybe widen the bounds?
        # Or just return it as is if it's the only evidence we have.
        # But the original logic was to filter out "bad" predictions.
        
        # Let's clamp it to [0.5 * last_year, 2.0 * last_year] if last_year > 0
        if self.last_year_value > 0:
            lower_bound = self.last_year_value * 0.2  # Allow drop to 20%
            upper_bound = self.last_year_value * 3.0  # Allow 3x growth
            
            # If evidence is outside these wide bounds, clamp it
            evidence = np.clip(evidence, lower_bound, upper_bound)
        
        if self.verbose:
            print("=== Evidence (Ratio Model) ===")
            print(f"Current week {predict_week}: {current_cumulative}")
            print(f"Historical ratios: {[f'{r:.2f}' for r in historical_ratios]}")
            print(f"Predicted: {[f'{x:.1f}' for x in evidence]}")
            print("==============================")
        
        return evidence
    


# ============================================================
#  Cluster-Level Bayesian Regression
# ============================================================

class BayesianClusterRegressor(BaseBayesianRegressor):
    """
    Simple Bayesian model using cluster-level yearly counts
    as posterior evidence.
    """

    def fit(self, train_prog, train_total, target):
        if self.verbose:
            print("=====================================")
            print("Cluster Model Fit")
            print("=====================================")

        y_programme = train_prog['Aantal_studenten']

        # Replace nan with 0
        y_programme = np.nan_to_num(y_programme, nan=0)

        prior = self.local_trend_prior(y_programme)
        self.prior_mean_ = prior["mean_lambda"]
        self.prior_var_ = prior["var_lambda"] + self.EPS
        self.prior_info_ = prior

        # --- Get the weights ---
        self.weights = self._compute_predictive_weights(train_prog, y_programme, self.predict_week)
        
        # Override informative_r2_: The cluster model structure handles the relationship
        # explicitly, so we don't want low raw correlation (R2) to suppress the evidence.
        self.informative_r2_ = 0.9

        # --- Get the cluster and target current value ---
        self.cluster_data, self.target_current_value = self._get_cluster(train_total, target)

        self.predict_week_ = self.predict_week
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------
    def predict(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit() before predict().")

        evidence = self._cluster_likelihood(self.cluster_data, self.target_current_value)
        return self._combine_prior_posterior(evidence, lambda_decay=0.2, posterior_trust = 1.0)

    # ------------------------------------------------------------

    def _cluster_likelihood(self, cluster_data, target_current_value):
        """
        Calculate evidence using cluster's historical ratios applied to current value.
        
        Args:
            cluster_data: DataFrame with cluster programmes' historical data
            target_current_value: Current year's cumulative value at predict_week
        """
        if len(cluster_data) == 0 or target_current_value == 0:
            return np.array([])
        
        # Get week column
        week_col = str(self.predict_week)
        
        # Calculate historical ratios from cluster: final / current_week
        historical_ratios = []
        for _, row in cluster_data.iterrows():
            if week_col in row.index and self.TARGET_COL in row.index:
                current_val = row[week_col]
                final_val = row[self.TARGET_COL]
                
                if pd.notna(current_val) and pd.notna(final_val) and current_val > 0:
                    ratio = final_val / current_val
                    historical_ratios.append(ratio)
        
        if len(historical_ratios) == 0:
            # No valid historical ratios - use simple extrapolation
            weeks_elapsed = len(get_weeks_list(self.predict_week))
            # Estimate total weeks (approx 39 to 52 depending on how you count, let's say 40 for academic year)
            # Actually we can get it from the data if possible, but let's use a standard academic year length
            total_weeks = 52 
            
            if weeks_elapsed > 0:
                extrapolation_ratio = total_weeks / weeks_elapsed
                # Cap extrapolation to avoid crazy values early on
                extrapolation_ratio = min(extrapolation_ratio, 5.0)
                evidence = np.array([target_current_value * extrapolation_ratio])
            else:
                evidence = np.array([target_current_value])
        else:
            # Use weighted average of cluster ratios (more weight to recent/closer programmes)
            if len(historical_ratios) == 1:
                prediction_ratio = historical_ratios[0]
            else:
                # Use simple average for cluster
                prediction_ratio = np.mean(historical_ratios)
            
            # Apply ratio to current value
            evidence = np.array([target_current_value * prediction_ratio])
        
        if self.verbose:
            print("=== Evidence (Cluster Model) ===")
            print(f"Current value at week {self.predict_week}: {target_current_value:.1f}")
            print(f"Cluster ratios: {[f'{r:.2f}' for r in historical_ratios]}")
            print(f"Predicted: {evidence[0]:.1f}")
            print("==============================")
        
        return evidence
    
            
    # ------------------------------------------------------------
    def _get_cluster(self, train, target, ratio_threshold = 1.5, force_programmes = 5):
        """
        Get cluster of similar programmes and extract current week value for target.
        
        Returns:
            tuple: (cluster_data, target_current_value)
        """
        # Get current week column
        week_col = str(self.predict_week)
        
        # Get target's current cumulative value
        if week_col not in target.columns or len(target) == 0:
            return pd.DataFrame(), 0
        
        target_current_value = float(target[week_col].iloc[0])

        # --- Apply weekly weights ---
        valid_weeks = [str(x) for x in get_weeks_list(self.predict_week)] 
        available_weeks = [w for w in valid_weeks if w in train.columns]
        if len(available_weeks) < len(valid_weeks) and self.verbose:
            print(f"Warning: using only available columns {available_weeks} for clustering (requested {valid_weeks})")

        # --- Filter unreasonable rows (outside threshold) --- 
        threshold = np.maximum(self.last_year_value * self.REL_THRESHOLD, self.ABS_THRESHOLD)

        # Compute min/max bounds
        minimum_value = self.last_year_value - threshold
        maximum_value = self.last_year_value + threshold

        # Keep only evidence values within thresholds
        mask = (train[self.TARGET_COL] >= minimum_value) & (train[self.TARGET_COL] <= maximum_value)
        train = train[mask]

        train_original = train.copy()

        if not available_weeks:
            train_features = np.zeros((len(train), 0))
            target_features = np.zeros((len(target), 0))
        else:
            train_features = train[available_weeks].to_numpy(dtype=float)
            target_features = target[available_weeks].to_numpy(dtype=float)

        # Align weights with available weeks
        w = np.asarray(self.weights)
        if w.size < train_features.shape[1]:
            # pad with last weight value
            if w.size > 0:
                w = np.pad(w, (0, train_features.shape[1] - w.size), constant_values=w[-1])
            else:
                w = np.ones(train_features.shape[1]) / max(1, train_features.shape[1])
        elif w.size > train_features.shape[1]:
            w = w[:train_features.shape[1]]
        train_features = train_features * w
        target_features = target_features * w
        
        # --- Fit NearestNeighbors on population ---
        n_samples = len(train_features)
        n_neighbors = min(10, n_samples - 1) if n_samples > 1 else 1
        
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            
        # Fit the nearest neighbors model
        nn.fit(train_features)

        # --- Find closest population row for each target row ---
        distances, indices = nn.kneighbors(target_features)

        # Flatten results for a single target row
        distances = distances.flatten()
        indices = indices.flatten()

        # --- Sort by distance ascending ---
        sorted_idx = np.argsort(distances)
        distances_sorted = distances[sorted_idx]
        indices_sorted = indices[sorted_idx]

        # --- Filter out dissimilar programmes ---
        if len(distances_sorted) > 0:
            min_d = np.nanmin(distances_sorted)
            mask = distances_sorted <= ratio_threshold * min_d
        else:
            mask = np.array([], dtype=bool)

        indices_filt = indices_sorted[mask]
        distances_filt = distances_sorted[mask]

        # --- Guarantee at least N results if force_programmes is set ---
        if force_programmes is not None:
            n_force = int(force_programmes)
            if len(indices_filt) < n_force:
                # Take top N closest from the sorted arrays (not the filtered ones)
                indices_filt = indices_sorted[:n_force]
                distances_filt = distances_sorted[:n_force]

        # --- Use the final filtered arrays ---
        indices = indices_filt
        distances = distances_filt

        # --- Create a DataFrame of closest rows with distances ---
        closest_df = train_original.iloc[indices].copy()
        closest_df["distance"] = distances

        # Sort by distance ascending
        closest_df = closest_df.sort_values('distance', ascending=False)

        if self.verbose:
            print('Cluster df:')
            # Only print columns that exist
            cols_to_print = ['Collegejaar', self.PROGRAMME_COL, self.EXAMTYPE_COL, 'distance']
            cols_to_print = [c for c in cols_to_print if c in closest_df.columns]
            print(closest_df[cols_to_print])

        return closest_df, target_current_value





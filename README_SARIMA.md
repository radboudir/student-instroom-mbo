# SARIMA Functionality - Disabled

## Status: DISABLED (2025-11-29)

### Reason for Disabling
SARIMA prediction was disabled due to persistent issues with zero/NaN predictions. The user requested to comment out SARIMA functionality to focus on the working Bayesian models (Individual_ratio and Individual_mean).

### Current Implementation
The current `individual_mbo.py` file was restored from `individual_mbo_complete.py` backup, which does NOT include SARIMA functionality. The file only includes:
- **Individual_ratio**: Bayesian ratio-based prediction
- **Individual_mean**: Bayesian cluster-based prediction

### Files Modified
- `scripts/models/individual_mbo.py` - Restored from backup without SARIMA

### How to Re-enable SARIMA (Future Reference)
If you want to restore SARIMA functionality in the future:

1. **Backup files available**:
   - The full implementation with SARIMA debugging is in the git history (if using version control)
   - Check for any `individual_mbo_*.py` backup files in `scripts/models/`

2. **Key issues to resolve before re-enabling**:
   - Type mismatch between `Opleidingscode` (int vs float) causing data filtering issues
   - Double-mapping of `Inschrijfstatus` converting probabilities to zeros
   - Need to use summary data (ground truth) for scaling predictions
   - Historical data showing as zeros despite existing in raw data

3. **SARIMA implementation notes**:
   - Should use predicted probabilities for current year
   - Should use actual historical enrollment data (binary 0/1) for training
   - Needs scaling factor from summary file to match true enrollment magnitude
   - Requires at least 10 weeks of data (reduced from 52)
   - Uses adaptive model complexity based on years of historical data available

### Output Columns
The current model outputs:
- `Individual_ratio` - Bayesian ratio prediction
- `Individual_mean` - Bayesian cluster prediction
- `SARIMA_individual` - NOT INCLUDED (column will not exist in output)

### Testing
To verify the model works without SARIMA:
```bash
python -m scripts.models.individual_mbo -y 2024 -w 4 -wf
```

Expected output file: `output/output_mbo_YYYYMMDD_HHMMSS.xlsx`
Expected columns: `Individual_ratio`, `Individual_mean` (no SARIMA column)

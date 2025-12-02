# Examples and Testing Scripts

This directory contains example scripts and testing utilities for the MBO prediction system.

## Files

### `test_sarima_simple.py`
Simplified SARIMA testing script demonstrating:
- How to use 2023-2024 historical data
- Predicting 2025 enrollment without ground truth
- Simple prediction methods that work with limited data

**Usage:**
```bash
python examples/test_sarima_simple.py
```

### `test_sarima_module.py`
Comprehensive SARIMA testing for final enrollment prediction:
- Tests within-year predictions (predicting final enrollment from mid-year data)
- Tests cross-year predictions (using historical years to predict next year)
- Demonstrates probability sum-based predictions

**Usage:**
```bash
python examples/test_sarima_module.py
```

### `inspect_data.py`
Data inspection utility to examine the structure and content of input CSV files.

**Usage:**
```bash
python examples/inspect_data.py
```

## Purpose

These scripts are provided for:
- **Testing**: Verify the prediction models work correctly
- **Learning**: Understand how the prediction algorithms function
- **Development**: Experiment with new prediction approaches

## Note

These are development/testing scripts and are not required for normal operation of the prediction system. The main prediction pipeline is run through `main.py` in the root directory.

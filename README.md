# event_study

A Python package for conducting event study analysis in finance. Supports Market Model, Market-Adjusted Model, and Fama-French Three Factor Model.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Event study is a standard method in finance and accounting research to measure the impact of a specific event on the value of a firm. This package provides:

- **Market Model**: $\( R_{i,t} = \alpha_i + \beta_i R_{m,t} + \epsilon_{i,t} \$)
- **Market-Adjusted Model**: \( R_{i,t} = R_{m,t} + \epsilon_{i,t} \)
- **Fama-French Three Factor Model**: \( R_{i,t} - R_{f,t} = \alpha_i + \beta_1[R_{m,t} - R_{f,t}] + \beta_2 SMB_t + \beta_3 HML_t + \epsilon_{i,t} \)

## Requirements

- pandas
- statsmodels
- scipy

## Quick Start

```python
import pandas as pd
import event_study.event_analysis as es

# Load your event study data
df_eventstudy = pd.read_csv('your_data.csv')

# Define event windows (relative to event date)
event_window_list = [(-20, -11), (-10, -6), (-5, 10), (11, 20), (21, 60)]

# Run event study
result = es.event_study(
    df_eventstudy=df_eventstudy,
    event_window_list=event_window_list,
    est_window=(-210, -10),
    predict_model='market'
)
```

## Data Format

Your input DataFrame (`df_eventstudy`) must contain the following columns:

| Column | Description |
|--------|-------------|
| `stockid` | Stock code |
| `date` | Trading date |
| `eventdate` | Event date | 
| `sreturn` | Stock return | 
| `mreturn` | Market return | 

### Example Data

| stockid | sreturn | date | mreturn | eventdate |
|---------|---------|------|---------|-----------|
| 600028 | -0.0997 | 2013-03-04 | -0.0461 | 2022-07-15 |
| 600028 | -0.0065 | 2017-08-24 | -0.0057 | 2022-07-15 |
| 600028 | -0.0095 | 2016-12-07 | 0.0048 | 2022-07-15 |

### Notes

- Stock codes with or without leading zeros are both acceptable (automatically standardized)
- Date columns can be in any standard date format (automatically converted)
- The function automatically removes rows with missing values
- Events with fewer than `min_est_days` (default: 200) trading days in the year before the event are excluded
- Events where the stock is suspended on the event date are automatically excluded (if suspension data is provided)

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df_eventstudy` | DataFrame | Required | Input data containing stock returns and event dates |
| `event_window_list` | list of tuples | Required | List of event windows, e.g., `[(-5, 10), (-1, 1)]` |
| `est_window` | tuple | `(-210, -10)` | Estimation window (relative days before event) |
| `predict_model` | str | `'market'` | Prediction model: `'market'`, `'market_adj'`, or `'fama3'` |
| `save_path` | str | Current directory | Path to save output files |
| `suspension_file` | str | `None` | Path to suspension data CSV file |
| `min_est_days` | int | `200` | Minimum trading days required in estimation window |
| `print_event_details` | bool | `False` | Print detailed event information |
| `event_detail_days` | int | `10` | Days around event to print when `print_event_details=True` |
| `check_stockid` | str | `None` | Specific stock ID to debug |

## Output

The function returns a tuple of DataFrames and saves results to CSV/Excel files:

1. **T-test Results** (`*_model_ttest.xlsx`): Statistical significance of CAR across event windows
2. **AR by Company** (`*_model_AR_by_company.csv`): Abnormal returns for each company on each day
3. **AAR Daily** (`*_model_AAR_daily.csv`): Average Abnormal Returns with daily statistics
4. **Exclusion Records** (`*_model_exclusion_records.csv`): Events excluded due to data issues

### Output Statistics

The AAR daily output includes:

- **AAR**: Average Abnormal Return
- **Std. E. AAR**: Standard Error of AAR
- **AAR_T-stat**: T-statistic for AAR
- **ACAR/CAAR**: Average/Cumulative Average CAR
- **Median_AR**: Median Abnormal Return
- **Wilcoxon_Z**: Non-parametric test statistic
- **Positive_AR(%)**: Percentage of positive ARs
- **Sign_Z**: Binomial sign test statistic

## Suspension Data Format

If you provide suspension data, it should be a CSV with columns:

| Column | Description |
|--------|-------------|
| `Stkcd` | Stock code |
| `Suspdate_start` | Suspension start date |
| `Suspdate_end` | Suspension end date |

## Example

See `example.ipynb` for a complete walkthrough with real data.
See  `event_study mannual.pdf` for detailed instructions in Chinese.

## License

MIT License

## Contact
You can contact me through [email](mailto:xiali1118@qq.com).
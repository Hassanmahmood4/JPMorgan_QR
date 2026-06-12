# JPMorgan Quantitative Research — Natural Gas Price Estimation

Forage job simulation project analyzing monthly natural gas prices and building a model to estimate prices for any historical date and extrapolate one year into the future.

## Overview

This project loads monthly natural gas purchase prices (Oct 2020 – Sep 2024), identifies seasonal patterns, and fits a **trend + seasonality** model to:

- **Estimate** prices for any date within the historical range (via interpolation)
- **Forecast** prices up to one year beyond the data (via extrapolation)

## Project Structure

| File | Description |
|---|---|
| `natural_gas_analysis.ipynb` | Main notebook — EDA, visualizations, model training, interactive price lookup |
| `natural_gas_model.py` | Reusable model and `get_price_estimate()` function |
| `Natural Gas Data.csv` | Monthly price data (month-end snapshots) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
git clone https://github.com/Hassanmahmood4/JPMorgan_QR.git
cd JPMorgan_QR
python3 -m pip install -r requirements.txt
jupyter notebook natural_gas_analysis.ipynb
```

## Usage

### Interactive (notebook)

Run **Section 8** in the notebook and enter a date when prompted:

```
Enter a date (YYYY-MM-DD): 2023-07-04

Estimated price on 2023-07-04: $11.42
```

### Programmatic

```python
from natural_gas_model import get_price_estimate

price = get_price_estimate("2023-07-04")
print(f"${price:.2f}")
```

Valid date range: **2020-10-31** through **2025-09-30**.

## Model Approach

1. **Trend** — linear regression on time (captures upward price drift)
2. **Seasonality** — monthly deviation from trend (winter highs, summer lows)
3. **Historical dates** — linear interpolation between month-end fitted values
4. **Future dates** — extend trend and apply the same seasonal pattern

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib
- jupyter

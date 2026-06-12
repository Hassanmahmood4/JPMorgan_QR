# JPMorgan Quantitative Research — Forage Simulation

Natural gas pricing and storage contract valuation for the JPMorgan Chase quantitative research job simulation.

## Tasks

| Notebook | Goal |
|---|---|
| [`task_1.ipynb`](task_1.ipynb) | Estimate natural gas prices for any date (interpolation + 1-year forecast) |
| [`task_2.ipynb`](task_2.ipynb) | Price a gas storage contract from injection/withdrawal schedules |
| [`task_3.ipynb`](task_3.ipynb) | Predict loan default probability and calculate expected loss |
| [`task_4.ipynb`](task_4.ipynb) | Quantize FICO scores into credit ratings (MSE & log-likelihood) |

## Project Structure

| File | Description |
|---|---|
| `task_1.ipynb` | Task 1 — EDA, seasonal analysis, `get_price_estimate()` |
| `task_2.ipynb` | Task 2 — storage contract model, tests, interactive input |
| `task_3.ipynb` | Task 3 — loan default model, expected loss, interactive input |
| `natural_gas_model.py` | Trend + seasonality price model (Task 1) |
| `storage_contract.py` | Contract cash-flow valuation (Task 2) |
| `loan_risk_model.py` | PD model and `calculate_expected_loss()` (Task 3) |
| `task_4.ipynb` | Task 4 — FICO quantization, rating map, `map_rating()` |
| `fico_quantization.py` | MSE and log-likelihood bucket optimization (Task 4) |
| `Natural Gas Data.csv` | Monthly natural gas prices (Oct 2020 – Sep 2024) |
| `Task 3 and 4 Loan Data.csv` | Borrower/loan data for default modeling (Task 3) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
git clone https://github.com/Hassanmahmood4/JPMorgan_QR.git
cd JPMorgan_QR
python3 -m pip install -r requirements.txt
```

Run any notebook:

```bash
python3 -m jupyter notebook task_1.ipynb
python3 -m jupyter notebook task_2.ipynb
python3 -m jupyter notebook task_3.ipynb
python3 -m jupyter notebook task_4.ipynb
```

## Task 1 — Price Estimation

**Run:** `task_1.ipynb` → Section 8

Enter a date when prompted:

```
Enter a date (YYYY-MM-DD): 2023-07-04

Estimated price on 2023-07-04: $11.42
```

**Programmatic:**

```python
from natural_gas_model import get_price_estimate

price = get_price_estimate("2023-07-04")
```

Valid dates: **2020-10-31** through **2025-09-30**.

**Model:** linear trend + monthly seasonal factors; linear interpolation for historical dates; trend extrapolation for the next 12 months.

## Task 2 — Storage Contract Pricing

**Run:** `task_2.ipynb` → Section 4

**Inputs (6):**
1. Injection dates
2. Withdrawal dates
3. Prices (`get_price_estimate` from Task 1, or a price dictionary)
4. Injection/withdrawal rate (max volume per event)
5. Maximum storage volume
6. Storage cost (per unit per day)

**Output:** total contract value = withdrawal revenue − injection cost − storage fees

**Programmatic:**

```python
from natural_gas_model import get_price_estimate
from storage_contract import price_storage_contract

value = price_storage_contract(
    injection_dates=["2021-06-30"],
    withdrawal_dates=["2021-12-31"],
    prices=get_price_estimate,
    rate=5.0,
    max_volume=10.0,
    storage_cost=0.02,
)
```

**Assumptions:** instant transport, zero interest rates, no holiday adjustments.

## Task 3 — Loan Default & Expected Loss

**Run:** `task_3.ipynb` → Section 8

**Goal:** Predict probability of default (PD), then calculate expected loss with a **10% recovery rate**.

$$\text{Expected Loss} = PD \times \text{Loan Amount Outstanding} \times 0.90$$

**Programmatic:**

```python
from loan_risk_model import calculate_expected_loss, get_probability_of_default

loan = {
    "credit_lines_outstanding": 1,
    "loan_amt_outstanding": 5000,
    "total_debt_outstanding": 8000,
    "income": 60000,
    "years_employed": 4,
    "fico_score": 650,
}

pd = get_probability_of_default(loan)
el = calculate_expected_loss(loan)
```

**Models compared:** Logistic Regression, Decision Tree, Random Forest.

## Task 4 — FICO Rating Map

**Run:** `task_4.ipynb` → Section 8

**Goal:** Quantize FICO scores into ratings where **1 = best credit** (highest FICO bucket).

**Methods (dynamic programming):**
- **MSE** — minimize within-bucket squared error
- **Log-likelihood** — maximize fit to default distribution per bucket

**Programmatic:**

```python
from fico_quantization import map_rating, generate_mse_buckets

rating = map_rating(720)  # 1 = best, 10 = worst
buckets = generate_mse_buckets(num_buckets=10)
```

## Requirements

- Python 3.10+
- pandas, numpy, matplotlib, scikit-learn, jupyter

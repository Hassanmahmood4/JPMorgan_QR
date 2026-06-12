"""Natural gas price estimation model for the Forage task."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "Natural Gas Data.csv"
_model = None


class NaturalGasPriceModel:
    """Estimate natural gas prices for any date using trend + seasonality."""

    def __init__(self):
        self.trend_slope = None
        self.trend_intercept = None
        self.seasonal_factors = {}
        self.month_end_dates = None
        self.month_end_fitted = None
        self.last_date = None
        self.forecast_end = None
        self._t0 = None

    def fit(self, dates: pd.Series, prices: pd.Series) -> "NaturalGasPriceModel":
        dates = pd.to_datetime(dates)
        prices = pd.to_numeric(prices)
        order = dates.argsort()
        dates = dates.iloc[order].reset_index(drop=True)
        prices = prices.iloc[order].reset_index(drop=True)

        self._t0 = dates.iloc[0]
        time_days = (dates - self._t0).dt.days.values.astype(float)

        self.trend_slope, self.trend_intercept = np.polyfit(time_days, prices.values, deg=1)
        trend_values = self.trend_slope * time_days + self.trend_intercept
        residuals = prices.values - trend_values

        months = dates.dt.month
        for m in range(1, 13):
            mask = months == m
            self.seasonal_factors[m] = float(residuals[mask].mean()) if mask.any() else 0.0

        mean_seasonal = np.mean(list(self.seasonal_factors.values()))
        self.seasonal_factors = {
            m: v - mean_seasonal for m, v in self.seasonal_factors.items()
        }

        self.month_end_dates = dates.values
        self.month_end_fitted = np.array([self._raw_estimate(d) for d in dates])
        self.last_date = dates.iloc[-1]
        self.forecast_end = self.last_date + pd.DateOffset(years=1)
        return self

    def _days_from_start(self, date: pd.Timestamp) -> float:
        return (date - self._t0).days

    def _raw_estimate(self, date: pd.Timestamp) -> float:
        days = self._days_from_start(date)
        trend = self.trend_slope * days + self.trend_intercept
        seasonal = self.seasonal_factors[date.month]
        return trend + seasonal

    def estimate(self, date) -> float:
        date = pd.Timestamp(date)

        if date > self.forecast_end:
            raise ValueError(
                f"Date {date.date()} is beyond the 1-year forecast horizon "
                f"(max: {self.forecast_end.date()})."
            )

        if date <= self.last_date:
            me_dates = pd.to_datetime(self.month_end_dates)
            if date <= me_dates[0]:
                return float(self.month_end_fitted[0])
            if date >= me_dates[-1]:
                return float(self.month_end_fitted[-1])

            idx = np.searchsorted(me_dates, date, side="right") - 1
            d0, d1 = me_dates[idx], me_dates[idx + 1]
            p0, p1 = self.month_end_fitted[idx], self.month_end_fitted[idx + 1]
            weight = (date - d0) / (d1 - d0)
            return float(p0 + weight * (p1 - p0))

        return float(self._raw_estimate(date))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def get_model() -> NaturalGasPriceModel:
    global _model
    if _model is None:
        data = load_data()
        _model = NaturalGasPriceModel().fit(data["date"], data["price"])
    return _model


def get_price_estimate(date) -> float:
    """
    Estimate the natural gas purchase price for a given date.

    Parameters
    ----------
    date : str, datetime, or pd.Timestamp
        Any date from Oct 2020 through Sep 2025 (1 year beyond the data).

    Returns
    -------
    float
        Estimated price in dollars.
    """
    return get_model().estimate(pd.Timestamp(date))

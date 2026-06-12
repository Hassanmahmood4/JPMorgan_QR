"""Gas storage contract valuation for Task 2."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, Union

import pandas as pd

PriceLookup = Union[Mapping, Callable[[pd.Timestamp], float]]


def _resolve_price(prices: PriceLookup, date: pd.Timestamp) -> float:
    if callable(prices):
        return float(prices(date))
    if date in prices:
        return float(prices[date])
    key = date.strftime("%Y-%m-%d")
    if key in prices:
        return float(prices[key])
    raise KeyError(f"No price provided for {key}")


def _charge_storage(
    inventory: float,
    storage_cost: float,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
) -> float:
    if inventory <= 0 or start is None:
        return 0.0
    days = (end - start).days
    if days <= 0:
        return 0.0
    return inventory * storage_cost * days


def calculate_contract_value(
    injection_dates: Sequence,
    withdrawal_dates: Sequence,
    prices: PriceLookup,
    rate: float,
    max_volume: float,
    storage_cost: float,
) -> float:
    """
    Calculate the total value of a natural gas storage contract.

    Parameters
    ----------
    injection_dates : sequence
        Dates when gas is injected into storage.
    withdrawal_dates : sequence
        Dates when gas is withdrawn from storage.
    prices : dict or callable
        Purchase/sale prices by date, or a function that returns a price for a date.
        Use ``natural_gas_model.get_price_estimate`` from Task 1 if desired.
    rate : float
        Maximum injection or withdrawal volume per event.
    max_volume : float
        Maximum storage capacity.
    storage_cost : float
        Cost per unit of gas per day while held in storage.

    Returns
    -------
    float
        Net contract value (withdrawal revenue - injection cost - storage fees).
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    if max_volume <= 0:
        raise ValueError("max_volume must be positive")
    if storage_cost < 0:
        raise ValueError("storage_cost cannot be negative")

    events: list[tuple[pd.Timestamp, str, int]] = []
    for i, d in enumerate(injection_dates):
        events.append((pd.Timestamp(d), "inject", i))
    for i, d in enumerate(withdrawal_dates):
        events.append((pd.Timestamp(d), "withdraw", i))

    if not events:
        return 0.0

    events.sort(key=lambda item: (item[0], 0 if item[1] == "inject" else 1))

    inventory = 0.0
    contract_value = 0.0
    last_date: pd.Timestamp | None = None

    for date, action, _ in events:
        contract_value -= _charge_storage(inventory, storage_cost, last_date, date)

        if action == "inject":
            volume = min(rate, max_volume - inventory)
            if volume > 0:
                price = _resolve_price(prices, date)
                contract_value -= volume * price
                inventory += volume
        else:
            volume = min(rate, inventory)
            if volume > 0:
                price = _resolve_price(prices, date)
                contract_value += volume * price
                inventory -= volume

        last_date = date

    return float(contract_value)


def price_storage_contract(
    injection_dates: Sequence,
    withdrawal_dates: Sequence,
    prices: PriceLookup,
    rate: float,
    max_volume: float,
    storage_cost: float,
) -> float:
    """Return the total value of a gas storage contract (Task 2 deliverable)."""
    return calculate_contract_value(
        injection_dates=injection_dates,
        withdrawal_dates=withdrawal_dates,
        prices=prices,
        rate=rate,
        max_volume=max_volume,
        storage_cost=storage_cost,
    )

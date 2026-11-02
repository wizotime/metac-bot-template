"""Small safety helpers: numeric percentile repair and a monthly cost cap."""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- percentiles


def make_values_strictly_increasing(
    values: list[float],
    lower_bound: float,
    upper_bound: float,
    open_lower_bound: bool,
    open_upper_bound: bool,
) -> list[float]:
    """Sort values ascending and nudge ties apart so they are strictly increasing.

    Values are kept inside closed bounds. The nudge is tiny (1e-6 of the range).
    """
    vals = sorted(float(v) for v in values)
    span = abs(upper_bound - lower_bound) or max(abs(v) for v in vals) or 1.0
    eps = span * 1e-6
    if not open_lower_bound:
        vals = [max(v, lower_bound) for v in vals]
    if not open_upper_bound:
        vals = [min(v, upper_bound) for v in vals]
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            vals[i] = vals[i - 1] + eps
    if not open_upper_bound and vals and vals[-1] > upper_bound:
        vals[-1] = upper_bound
        for i in range(len(vals) - 2, -1, -1):
            if vals[i] >= vals[i + 1]:
                vals[i] = vals[i + 1] - eps
    return vals


def fix_percentile_list(percentile_list, question):
    """Return Percentile objects with ascending, strictly increasing values."""
    if len(percentile_list) < 2:
        return percentile_list
    ordered = sorted(percentile_list, key=lambda p: p.percentile)
    new_vals = make_values_strictly_increasing(
        [p.value for p in ordered],
        float(question.lower_bound),
        float(question.upper_bound),
        bool(question.open_lower_bound),
        bool(question.open_upper_bound),
    )
    if new_vals != [p.value for p in ordered]:
        logger.warning(
            f"Repaired numeric percentiles for {question.page_url}: "
            f"{[p.value for p in ordered]} -> {new_vals}"
        )
    return [p.model_copy(update={"value": v}) for p, v in zip(ordered, new_vals)]


# ---------------------------------------------------------------- cost cap

# Stop new forecasts once the estimated LLM spend this calendar month (UTC)
# reaches this many USD. Kept well under 10 CHF because forecasting_tools'
# estimate does not include Google Search grounding costs.
MONTHLY_COST_CAP_USD = float(os.getenv("MONTHLY_COST_CAP_USD", "9.0"))
COST_LEDGER_PATH = os.getenv("COST_LEDGER_PATH", ".cost_ledger/ledger.json")


def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def load_ledger(path: str = COST_LEDGER_PATH) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def month_spend(ledger: dict, month: str | None = None) -> float:
    return float(ledger.get(month or current_month(), {}).get("usd", 0.0))


def add_spend(usd: float, path: str = COST_LEDGER_PATH, month: str | None = None) -> float:
    month = month or current_month()
    ledger = load_ledger(path)
    entry = ledger.setdefault(month, {"usd": 0.0, "runs": 0})
    entry["usd"] = round(float(entry.get("usd", 0.0)) + float(usd), 6)
    entry["runs"] = int(entry.get("runs", 0)) + 1
    entry["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2, sort_keys=True)
    return entry["usd"]


def cap_reached(path: str = COST_LEDGER_PATH, cap: float | None = None) -> bool:
    cap = MONTHLY_COST_CAP_USD if cap is None else cap
    return month_spend(load_ledger(path)) >= cap

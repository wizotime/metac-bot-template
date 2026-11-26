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


# ---------------------------------------------------------------- tail mass at open bounds
#
# forecasting_tools puts only (1 - p_max) / 2 beyond an open bound when the highest
# declared percentile lies inside the range (5% with a P90, 2.5% with a P95). The loss
# analysis of 2026-11-26 showed several FutureEval questions resolving beyond an open
# bound with only ~5% there. Rule (backtested on 81 resolved numeric/discrete questions):
#   - every open bound keeps at least TAIL_BASE_MIN (5%, i.e. the old behaviour), and
#   - if the forecaster's own P90 (P10) lies within TAIL_NEAR_FRAC of the range of the
#     open upper (lower) bound or beyond it, that side gets at least TAIL_NEAR_MIN (13%).
# The mass is moved by a monotone piecewise-linear remap of the declared percentile
# levels, so ordering, spacing and Metaculus' CDF standardisation stay intact.

TAIL_BASE_MIN = float(os.getenv("TAIL_BASE_MIN", "0.05"))
TAIL_NEAR_MIN = float(os.getenv("TAIL_NEAR_MIN", "0.13"))
TAIL_NEAR_FRAC = float(os.getenv("TAIL_NEAR_FRAC", "0.10"))


def _cdf_location(value: float, lower: float, upper: float, zero_point: float | None) -> float:
    """Same mapping as forecasting_tools' NumericDistribution (0 = lower, 1 = upper)."""
    if zero_point is not None:
        import math

        ratio = (upper - zero_point) / (lower - zero_point)
        if value == zero_point:
            value += 1e-10
        return (
            math.log((value - lower) * (ratio - 1) + (upper - lower)) - math.log(upper - lower)
        ) / math.log(ratio)
    return (value - lower) / (upper - lower)


def _height_at(points: list[tuple[float, float]], loc: float) -> float:
    """Interpolated CDF height at a location; points = sorted (location, height)."""
    for (l0, h0), (l1, h1) in zip(points, points[1:]):
        if l0 - 1e-12 <= loc <= l1 + 1e-12:
            return h0 if l1 == l0 else h0 + (h1 - h0) * (loc - l0) / (l1 - l0)
    raise ValueError("location outside declared points")


def _value_loc_at_level(points: list[tuple[float, float]], level: float) -> float:
    """Location (cdf x-axis) at which the declared CDF reaches `level`."""
    if level <= points[0][1]:
        return points[0][0]
    for (l0, h0), (l1, h1) in zip(points, points[1:]):
        if h0 <= level <= h1:
            return l0 if h1 == h0 else l0 + (l1 - l0) * (level - h0) / (h1 - h0)
    return points[-1][0]


def widen_open_tails_raw(
    pv: list[tuple[float, float]],
    lower: float,
    upper: float,
    open_lower: bool,
    open_upper: bool,
    zero_point: float | None = None,
    base_min: float = TAIL_BASE_MIN,
    near_min: float = TAIL_NEAR_MIN,
    near_frac: float = TAIL_NEAR_FRAC,
) -> tuple[list[tuple[float, float]], dict]:
    """pv: sorted (percentile in 0..1, value). Returns (new pv, info)."""
    info = {"upper": None, "lower": None}
    if len(pv) < 2 or not (open_lower or open_upper):
        return pv, info
    pv = sorted(pv)
    pts = [(_cdf_location(v, lower, upper, zero_point), p) for p, v in pv]
    p_min, p_max = pts[0][1], pts[-1][1]
    # what forecasting_tools would assign at the bounds
    ext = list(pts)
    if open_upper and ext[-1][0] < 1:
        ext.append((1.0, 1 - 0.5 * (1 - p_max)))
    if open_lower and ext[0][0] > 0:
        ext.insert(0, (0.0, 0.5 * p_min))
    f_lo = _height_at(ext, 0.0) if open_lower else 0.0
    f_hi = _height_at(ext, 1.0) if open_upper else 1.0
    t_lo, t_hi = f_lo, f_hi
    if open_upper:
        near = _value_loc_at_level(pts, 0.9) >= 1 - near_frac
        want = near_min if near else base_min
        if 1 - f_hi < want:
            t_hi = 1 - want
            info["upper"] = ("near" if near else "base", round(1 - f_hi, 4), want)
    if open_lower:
        near = _value_loc_at_level(pts, 0.1) <= near_frac
        want = near_min if near else base_min
        if f_lo < want:
            t_lo = want
            info["lower"] = ("near" if near else "base", round(f_lo, 4), want)
    if (t_lo, t_hi) == (f_lo, f_hi) or not (t_lo < t_hi) or not (f_lo < f_hi):
        return pv, info
    xs, ys = [0.0, f_lo, f_hi, 1.0], [0.0, t_lo, t_hi, 1.0]

    def g(p: float) -> float:
        for j in range(1, 4):
            if p <= xs[j] + 1e-15:
                if xs[j] == xs[j - 1]:
                    return ys[j]
                return ys[j - 1] + (ys[j] - ys[j - 1]) * (p - xs[j - 1]) / (xs[j] - xs[j - 1])
        return 1.0

    out = [(g(p), v) for p, v in pv]
    # anchor the new mass exactly at the bound(s)
    if info["upper"] and not any(v == upper for _, v in out):
        out.append((t_hi, upper))
    if info["lower"] and not any(v == lower for _, v in out):
        out.append((t_lo, lower))
    out.sort(key=lambda x: (x[1], x[0]))
    # keep percentiles strictly increasing and >= 5e-5 apart (forecasting_tools check)
    cleaned: list[tuple[float, float]] = []
    for p, v in out:
        if not (0.0 < p < 1.0):
            continue
        if cleaned and (p - cleaned[-1][0] < 1e-4 or v <= cleaned[-1][1]):
            continue
        cleaned.append((p, v))
    if len(cleaned) < 2:
        return pv, {"upper": None, "lower": None}
    return cleaned, info


def widen_open_tails(percentile_list, question):
    """Percentile-object wrapper used by the bot; returns the list unchanged on any error."""
    try:
        pv = [(float(p.percentile), float(p.value)) for p in percentile_list]
        new_pv, info = widen_open_tails_raw(
            pv,
            float(question.lower_bound),
            float(question.upper_bound),
            bool(question.open_lower_bound),
            bool(question.open_upper_bound),
            getattr(question, "zero_point", None),
        )
        if new_pv == sorted(pv):
            return percentile_list
        proto = percentile_list[0]
        logger.info(f"Tail mass adjusted for {getattr(question, 'page_url', '?')}: {info}")
        return [proto.model_copy(update={"percentile": p, "value": v}) for p, v in new_pv]
    except Exception as e:  # never lose a forecast because of this
        logger.warning(f"widen_open_tails failed ({e}); using original percentiles")
        return percentile_list


# ---------------------------------------------------------------- cost cap

# Stop new forecasts once the estimated LLM spend this calendar month (UTC)
# reaches this many USD. Kept well under 10 CHF because forecasting_tools'
# estimate does not include Google Search grounding costs.
MONTHLY_COST_CAP_USD = float(os.getenv("MONTHLY_COST_CAP_USD", "14.0"))
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

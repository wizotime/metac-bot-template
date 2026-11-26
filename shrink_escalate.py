"""Optional post-processing for SanBe-Bot, all OFF by default.

Flags (environment variables):

SHRINK_TO_COMMUNITY=1        Blend binary forecasts toward the community prediction (CP)
                             in log-odds space, but only if Metaculus returned a CP.
                             (Bot tournaments such as FutureEval/MiniBench hide the CP,
                             so there this is a no-op.)
SHRINK_WEIGHT=0.3            Weight of the CP in that blend (0 = ignore CP, 1 = copy CP).
SHRINK_FALLBACK_WEIGHT=0.0   If no CP is available: shrink toward 50% in log-odds with
                             this weight. Default 0 because the backtest on our resolved
                             binary questions showed no benefit.
CLAMP_MIN / CLAMP_MAX        Final clamp for binary forecasts (default 0.03 / 0.97, the
                             same values the bot already used before this change).

ESCALATE_MODEL=1             For binary and multiple-choice questions where the cheap
                             forecasts look uncertain, run ONE extra forecast with a
                             stronger model and blend it in.
ESCALATE_MODEL_NAME          litellm model name, default "gemini/gemini-4-argon".
ESCALATE_INPUT_USD_PER_M     Price per 1M input tokens used for cost tracking
                             (default 4.0 = Google list price from 2026-12-01).
ESCALATE_OUTPUT_USD_PER_M    Price per 1M output tokens incl. thinking (default 20.0).
ESCALATE_SPREAD=0.15         Escalate if the two cheap forecasts differ by >= this
                             (binary: absolute probability; MC: max option difference).
ESCALATE_MID_LOW / _HIGH     Binary: also escalate if the median lies in [0.35, 0.65].
ESCALATE_WEIGHT=0.5          Weight of the strong model's forecast in the final blend.
ESCALATE_BUDGET_SHARE=0.8    Skip escalation once month spend >= share * monthly cap,
                             so the cap is never reached because of escalations.

Broad escalation (approved 2026-11-26), active from ESCALATE_BROAD_FROM (YYYY-MM-DD,
Europe/Zurich date; empty = never):
ESCALATE_BROAD_FROM          e.g. 2026-12-01. From that day on, binary and MC questions
                             first get ONE cheap forecast. Binary questions whose cheap
                             forecast lies in [ESCALATE_BROAD_LOW, ESCALATE_BROAD_HIGH]
                             and all MC questions then get the strong model INSTEAD of a
                             second cheap forecast. Other binaries get the second cheap
                             forecast and still escalate under the spread rule above.
ESCALATE_BROAD_LOW / _HIGH   default 0.15 / 0.85.
ESCALATE_BROAD_BUDGET_SHARE  default 0.95; replaces ESCALATE_BUDGET_SHARE once broad
                             escalation is active.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning(f"Invalid float for {name}={val!r}; using {default}")
        return default


@dataclass(frozen=True)
class ShrinkEscalateConfig:
    shrink_to_community: bool = False
    shrink_weight: float = 0.3
    shrink_fallback_weight: float = 0.0
    clamp_min: float = 0.03
    clamp_max: float = 0.97
    escalate_model: bool = False
    escalate_model_name: str = "gemini/gemini-4-argon"
    escalate_input_usd_per_m: float = 4.0
    escalate_output_usd_per_m: float = 20.0
    escalate_spread: float = 0.15
    escalate_mid_low: float = 0.35
    escalate_mid_high: float = 0.65
    escalate_weight: float = 0.5
    escalate_budget_share: float = 0.8
    escalate_broad_from: str = ""
    escalate_broad_low: float = 0.15
    escalate_broad_high: float = 0.85
    escalate_broad_budget_share: float = 0.95

    @classmethod
    def from_env(cls) -> "ShrinkEscalateConfig":
        d = cls()
        return cls(
            shrink_to_community=_env_bool("SHRINK_TO_COMMUNITY", d.shrink_to_community),
            shrink_weight=_clip01(_env_float("SHRINK_WEIGHT", d.shrink_weight)),
            shrink_fallback_weight=_clip01(
                _env_float("SHRINK_FALLBACK_WEIGHT", d.shrink_fallback_weight)
            ),
            clamp_min=_env_float("CLAMP_MIN", d.clamp_min),
            clamp_max=_env_float("CLAMP_MAX", d.clamp_max),
            escalate_model=_env_bool("ESCALATE_MODEL", d.escalate_model),
            escalate_model_name=os.getenv("ESCALATE_MODEL_NAME") or d.escalate_model_name,
            escalate_input_usd_per_m=_env_float(
                "ESCALATE_INPUT_USD_PER_M", d.escalate_input_usd_per_m
            ),
            escalate_output_usd_per_m=_env_float(
                "ESCALATE_OUTPUT_USD_PER_M", d.escalate_output_usd_per_m
            ),
            escalate_spread=_env_float("ESCALATE_SPREAD", d.escalate_spread),
            escalate_mid_low=_env_float("ESCALATE_MID_LOW", d.escalate_mid_low),
            escalate_mid_high=_env_float("ESCALATE_MID_HIGH", d.escalate_mid_high),
            escalate_weight=_clip01(_env_float("ESCALATE_WEIGHT", d.escalate_weight)),
            escalate_budget_share=_env_float(
                "ESCALATE_BUDGET_SHARE", d.escalate_budget_share
            ),
            escalate_broad_from=(os.getenv("ESCALATE_BROAD_FROM") or "").strip(),
            escalate_broad_low=_env_float("ESCALATE_BROAD_LOW", d.escalate_broad_low),
            escalate_broad_high=_env_float("ESCALATE_BROAD_HIGH", d.escalate_broad_high),
            escalate_broad_budget_share=_env_float(
                "ESCALATE_BROAD_BUDGET_SHARE", d.escalate_broad_budget_share
            ),
        )


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ------------------------------------------------------------------ binary maths

_EPS = 1e-6


def logit(p: float) -> float:
    p = min(1 - _EPS, max(_EPS, float(p)))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def blend_logodds(p: float, target: float, weight: float) -> float:
    """Move p toward target in log-odds space. weight=0 -> p, weight=1 -> target."""
    w = _clip01(weight)
    return sigmoid((1 - w) * logit(p) + w * logit(target))


def clamp(p: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(p)))


def postprocess_binary(
    p: float, community: float | None, cfg: ShrinkEscalateConfig
) -> float:
    """Apply shrinkage (if enabled) and the final clamp to a binary probability."""
    out = float(p)
    if cfg.shrink_to_community:
        if community is not None and 0.0 < community < 1.0:
            out = blend_logodds(out, community, cfg.shrink_weight)
        elif cfg.shrink_fallback_weight > 0:
            out = blend_logodds(out, 0.5, cfg.shrink_fallback_weight)
    return clamp(out, cfg.clamp_min, cfg.clamp_max)


# ------------------------------------------------------------------ escalation logic


def binary_is_uncertain(preds: list[float], cfg: ShrinkEscalateConfig) -> bool:
    if not preds:
        return False
    spread = max(preds) - min(preds)
    s = sorted(preds)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return spread >= cfg.escalate_spread or (
        cfg.escalate_mid_low <= median <= cfg.escalate_mid_high
    )


def mc_is_uncertain(
    preds: list[dict[str, float]], cfg: ShrinkEscalateConfig
) -> bool:
    """preds: list of {option_name: probability}. Uncertain if forecasts disagree."""
    if len(preds) < 2:
        return False
    names = set(preds[0])
    max_diff = 0.0
    for name in names:
        vals = [p.get(name, 0.0) for p in preds]
        max_diff = max(max_diff, max(vals) - min(vals))
    return max_diff >= cfg.escalate_spread


def blend_binary(base: float, strong: float, weight: float) -> float:
    w = _clip01(weight)
    return (1 - w) * base + w * strong


def blend_mc(
    base: dict[str, float], strong: dict[str, float], weight: float
) -> dict[str, float]:
    w = _clip01(weight)
    mixed = {k: (1 - w) * base[k] + w * strong.get(k, 0.0) for k in base}
    total = sum(mixed.values()) or 1.0
    return {k: v / total for k, v in mixed.items()}


def now_zurich() -> datetime:
    """Current time in Europe/Zurich (fixed UTC+1 fallback if tzdata is missing)."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Europe/Zurich"))
    except Exception:  # pragma: no cover - tzdata missing
        return datetime.now(timezone(timedelta(hours=1)))


def broad_active(cfg: ShrinkEscalateConfig, now: datetime | None = None) -> bool:
    """True once the Zurich date is on/after ESCALATE_BROAD_FROM (and escalation is on)."""
    if not cfg.escalate_model or not cfg.escalate_broad_from:
        return False
    try:
        start = date.fromisoformat(cfg.escalate_broad_from)
    except ValueError:
        logger.warning(f"Invalid ESCALATE_BROAD_FROM={cfg.escalate_broad_from!r}; broad escalation off")
        return False
    if now is None:
        now = now_zurich()
    elif now.tzinfo is not None:
        try:
            from zoneinfo import ZoneInfo

            now = now.astimezone(ZoneInfo("Europe/Zurich"))
        except Exception:  # pragma: no cover
            now = now.astimezone(timezone(timedelta(hours=1)))
    return now.date() >= start


def effective_budget_share(cfg: ShrinkEscalateConfig, now: datetime | None = None) -> float:
    return cfg.escalate_broad_budget_share if broad_active(cfg, now) else cfg.escalate_budget_share


def binary_in_broad_range(p: float, cfg: ShrinkEscalateConfig) -> bool:
    return cfg.escalate_broad_low <= float(p) <= cfg.escalate_broad_high


def escalation_allowed(
    month_spend: float,
    monthly_cap: float,
    cfg: ShrinkEscalateConfig,
    now: datetime | None = None,
) -> bool:
    return month_spend < effective_budget_share(cfg, now) * monthly_cap


def register_escalate_model_pricing(cfg: ShrinkEscalateConfig) -> None:
    """Make sure litellm (and therefore the monthly cost cap) knows the price of the
    escalation model. Without this an unknown model would be tracked as $0."""
    import litellm

    entry = {
        "input_cost_per_token": cfg.escalate_input_usd_per_m / 1e6,
        "output_cost_per_token": cfg.escalate_output_usd_per_m / 1e6,
        "litellm_provider": cfg.escalate_model_name.split("/")[0]
        if "/" in cfg.escalate_model_name
        else "gemini",
        "mode": "chat",
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 65_536,
    }
    names = {cfg.escalate_model_name}
    if "/" in cfg.escalate_model_name:
        names.add(cfg.escalate_model_name.split("/", 1)[1])
    # register_model is litellm's official API (it also clears its model-info cache)
    litellm.register_model({name: dict(entry) for name in names})
    # Some modules bound `model_cost` at import time; make sure they see it too.
    dicts = [litellm.model_cost]
    try:
        from forecasting_tools.ai_models import general_llm as _gl

        dicts.append(_gl.model_cost)
    except Exception:  # pragma: no cover - forecasting_tools not installed
        pass
    for d in dicts:
        for name in names:
            d[name] = {**d.get(name, {}), **entry}
    logger.info(
        f"Escalation model {cfg.escalate_model_name} priced at "
        f"${cfg.escalate_input_usd_per_m}/M in, ${cfg.escalate_output_usd_per_m}/M out"
    )

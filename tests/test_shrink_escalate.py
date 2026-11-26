"""Offline tests (no network, no API keys) for shrink_escalate.py and the bot hooks."""
import asyncio
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shrink_escalate as se
from shrink_escalate import ShrinkEscalateConfig


def test_defaults_are_off(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("SHRINK_", "ESCALATE_", "CLAMP_")):
            monkeypatch.delenv(k, raising=False)
    cfg = ShrinkEscalateConfig.from_env()
    assert cfg.shrink_to_community is False
    assert cfg.escalate_model is False
    assert (cfg.clamp_min, cfg.clamp_max) == (0.03, 0.97)


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("SHRINK_TO_COMMUNITY", "1")
    monkeypatch.setenv("SHRINK_WEIGHT", "0.4")
    monkeypatch.setenv("ESCALATE_MODEL", "true")
    monkeypatch.setenv("ESCALATE_MODEL_NAME", "gemini/gemini-3.1-pro-preview")
    monkeypatch.setenv("ESCALATE_SPREAD", "bogus")
    cfg = ShrinkEscalateConfig.from_env()
    assert cfg.shrink_to_community and cfg.escalate_model
    assert cfg.shrink_weight == 0.4
    assert cfg.escalate_model_name == "gemini/gemini-3.1-pro-preview"
    assert cfg.escalate_spread == 0.15  # invalid -> default


def test_blend_logodds_endpoints_and_symmetry():
    assert se.blend_logodds(0.9, 0.5, 0) == pytest.approx(0.9)
    assert se.blend_logodds(0.9, 0.3, 1) == pytest.approx(0.3)
    a = se.blend_logodds(0.9, 0.5, 0.5)
    b = se.blend_logodds(0.1, 0.5, 0.5)
    assert a == pytest.approx(1 - b)
    assert 0.5 < a < 0.9


def test_postprocess_off_only_clamps():
    cfg = ShrinkEscalateConfig()
    assert se.postprocess_binary(0.995, 0.2, cfg) == 0.97
    assert se.postprocess_binary(0.6, 0.2, cfg) == 0.6


def test_postprocess_uses_cp_when_available():
    cfg = ShrinkEscalateConfig(shrink_to_community=True, shrink_weight=0.5)
    out = se.postprocess_binary(0.9, 0.5, cfg)
    assert out == pytest.approx(se.sigmoid(0.5 * math.log(9)))


def test_postprocess_hidden_cp_is_noop_unless_fallback():
    cfg = ShrinkEscalateConfig(shrink_to_community=True, shrink_weight=0.5)
    assert se.postprocess_binary(0.9, None, cfg) == 0.9
    cfg2 = ShrinkEscalateConfig(shrink_to_community=True, shrink_fallback_weight=0.2)
    assert 0.5 < se.postprocess_binary(0.9, None, cfg2) < 0.9


def test_binary_uncertainty():
    cfg = ShrinkEscalateConfig()
    assert se.binary_is_uncertain([0.2, 0.4], cfg)        # spread 0.2
    assert se.binary_is_uncertain([0.48, 0.52], cfg)      # mid-range
    assert not se.binary_is_uncertain([0.1, 0.12], cfg)
    assert not se.binary_is_uncertain([0.85, 0.9], cfg)


def test_mc_uncertainty_and_blend():
    cfg = ShrinkEscalateConfig()
    assert se.mc_is_uncertain([{"a": 0.6, "b": 0.4}, {"a": 0.4, "b": 0.6}], cfg)
    assert not se.mc_is_uncertain([{"a": 0.6, "b": 0.4}, {"a": 0.55, "b": 0.45}], cfg)
    m = se.blend_mc({"a": 0.6, "b": 0.4}, {"a": 0.2, "b": 0.8}, 0.5)
    assert m == pytest.approx({"a": 0.4, "b": 0.6})
    assert sum(m.values()) == pytest.approx(1)


def test_budget_guard():
    cfg = ShrinkEscalateConfig(escalate_budget_share=0.8)
    assert se.escalation_allowed(9.0, 12.0, cfg)
    assert not se.escalation_allowed(9.7, 12.0, cfg)


def test_pricing_registration_feeds_cost_tracking():
    litellm = pytest.importorskip("litellm")
    cfg = ShrinkEscalateConfig(
        escalate_model_name="gemini/sanbe-test-model",
        escalate_input_usd_per_m=4.0,
        escalate_output_usd_per_m=20.0,
    )
    se.register_escalate_model_pricing(cfg)
    inp, out = litellm.cost_per_token(
        model="gemini/sanbe-test-model", prompt_tokens=10_000, completion_tokens=5_000
    )
    assert inp + out == pytest.approx(10_000 * 4e-6 + 5_000 * 20e-6)
    # the path MonetaryCostManager uses: litellm.completion_cost on a response object
    from litellm import ModelResponse
    from litellm.types.utils import Usage
    resp = ModelResponse(model="gemini/sanbe-test-model",
                         usage=Usage(prompt_tokens=10_000, completion_tokens=5_000, total_tokens=15_000))
    cost = litellm.completion_cost(completion_response=resp, model="gemini/sanbe-test-model")
    assert cost == pytest.approx(0.14)
    # and the fallback used by GeneralLlm.calculate_cost_from_tokens
    from forecasting_tools import GeneralLlm
    llm = GeneralLlm(model="gemini/sanbe-test-model")
    assert llm.calculate_cost_from_tokens(10_000, 5_000) == pytest.approx(0.14)


# ------------------------------------------------------------- bot integration (offline)

def _make_bot(cfg):
    pytest.importorskip("forecasting_tools")
    import main

    main.FallTemplateBot2026._se_cfg = cfg
    main.FallTemplateBot2026._escalated = {}
    bot = main.FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=2,
        publish_reports_to_metaculus=False,
        llms={"default": "gemini/gemini-3.8-flash", "summarizer": "gemini/gemini-3.5-flash-lite",
              "researcher": "no_research", "parser": "gemini/gemini-3.5-flash-lite"},
    )
    return main, bot


def _binary_question(cp=None):
    from forecasting_tools import BinaryQuestion
    return BinaryQuestion(question_text="Test?", page_url="https://example.org/q/1",
                          community_prediction_at_access_time=cp)


def _patch_predictions(monkeypatch, main, bot, cheap_values, strong_value):
    from forecasting_tools import ReasonedPrediction
    calls = {"strong": 0, "cheap": 0}
    it = iter(cheap_values)

    async def fake_make_prediction(self, question, research):
        if self._llm_override.get() is not None:
            calls["strong"] += 1
            return ReasonedPrediction(prediction_value=strong_value, reasoning="strong")
        calls["cheap"] += 1
        return ReasonedPrediction(prediction_value=next(it), reasoning="cheap")

    async def fake_research(self, question):
        return "research"

    monkeypatch.setattr(main.FallTemplateBot2026, "_make_prediction", fake_make_prediction)
    monkeypatch.setattr(main.FallTemplateBot2026, "run_research", fake_research)
    return calls


async def _research_and_aggregate(bot, q):
    notepad = await bot._initialize_notepad(q)
    bot._note_pads.append(notepad)
    res = await bot._research_and_make_predictions(q)
    vals = [p.prediction_value for p in res.predictions]
    return vals, await bot._aggregate_predictions(vals, q)


def test_bot_flags_off_is_plain_median(monkeypatch):
    main, bot = _make_bot(ShrinkEscalateConfig())
    calls = _patch_predictions(monkeypatch, main, bot, [0.45, 0.55], 0.9)
    vals, agg = asyncio.run(_research_and_aggregate(bot, _binary_question(cp=0.1)))
    assert calls["strong"] == 0 and len(vals) == 2
    assert agg == pytest.approx(0.5)


def test_bot_escalates_uncertain_binary(monkeypatch):
    main, bot = _make_bot(ShrinkEscalateConfig(escalate_model=True, escalate_weight=0.5))
    calls = _patch_predictions(monkeypatch, main, bot, [0.45, 0.55], 0.9)
    vals, agg = asyncio.run(_research_and_aggregate(bot, _binary_question()))
    assert calls["strong"] == 1 and len(vals) == 3
    assert agg == pytest.approx(0.5 * 0.5 + 0.5 * 0.9)


def test_bot_no_escalation_when_confident(monkeypatch):
    main, bot = _make_bot(ShrinkEscalateConfig(escalate_model=True))
    calls = _patch_predictions(monkeypatch, main, bot, [0.1, 0.12], 0.9)
    vals, agg = asyncio.run(_research_and_aggregate(bot, _binary_question()))
    assert calls["strong"] == 0
    assert agg == pytest.approx(0.11)


def test_bot_escalation_respects_budget(monkeypatch):
    main, bot = _make_bot(ShrinkEscalateConfig(escalate_model=True, escalate_budget_share=0.8))
    monkeypatch.setattr(main.FallTemplateBot2026, "_month_spend_before_run", 11.5)
    calls = _patch_predictions(monkeypatch, main, bot, [0.45, 0.55], 0.9)
    asyncio.run(_research_and_aggregate(bot, _binary_question()))
    assert calls["strong"] == 0


def test_bot_shrinks_toward_cp(monkeypatch):
    main, bot = _make_bot(ShrinkEscalateConfig(shrink_to_community=True, shrink_weight=0.5))
    _patch_predictions(monkeypatch, main, bot, [0.9, 0.9], 0.9)
    _, agg = asyncio.run(_research_and_aggregate(bot, _binary_question(cp=0.5)))
    assert agg == pytest.approx(se.sigmoid(0.5 * math.log(9)))


def test_bot_escalates_mc(monkeypatch):
    from forecasting_tools import MultipleChoiceQuestion, PredictedOptionList
    from forecasting_tools.data_models.multiple_choice_report import PredictedOption

    def pol(a):
        return PredictedOptionList(predicted_options=[
            PredictedOption(option_name="A", probability=a),
            PredictedOption(option_name="B", probability=1 - a)])

    main, bot = _make_bot(ShrinkEscalateConfig(escalate_model=True, escalate_weight=0.5))
    calls = _patch_predictions(monkeypatch, main, bot, [pol(0.7), pol(0.4)], pol(0.1))
    q = MultipleChoiceQuestion(question_text="MC?", page_url="https://example.org/q/2",
                               options=["A", "B"])
    vals, agg = asyncio.run(_research_and_aggregate(bot, q))
    assert calls["strong"] == 1
    a = {o.option_name: o.probability for o in agg.predicted_options}["A"]
    assert a == pytest.approx(0.5 * 0.55 + 0.5 * 0.1, abs=0.02)


# ------------------------------------------------------------- broad escalation (from 2026-12-01)

from datetime import datetime, timezone


def _broad_cfg(**kw):
    base = dict(escalate_model=True, escalate_weight=0.5, escalate_broad_from="2026-12-01")
    base.update(kw)
    return ShrinkEscalateConfig(**base)


def test_broad_date_boundary_is_zurich():
    cfg = _broad_cfg()
    # 2026-11-30 22:59 UTC = 23:59 Zurich -> still November
    assert not se.broad_active(cfg, datetime(2026, 11, 30, 22, 59, tzinfo=timezone.utc))
    # 2026-11-30 23:00 UTC = 00:00 Zurich on 1 Dec -> active
    assert se.broad_active(cfg, datetime(2026, 11, 30, 23, 0, tzinfo=timezone.utc))
    assert se.broad_active(cfg, datetime(2027, 1, 5, 12, 0, tzinfo=timezone.utc))
    # off when not configured, invalid, or escalation disabled
    assert not se.broad_active(_broad_cfg(escalate_broad_from=""), datetime(2027, 1, 1, tzinfo=timezone.utc))
    assert not se.broad_active(_broad_cfg(escalate_broad_from="bogus"), datetime(2027, 1, 1, tzinfo=timezone.utc))
    assert not se.broad_active(_broad_cfg(escalate_model=False), datetime(2027, 1, 1, tzinfo=timezone.utc))


def test_broad_env_parsing(monkeypatch):
    monkeypatch.setenv("ESCALATE_BROAD_FROM", " 2026-12-01 ")
    cfg = ShrinkEscalateConfig.from_env()
    assert cfg.escalate_broad_from == "2026-12-01"
    assert (cfg.escalate_broad_low, cfg.escalate_broad_high) == (0.15, 0.85)
    assert cfg.escalate_broad_budget_share == 0.95


def test_budget_share_switches_on_date():
    cfg = _broad_cfg()
    nov = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)
    dec = datetime(2026, 12, 1, 12, tzinfo=timezone.utc)
    assert se.effective_budget_share(cfg, nov) == 0.8
    assert se.effective_budget_share(cfg, dec) == 0.95
    assert not se.escalation_allowed(12.0, 14.0, cfg, nov)   # 12 >= 11.2
    assert se.escalation_allowed(12.0, 14.0, cfg, dec)       # 12 < 13.3
    assert not se.escalation_allowed(13.4, 14.0, cfg, dec)


def test_broad_range():
    cfg = _broad_cfg()
    assert se.binary_in_broad_range(0.15, cfg) and se.binary_in_broad_range(0.85, cfg)
    assert se.binary_in_broad_range(0.3, cfg)
    assert not se.binary_in_broad_range(0.1, cfg) and not se.binary_in_broad_range(0.9, cfg)


def _force_broad(monkeypatch, active=True):
    import main
    monkeypatch.setattr(main, "broad_active", lambda cfg, now=None: active and cfg.escalate_model and bool(cfg.escalate_broad_from))


def test_bot_broad_binary_in_range_uses_one_flash_plus_strong(monkeypatch):
    main, bot = _make_bot(_broad_cfg())
    _force_broad(monkeypatch)
    calls = _patch_predictions(monkeypatch, main, bot, [0.3, 0.99], 0.6)
    vals, agg = asyncio.run(_research_and_aggregate(bot, _binary_question()))
    assert calls == {"strong": 1, "cheap": 1} and len(vals) == 2
    assert agg == pytest.approx(0.5 * 0.3 + 0.5 * 0.6)


def test_bot_broad_binary_extreme_keeps_two_flash(monkeypatch):
    main, bot = _make_bot(_broad_cfg())
    _force_broad(monkeypatch)
    calls = _patch_predictions(monkeypatch, main, bot, [0.9, 0.92], 0.5)
    vals, agg = asyncio.run(_research_and_aggregate(bot, _binary_question()))
    assert calls == {"strong": 0, "cheap": 2}
    assert agg == pytest.approx(0.91)


def test_bot_broad_binary_extreme_but_spread_still_escalates(monkeypatch):
    main, bot = _make_bot(_broad_cfg())
    _force_broad(monkeypatch)
    calls = _patch_predictions(monkeypatch, main, bot, [0.9, 0.6], 0.5)
    vals, agg = asyncio.run(_research_and_aggregate(bot, _binary_question()))
    assert calls == {"strong": 1, "cheap": 2} and len(vals) == 3
    assert agg == pytest.approx(0.5 * 0.75 + 0.5 * 0.5)


def test_bot_broad_strong_failure_falls_back_to_second_flash(monkeypatch):
    from forecasting_tools import ReasonedPrediction
    main, bot = _make_bot(_broad_cfg())
    _force_broad(monkeypatch)
    it = iter([0.4, 0.5])
    calls = {"strong": 0, "cheap": 0}

    async def fake_make_prediction(self, question, research):
        if self._llm_override.get() is not None:
            calls["strong"] += 1
            raise RuntimeError("model not found")
        calls["cheap"] += 1
        return ReasonedPrediction(prediction_value=next(it), reasoning="cheap")

    async def fake_research(self, question):
        return "research"

    monkeypatch.setattr(main.FallTemplateBot2026, "_make_prediction", fake_make_prediction)
    monkeypatch.setattr(main.FallTemplateBot2026, "run_research", fake_research)
    vals, agg = asyncio.run(_research_and_aggregate(bot, _binary_question()))
    assert calls == {"strong": 1, "cheap": 2}  # no second strong attempt
    assert agg == pytest.approx(0.45)


def test_bot_broad_respects_095_budget(monkeypatch):
    main, bot = _make_bot(_broad_cfg())
    _force_broad(monkeypatch)
    monkeypatch.setattr(main, "effective_budget_share", lambda cfg, now=None: 0.95)
    monkeypatch.setattr(se, "broad_active", lambda cfg, now=None: True)
    monkeypatch.setattr(main.FallTemplateBot2026, "_month_spend_before_run", 13.4)  # >= 13.3
    calls = _patch_predictions(monkeypatch, main, bot, [0.4, 0.5], 0.9)
    vals, agg = asyncio.run(_research_and_aggregate(bot, _binary_question()))
    assert calls == {"strong": 0, "cheap": 2}
    monkeypatch.setattr(main.FallTemplateBot2026, "_month_spend_before_run", 12.0)  # < 13.3, > 11.2
    calls = _patch_predictions(monkeypatch, main, bot, [0.4, 0.5], 0.9)
    asyncio.run(_research_and_aggregate(bot, _binary_question()))
    assert calls["strong"] == 1


def test_bot_broad_mc_always_escalates(monkeypatch):
    from forecasting_tools import MultipleChoiceQuestion, PredictedOptionList
    from forecasting_tools.data_models.multiple_choice_report import PredictedOption

    def pol(a):
        return PredictedOptionList(predicted_options=[
            PredictedOption(option_name="A", probability=a),
            PredictedOption(option_name="B", probability=1 - a)])

    main, bot = _make_bot(_broad_cfg())
    _force_broad(monkeypatch)
    # the two flash forecasts would agree -> old rule would NOT escalate
    calls = _patch_predictions(monkeypatch, main, bot, [pol(0.7), pol(0.7)], pol(0.3))
    q = MultipleChoiceQuestion(question_text="MC?", page_url="https://example.org/q/3", options=["A", "B"])
    vals, agg = asyncio.run(_research_and_aggregate(bot, q))
    assert calls == {"strong": 1, "cheap": 1}
    a = {o.option_name: o.probability for o in agg.predicted_options}["A"]
    assert a == pytest.approx(0.5, abs=0.02)


def test_bot_before_december_unchanged(monkeypatch):
    main, bot = _make_bot(_broad_cfg())
    _force_broad(monkeypatch, active=False)
    calls = _patch_predictions(monkeypatch, main, bot, [0.3, 0.32], 0.9)
    vals, agg = asyncio.run(_research_and_aggregate(bot, _binary_question()))
    # 0.31 is outside the old [0.35, 0.65] and spread < 0.15 -> no escalation
    assert calls == {"strong": 0, "cheap": 2}
    assert agg == pytest.approx(0.31)


def test_bot_numeric_never_broad(monkeypatch):
    main, bot = _make_bot(_broad_cfg())
    _force_broad(monkeypatch)
    q = _numeric_question()
    assert not bot._should_escalate(q, [object(), object()])


# ------------------------------------------------------------- tail mass at open bounds

import cost_and_percentiles as cp


def _numeric_question(lo=30.0, hi=60.0, open_lo=True, open_hi=True):
    from forecasting_tools import NumericQuestion
    return NumericQuestion(question_text="N?", page_url="https://example.org/q/4",
                           lower_bound=lo, upper_bound=hi, open_lower_bound=open_lo,
                           open_upper_bound=open_hi, unit_of_measure="%")


def _final_cdf(pv, q):
    from forecasting_tools import NumericDistribution, Percentile
    pl = [Percentile(percentile=p, value=v) for p, v in pv]
    pl = cp.widen_open_tails(cp.fix_percentile_list(pl, q), q)
    return [x.percentile for x in NumericDistribution.from_question(pl, q).get_cdf()]


CAPE_VERDE = [(0.05, 41.0), (0.1, 43.5), (0.2, 46.5), (0.4, 49.5), (0.6, 52.0), (0.8, 54.5), (0.9, 57.0), (0.95, 58.5)]
CENTRAL = [(0.05, 36.0), (0.1, 38.0), (0.2, 40.0), (0.4, 44.0), (0.6, 46.0), (0.8, 49.0), (0.9, 51.0), (0.95, 53.0)]


def test_tail_near_upper_bound_gets_floor():
    q = _numeric_question()
    cdf = _final_cdf(CAPE_VERDE, q)
    assert 1 - cdf[-1] >= 0.12          # >= ~13% above the open upper bound
    assert 0.04 <= cdf[0] <= 0.07       # lower side keeps the old ~5%
    assert all(b - a >= 0.01 / 200 - 1e-12 for a, b in zip(cdf, cdf[1:]))  # Metaculus min step
    assert all(b - a <= 0.2 + 1e-9 for a, b in zip(cdf, cdf[1:]))          # Metaculus max step


def test_tail_central_forecast_keeps_old_five_percent():
    q = _numeric_question()
    cdf = _final_cdf(CENTRAL, q)
    # with P5/P95 forecasting_tools alone would leave only ~2.5% per side
    assert 0.045 <= 1 - cdf[-1] <= 0.07
    assert 0.045 <= cdf[0] <= 0.07


def test_tail_closed_bounds_untouched():
    pv = [(0.05, 41.0), (0.5, 50.0), (0.95, 59.5)]
    out, info = cp.widen_open_tails_raw(pv, 30.0, 60.0, False, False)
    assert out == pv and info == {"upper": None, "lower": None}


def test_tail_raw_monotone_and_anchored():
    out, info = cp.widen_open_tails_raw(CAPE_VERDE, 30.0, 60.0, True, True)
    assert info["upper"][0] == "near" and info["upper"][2] == 0.13
    ps = [p for p, _ in out]; vs = [v for _, v in out]
    assert ps == sorted(ps) and vs == sorted(vs)
    assert all(b - a >= 1e-4 for a, b in zip(ps, ps[1:]))
    assert (pytest.approx(0.87), 60.0) in [(pytest.approx(p), v) for p, v in out if v == 60.0]


def test_tail_beyond_bound_values_and_discrete():
    # P90 already beyond the open upper bound -> enough mass already, nothing to add above
    pv = [(0.05, 700.0), (0.1, 800.0), (0.5, 950.0), (0.9, 1050.0), (0.95, 1100.0)]
    out, info = cp.widen_open_tails_raw(pv, 700.0, 1000.0, True, True)
    assert info["upper"] is None
    # discrete question (closed lower, open upper) still validates
    from forecasting_tools import DiscreteQuestion, NumericDistribution, Percentile
    q = DiscreteQuestion(question_text="D?", page_url="https://example.org/q/5", lower_bound=-0.5,
                         upper_bound=10.5, open_lower_bound=False, open_upper_bound=True,
                         cdf_size=12, unit_of_measure="")
    pl = [Percentile(percentile=p, value=v) for p, v in [(0.05, 2), (0.1, 3), (0.5, 7), (0.9, 9.8), (0.95, 10.2)]]
    pl = cp.widen_open_tails(cp.fix_percentile_list(pl, q), q)
    cdf = [x.percentile for x in NumericDistribution.from_question(pl, q).get_cdf()]
    assert cdf[0] == 0.0 and 1 - cdf[-1] >= 0.12


def test_tail_wrapper_never_raises():
    class Bad:
        lower_bound = "x"
    assert cp.widen_open_tails([1, 2], Bad()) == [1, 2]

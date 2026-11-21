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

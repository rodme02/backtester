"""LLM client tests — exercise parsing + caching without hitting Groq."""


import pytest

from backtester.data.llm import (
    SentimentScore,
    _parse_score,
    score_headline,
)


def test_parse_score_clamps():
    assert _parse_score("+0.6") == pytest.approx(0.6)
    assert _parse_score("-0.95") == pytest.approx(-0.95)
    assert _parse_score("12") == 1.0  # clamp
    assert _parse_score("-50") == -1.0
    assert _parse_score("not a score") == 0.0
    assert _parse_score(None) == 0.0


def test_fixture_mode_synthetic_score(monkeypatch, tmp_path):
    monkeypatch.setenv("BACKTESTER_FIXTURE_MODE", "1")
    result = score_headline("AAPL", "Apple beats Q1 expectations")
    assert isinstance(result, SentimentScore)
    assert -1.0 <= result.score <= 1.0
    assert result.backend == "fixture"
    # Determinism: same headline → same score.
    again = score_headline("AAPL", "Apple beats Q1 expectations")
    assert again.score == result.score


def test_fixture_mode_empty_headline_zero_score(monkeypatch):
    monkeypatch.setenv("BACKTESTER_FIXTURE_MODE", "1")
    result = score_headline("AAPL", "")
    assert result.score == 0.0


def test_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv("BACKTESTER_FIXTURE_MODE", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Use a unique header so cache miss is forced.
    with pytest.raises(RuntimeError):
        score_headline("AAPL", "headline-without-key-test", cache=False)

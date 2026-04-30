"""LLM sentiment scoring with on-disk per-headline cache.

Default backend is **Groq** (free tier; ``GROQ_API_KEY`` env var; model
defaults to ``llama-3.3-70b-versatile``). When ``OLLAMA_HOST`` is set,
falls back to a local Ollama server (Apple Silicon MPS path
documented in CLAUDE.md). Each ``(ticker, headline_hash, model_id)``
result is cached so reruns are deterministic and free.

The scoring contract: per headline, return a single floating-point
sentiment score in ``[-1.0, +1.0]`` where +1 = strongly bullish for
the named ticker, -1 = strongly bearish, 0 = neutral / off-topic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ._fixture import fixture_mode_active, load_fixture

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data_cache" / "llm"

DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are a financial-news sentiment annotator. For each headline, "
    "output a single floating-point score in the range [-1, +1] expressing "
    "how bullish (+) or bearish (-) the headline is for the named ticker. "
    "Respond with the score only, no commentary."
)


@dataclass
class SentimentScore:
    ticker: str
    headline: str
    score: float
    model_id: str
    backend: str  # "groq" | "ollama" | "fixture"


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _cache_path(ticker: str, headline_hash: str, model_id: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}_{model_id}_{headline_hash}.json"


def _parse_score(text: str) -> float:
    """Parse a model response into a float clamped to [-1, +1]."""
    if text is None:
        return 0.0
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return 0.0
    try:
        v = float(m.group())
    except ValueError:
        return 0.0
    return max(-1.0, min(1.0, v))


def _score_with_groq(ticker: str, headline: str, model_id: str) -> str:
    load_dotenv(override=False)
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set; sign up free at https://groq.com/")
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "groq SDK not installed; run: pip install -e '.[llm]'"
        ) from exc
    client = Groq(api_key=key)
    user_prompt = f"Ticker: {ticker}\nHeadline: {headline}\nScore:"
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=8,
    )
    return resp.choices[0].message.content


def _score_with_ollama(ticker: str, headline: str, model_id: str) -> str:
    import requests  # imported lazily

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    user_prompt = f"Ticker: {ticker}\nHeadline: {headline}\nScore:"
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 8},
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def score_headline(
    ticker: str,
    headline: str,
    *,
    model_id: str = DEFAULT_MODEL,
    backend: str = "groq",
    cache: bool = True,
) -> SentimentScore:
    """Score a single headline.

    Cached on disk by ``(ticker, headline_hash, model_id)`` so reruns
    are deterministic. ``backend`` is ``"groq"`` (default) or
    ``"ollama"``.
    """
    headline = (headline or "").strip()
    if not headline:
        return SentimentScore(ticker, "", 0.0, model_id, "fixture")

    h = _hash(headline)
    if fixture_mode_active():
        df = load_fixture(f"llm_{ticker.upper()}_{model_id}_{h}.csv")
        if df is not None and not df.empty:
            return SentimentScore(
                ticker, headline, float(df["score"].iloc[0]), model_id, "fixture"
            )
        # Fallback: deterministic synthetic score from hash so CI runs
        # produce stable verdicts when the per-headline fixture isn't
        # built. Maps to [-1, +1] uniformly.
        synth = (int(h, 16) % 2001 - 1000) / 1000.0
        return SentimentScore(ticker, headline, synth, model_id, "fixture")

    if cache:
        path = _cache_path(ticker, h, model_id)
        if path.exists():
            cached = json.loads(path.read_text())
            return SentimentScore(
                ticker, headline, cached["score"], model_id, cached["backend"]
            )

    if backend == "ollama":
        text = _score_with_ollama(ticker, headline, model_id)
    else:
        text = _score_with_groq(ticker, headline, model_id)
    score = _parse_score(text)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(ticker, h, model_id).write_text(
            json.dumps({"score": score, "backend": backend})
        )
    return SentimentScore(ticker, headline, score, model_id, backend)


def score_dataframe(
    df: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    headline_col: str = "title",
    model_id: str = DEFAULT_MODEL,
    backend: str = "groq",
) -> pd.Series:
    """Vectorise ``score_headline`` over a DataFrame of headlines.

    Returns a Series of scores aligned to ``df.index``. Each row uses
    its own ``ticker`` and ``headline``; the per-headline cache makes
    re-runs cheap and deterministic.
    """
    scores = []
    for _, row in df.iterrows():
        result = score_headline(
            ticker=row[ticker_col],
            headline=row[headline_col],
            model_id=model_id,
            backend=backend,
        )
        scores.append(result.score)
    return pd.Series(scores, index=df.index, name="sentiment_score")

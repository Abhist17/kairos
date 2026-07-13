"""
Kairos Engine — News Sentiment Analyzer

Fetches financial news headlines and scores sentiment.
No API key needed — uses Google News RSS (free).

How it works:
  1. Fetch latest headlines for NIFTY/market/India
  2. Score each headline with keyword-based sentiment
  3. Aggregate into a market sentiment score (-1 to +1)
  4. Feed into signal generator as a score component

Keyword categories:
  BEARISH: war, crash, recession, sanctions, sell-off, FII selling, rate hike
  BULLISH: rally, growth, stimulus, rate cut, FII buying, peace deal, record high
  INDIA-SPECIFIC: RBI, budget, GST, monsoon, crude oil, rupee

This is NOT AI/ML sentiment — it's transparent keyword matching.
You can see exactly WHY the engine thinks news is bullish/bearish.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
import requests


@dataclass
class NewsItem:
    title: str
    source: str
    time: str
    sentiment: float  # -1 to +1
    matched_keywords: list[str] = field(default_factory=list)


@dataclass
class NewsSentimentResult:
    overall_score: float  # -1 (very bearish) to +1 (very bullish)
    headline_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    top_headlines: list[NewsItem] = field(default_factory=list)
    last_updated: str = ""
    sentiment_label: str = (
        "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL, VERY_BEARISH, VERY_BULLISH
    )


# Keywords with weights
BEARISH_KEYWORDS = {
    # Geopolitical
    "war": -0.8,
    "conflict": -0.6,
    "attack": -0.7,
    "missile": -0.8,
    "strike": -0.5,
    "sanctions": -0.6,
    "tensions": -0.5,
    "escalat": -0.7,
    "nuclear": -0.9,
    "invasion": -0.8,
    "military": -0.4,
    # Market
    "crash": -0.9,
    "sell-off": -0.8,
    "selloff": -0.8,
    "plunge": -0.8,
    "tumble": -0.7,
    "slump": -0.7,
    "drop": -0.4,
    "fall": -0.3,
    "bear": -0.5,
    "recession": -0.8,
    "correction": -0.5,
    "panic": -0.7,
    "fear": -0.5,
    "volatile": -0.3,
    "uncertainty": -0.4,
    "bloodbath": -0.9,
    "rout": -0.7,
    "collapse": -0.8,
    # India specific
    "fii sell": -0.7,
    "fii outflow": -0.7,
    "fpi sell": -0.7,
    "rate hike": -0.5,
    "inflation high": -0.5,
    "crude oil ris": -0.4,
    "rupee fall": -0.5,
    "rupee weak": -0.5,
    "fiscal deficit": -0.4,
    "downgrade": -0.6,
}

BULLISH_KEYWORDS = {
    # Geopolitical
    "peace": 0.6,
    "ceasefire": 0.7,
    "deal": 0.4,
    "agreement": 0.4,
    "diplomacy": 0.3,
    "talks": 0.2,
    "de-escalat": 0.6,
    # Market
    "rally": 0.7,
    "surge": 0.7,
    "soar": 0.7,
    "jump": 0.5,
    "gain": 0.3,
    "rise": 0.3,
    "bull": 0.5,
    "record high": 0.8,
    "all-time high": 0.8,
    "breakout": 0.6,
    "recovery": 0.5,
    "boom": 0.6,
    "optimis": 0.5,
    "confidence": 0.4,
    # India specific
    "fii buy": 0.7,
    "fii inflow": 0.7,
    "fpi buy": 0.7,
    "rate cut": 0.6,
    "rbi cut": 0.6,
    "stimulus": 0.6,
    "gdp growth": 0.5,
    "reform": 0.4,
    "investment": 0.3,
    "rupee strength": 0.5,
    "crude oil fall": 0.4,
    "upgrade": 0.6,
    "dii buy": 0.5,
    "monsoon normal": 0.3,
}

# News queries for Indian market
NEWS_QUERIES = [
    "NIFTY",
    "Indian stock market",
    "NSE BSE",
    "India economy",
    "global markets today",
]


class NewsSentimentAnalyzer:
    def __init__(self, cache_minutes: int = 5):
        self.cache_minutes = cache_minutes
        self._cached_result: NewsSentimentResult | None = None
        self._cache_time: datetime | None = None

    def analyze(self, extra_queries: list[str] | None = None) -> NewsSentimentResult:
        """Fetch and analyze latest news sentiment."""

        # Return cached result if fresh
        if (
            self._cached_result
            and self._cache_time
            and (datetime.now() - self._cache_time).total_seconds()
            < self.cache_minutes * 60
        ):
            return self._cached_result

        queries = NEWS_QUERIES + (extra_queries or [])
        all_headlines = []

        for query in queries[:5]:  # limit queries
            headlines = self._fetch_google_news(query)
            all_headlines.extend(headlines)

        # Deduplicate by title
        seen = set()
        unique = []
        for h in all_headlines:
            key = h.title.lower()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(h)

        # Score each headline
        scored = []
        for item in unique:
            score, keywords = self._score_headline(item.title)
            item.sentiment = score
            item.matched_keywords = keywords
            scored.append(item)

        # Aggregate
        if not scored:
            result = NewsSentimentResult(
                overall_score=0.0,
                headline_count=0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                last_updated=datetime.now().strftime("%H:%M:%S"),
                sentiment_label="NEUTRAL",
            )
            self._cached_result = result
            self._cache_time = datetime.now()
            return result

        bullish = [s for s in scored if s.sentiment > 0.1]
        bearish = [s for s in scored if s.sentiment < -0.1]
        neutral = [s for s in scored if -0.1 <= s.sentiment <= 0.1]

        # Weighted average (recent headlines weighted more)
        if scored:
            overall = sum(s.sentiment for s in scored) / len(scored)
        else:
            overall = 0.0

        # Clamp to -1, +1
        overall = max(-1.0, min(1.0, overall))

        # Label
        if overall > 0.3:
            label = "VERY_BULLISH"
        elif overall > 0.1:
            label = "BULLISH"
        elif overall < -0.3:
            label = "VERY_BEARISH"
        elif overall < -0.1:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        # Top headlines (sorted by absolute sentiment)
        top = sorted(scored, key=lambda s: abs(s.sentiment), reverse=True)[:5]

        result = NewsSentimentResult(
            overall_score=round(overall, 3),
            headline_count=len(scored),
            bullish_count=len(bullish),
            bearish_count=len(bearish),
            neutral_count=len(neutral),
            top_headlines=top,
            last_updated=datetime.now().strftime("%H:%M:%S"),
            sentiment_label=label,
        )

        self._cached_result = result
        self._cache_time = datetime.now()
        return result

    def _fetch_google_news(self, query: str) -> list[NewsItem]:
        """Fetch headlines from Google News RSS (free, no API key)."""
        try:
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})

            if r.status_code != 200:
                return []

            root = ET.fromstring(r.content)
            items = []

            for item in root.findall(".//item")[:10]:  # max 10 per query
                title = item.findtext("title", "")
                source = item.findtext("source", "")
                pub_date = item.findtext("pubDate", "")

                if title:
                    items.append(
                        NewsItem(
                            title=title,
                            source=source,
                            time=pub_date,
                            sentiment=0.0,
                        )
                    )

            return items

        except Exception:
            return []

    def _score_headline(self, title: str) -> tuple[float, list[str]]:
        """Score a headline using keyword matching."""
        title_lower = title.lower()
        total_score = 0.0
        matched = []

        for keyword, weight in BEARISH_KEYWORDS.items():
            if keyword in title_lower:
                total_score += weight
                matched.append(f"{keyword}({weight:+.1f})")

        for keyword, weight in BULLISH_KEYWORDS.items():
            if keyword in title_lower:
                total_score += weight
                matched.append(f"{keyword}({weight:+.1f})")

        # Clamp
        total_score = max(-1.0, min(1.0, total_score))

        return round(total_score, 3), matched

    def format_status(self) -> str:
        """Format for terminal display."""
        if not self._cached_result:
            return "  📰 News: not loaded yet"

        r = self._cached_result
        GRN = "\033[92m"
        RED = "\033[91m"
        YEL = "\033[93m"
        B = "\033[1m"
        RS = "\033[0m"

        color = (
            GRN if r.overall_score > 0.1 else (RED if r.overall_score < -0.1 else YEL)
        )

        lines = [
            f"  📰 News: {color}{B}{r.sentiment_label}{RS} "
            f"({r.overall_score:+.2f}) "
            f"│ {GRN}Bull:{r.bullish_count}{RS} "
            f"{RED}Bear:{r.bearish_count}{RS} "
            f"Neut:{r.neutral_count} "
            f"│ {r.headline_count} headlines @ {r.last_updated}"
        ]

        for h in r.top_headlines[:3]:
            hc = GRN if h.sentiment > 0 else (RED if h.sentiment < 0 else YEL)
            kw = ", ".join(h.matched_keywords[:3]) if h.matched_keywords else "neutral"
            title_short = h.title[:60] + "..." if len(h.title) > 60 else h.title
            lines.append(f"    {hc}{h.sentiment:+.2f}{RS} {title_short} [{kw}]")

        return "\n".join(lines)

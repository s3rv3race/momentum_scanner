# -*- coding: utf-8 -*-
"""
Momentum Breakout Options Scanner -- Flask API
Indicators: Wilder RSI, ADX, MACD, ATR, ROC, CMF, MFI, OBV, TTM Squeeze
Patterns:   VCP, Engulfing, Marubozu, Inside Bar Breakout, 3-Bar Play
Universe:   nasdaq_screener.csv only
"""

from flask import Flask, jsonify, request, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import os
import warnings
import traceback
import json
import threading
import time
import google.generativeai as genai
import tradier
import webull

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')

# ==================== DISCORD WEBHOOK ====================
# Paste your Discord webhook URL here
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1491624735711428728/wwLxGU-Zblv-n-8-5zlZHxYZrcZt7gsb6DJ-IU9g8ffSqi0qgg6Vft8gLPXF1VcNsMkb'

# ==================== GEMINI API ====================
# Get your free API key at: https://aistudio.google.com/apikey
# Used for: pre-filter scoring of candidates + per-signal analysis + Discord commentary
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or 'AIzaSyAFJLa5ctWFxAQ0vibRwm2326vVyuSiU6A'

# ==================== ANTHROPIC API (claude_analyzer.py only) ====================
# Only needed if you run backtest analysis via claude_analyzer.py.
# flask_app.py itself uses Gemini — this key stays here because claude_analyzer.py
# imports it from this file.
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY') or ''

# ==================== TRADIER API (real-time options data) ====================
# Paste your Tradier developer token here after your account is approved.
# Get one free at: https://tradier.com/individual/api
# Provides: real-time options chains with real Greeks (delta/gamma/theta/vega).
# Leave blank to fall back to yfinance (15-min delayed options).
TRADIER_TOKEN = os.environ.get('TRADIER_TOKEN') or ''
if TRADIER_TOKEN:
    tradier.init(TRADIER_TOKEN, live=True)

# ==================== WEBULL API (real-time stock data) ====================
# Paste your Webull app_key and app_secret from the developer portal:
#   webull.com/center#openApiManagement  →  API Management  →  My Application
# Provides: real-time stock quotes + intraday OHLCV bars (5-min live candles).
# NOTE: Webull's official API does NOT have options chains — that uses Tradier/yfinance.
WEBULL_APP_KEY    = os.environ.get('WEBULL_APP_KEY')    or '5d683de0f6b6d8506d3de5d207b990d2'
WEBULL_APP_SECRET = os.environ.get('WEBULL_APP_SECRET') or 'aca1cb446809017863e618387577f116'
if WEBULL_APP_KEY and WEBULL_APP_SECRET:
    webull.init(WEBULL_APP_KEY, WEBULL_APP_SECRET, region='us')

# ==================== CONFIGURATION ====================
CONFIG = {
    # STRATEGY PARAMETERS
    'min_price_move': 0.7,         # Lowered 0.9→0.7 (Claude rec): ~3x more signals with rs<30 as quality gate
    'min_volume_ratio': 1.4,       # Lowered 1.8→1.4 (Claude rec): vol<2.5 = 67% WR vs vol>2.5 = 43% WR
    'rsi_min': 52,                 # Raised 50→52: firmly in uptrend territory
    'rsi_max': 72,                 # Lowered 75→72: avoid extended/overbought setups
    'atr_stop_multiplier': 1.5,

    # TREND FILTERS (new)
    'min_adx': 15,                 # Lowered to 15: ADX is suppressed across market; big movers (ALAB,AAOI) have low ADX today
    'max_adx': 35,                 # NEW (Claude rec): ADX>35 has 11% WR vs 56% for ADX<=35
    'require_macd_aligned': True,  # MACD direction must match trade direction
    'max_gap_pct': 8.0,            # Reject gaps >8% — earnings/binary event risk

    # CHART PATTERN QUALITY
    'min_pattern_quality': 50,     # Lowered 55→50 (Claude rec): pat_q>=75 = 0% WR — quality score inverse at top end
    'max_gap_and_go_quality': 74,  # NEW: GAP_AND_GO pat_q>=75 = 0/3 WR — LHX,TRS,PSTG lost; VCP exempt
    'skip_v_reversals': True,      # Now True: V-reversals have poor follow-through
    'bullish_only': True,          # Disable BEARISH (0/1 win rate in backtest, scanner not calibrated for shorts)

    # OPTION SELECTION — room for the move to play out
    'dte_range': [7, 21],          # Widened 3-10→7-21: reduces theta decay risk
    'delta_range': [0.40, 0.60],   # Tightened upper end: avoid near-ITM lottery tickets
    'max_theta_per_day': -1.0,
    'min_open_interest': 200,      # Raised 100→200: better execution liquidity
    'max_bid_ask_spread': 0.10,    # Tightened 0.15→0.10: no wide-spread garbage
    'min_premium': 0.20,           # New: reject sub-$0.20 near-expiry lotto contracts

    # RISK MANAGEMENT
    'profit_target': 0.50,         # 50% profit target for momentum plays
    'stop_loss': -0.35,
    'min_composite_score': 25,     # Lowered 30→25 (Claude rec): conditional rules are quality gate, not composite
    'max_positions': 5,

    # STOCK FILTERS
    'min_stock_price': 20.0,
    'max_stock_price': 500.0,      # Lowered 800→500: $500-800 stocks have enormous spreads
    'min_volume': 1_000_000,
    'min_market_cap': 2_000_000_000,

    # PERFORMANCE
    'scan_workers': 12,
}


# ==================== TECHNICAL INDICATORS ====================

def calculate_rsi(prices, period=14):
    """Wilder's smoothing RSI — matches ThinkorSwim, TradingView, Bloomberg"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    if avg_loss.iloc[-1] == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    return float(100 - (100 / (1 + rs)))


def calculate_atr(hist, period=14):
    high, low, close = hist['High'], hist['Low'], hist['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean().iloc[-1]


def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    # Expanding histogram = accelerating momentum — most powerful MACD signal
    hist_expanding = bool(
        len(histogram) >= 3 and
        histogram.iloc[-1] > histogram.iloc[-2] > 0
    )
    return {
        'macd': float(macd_line.iloc[-1]),
        'signal': float(signal_line.iloc[-1]),
        'histogram': float(histogram.iloc[-1]),
        'bullish': bool(histogram.iloc[-1] > 0),
        'expanding': hist_expanding
    }


def calculate_adx(hist, period=14):
    """Average Directional Index — measures trend strength, not direction"""
    try:
        high = hist['High']
        low  = hist['Low']
        close = hist['Close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low  - close.shift()).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        h_diff = high.diff()
        l_diff = -low.diff()

        dm_plus  = np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0.0)
        dm_minus = np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0.0)

        atr14    = tr.ewm(alpha=1/period, adjust=False).mean()
        di_plus  = 100 * pd.Series(dm_plus,  index=tr.index).ewm(alpha=1/period, adjust=False).mean() / atr14
        di_minus = 100 * pd.Series(dm_minus, index=tr.index).ewm(alpha=1/period, adjust=False).mean() / atr14

        denom = (di_plus + di_minus).replace(0, np.nan)
        dx    = 100 * (di_plus - di_minus).abs() / denom
        adx   = dx.ewm(alpha=1/period, adjust=False).mean()

        adx_val  = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
        dip_val  = float(di_plus.iloc[-1])
        dim_val  = float(di_minus.iloc[-1])
        rising   = bool(len(adx) >= 4 and adx.iloc[-1] > adx.iloc[-3])

        return {
            'adx': round(adx_val, 1),
            'di_plus': round(dip_val, 1),
            'di_minus': round(dim_val, 1),
            'trending': adx_val > 20,
            'strong_trend': adx_val > 25,
            'bullish_di': dip_val > dim_val,
            'rising': rising
        }
    except:
        return {
            'adx': 0, 'di_plus': 0, 'di_minus': 0,
            'trending': False, 'strong_trend': False,
            'bullish_di': False, 'rising': False
        }


def calculate_roc(prices, period=5):
    """Rate of Change — measures price velocity over N bars"""
    if len(prices) < period + 1:
        return 0.0
    return float((prices.iloc[-1] / prices.iloc[-(period + 1)] - 1) * 100)


def analyze_52week(hist):
    """
    52-week high proximity — highest-alpha breakout signal.
    Stocks breaking above the 52-week high have zero overhead supply.
    """
    try:
        lookback = min(len(hist), 252)
        high_52w = float(hist['High'].iloc[-lookback:].max())
        current  = float(hist['Close'].iloc[-1])
        pct_from_high = ((current - high_52w) / high_52w) * 100
        return {
            'high_52w':    round(high_52w, 2),
            'pct_from_high': round(pct_from_high, 2),
            'at_52w_high':  pct_from_high >= -1.0,   # Within 1%: breakout territory
            'near_52w_high': pct_from_high >= -5.0,  # Within 5%: approaching resistance
        }
    except:
        return {
            'high_52w': 0, 'pct_from_high': -100,
            'at_52w_high': False, 'near_52w_high': False
        }


def calculate_cmf(hist, period=21):
    """
    Chaikin Money Flow — best volume indicator for breakout confirmation.
    CMF > +0.2 = strong institutional buying into resistance.
    CMF < -0.2 = distribution — avoid longs.
    """
    try:
        high  = hist['High']
        low   = hist['Low']
        close = hist['Close']
        vol   = hist['Volume']
        denom   = (high - low).replace(0, np.nan)
        mf_mult = ((close - low) - (high - close)) / denom
        mf_vol  = mf_mult * vol
        vol_sum = vol.rolling(period).sum().replace(0, np.nan)
        cmf = mf_vol.rolling(period).sum() / vol_sum
        val = float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0.0
        return round(val, 3)
    except:
        return 0.0


def calculate_mfi(hist, period=14):
    """
    Money Flow Index — volume-weighted RSI.
    MFI > 60 on breakout = strong institutional participation.
    MFI < 40 = weak money flow, skip the trade.
    """
    try:
        tp  = (hist['High'] + hist['Low'] + hist['Close']) / 3
        mf  = tp * hist['Volume']
        pos = mf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
        neg = mf.where(tp < tp.shift(1), 0.0).rolling(period).sum()
        mfr = pos / neg.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfr))
        val = float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0
        return round(val, 1)
    except:
        return 50.0


def calculate_obv_slope(hist):
    """
    On-Balance Volume slope — rising OBV = accumulation, falling = distribution.
    Normalized slope over 5 bars so it's comparable across tickers.
    """
    try:
        close = hist['Close']
        vol   = hist['Volume']
        obv   = (np.sign(close.diff()).fillna(0) * vol).cumsum()
        if len(obv) < 6:
            return {'rising': False, 'slope': 0.0}
        baseline = float(obv.abs().mean()) or 1.0
        slope    = float(obv.iloc[-1] - obv.iloc[-6]) / baseline
        return {'rising': slope > 0, 'slope': round(slope, 4)}
    except:
        return {'rising': False, 'slope': 0.0}


def detect_ttm_squeeze(hist):
    """
    TTM Squeeze (John Carter) — Bollinger Bands fully inside Keltner Channels.
    squeeze_on   = market coiling, energy building (red dots on TTM)
    fired_today  = squeeze just released = breakout imminent (green dot transition)
    squeeze_bars = how many days it was coiling (longer = more explosive)

    Standard params: BB(20, 2.0 std) · KC(20 EMA, 1.5 ATR)
    """
    try:
        close = hist['Close']
        high  = hist['High']
        low   = hist['Low']

        # Bollinger Bands
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std

        # Keltner Channels
        kc_mid   = close.ewm(span=20, adjust=False).mean()
        tr       = pd.concat([high - low,
                               (high - close.shift()).abs(),
                               (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr20    = tr.ewm(span=20, adjust=False).mean()
        kc_upper = kc_mid + 1.5 * atr20
        kc_lower = kc_mid - 1.5 * atr20

        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        in_squeeze_now = bool(squeeze.iloc[-1])
        was_in_squeeze = bool(squeeze.iloc[-2]) if len(squeeze) >= 2 else False
        fired_today    = was_in_squeeze and not in_squeeze_now

        # Count consecutive squeeze bars leading up to today
        squeeze_bars = 0
        for v in reversed(squeeze.iloc[:-1].tolist()):
            if v:
                squeeze_bars += 1
            else:
                break

        # Momentum direction on fire: close vs close 2 bars ago
        price_delta = float(close.iloc[-1] - close.iloc[-2]) if len(close) >= 2 else 0

        return {
            'in_squeeze':   in_squeeze_now,
            'fired_today':  fired_today,
            'squeeze_bars': squeeze_bars,
            'bullish_fire': fired_today and price_delta > 0,
        }
    except:
        return {'in_squeeze': False, 'fired_today': False,
                'squeeze_bars': 0, 'bullish_fire': False}


def detect_candlestick_patterns(hist):
    """
    Key candlestick patterns for breakout/momentum confirmation:

    MARUBOZU      — Body ≥80% of range, tiny wicks: pure conviction bar
    ENGULFING     — Bullish body fully engulfs prior bearish body on higher volume
    INSIDE_BAR    — Today's range inside yesterday's: coiling before the break
    IB_BREAKOUT   — Prior bar was inside, today breaks above the mother bar high
    3_BAR_PLAY    — Strong bar → controlled pullback → continuation above bar-1 high
    HIGH_CLOSE    — Close in top 25% of range: bulls controlled the whole session
    """
    try:
        if len(hist) < 4:
            return _empty_candle_result()

        o = hist['Open'].values
        h = hist['High'].values
        l = hist['Low'].values
        c = hist['Close'].values
        v = hist['Volume'].values

        o0, h0, l0, c0 = o[-1], h[-1], l[-1], c[-1]   # today
        o1, h1, l1, c1 = o[-2], h[-2], l[-2], c[-2]   # yesterday
        o2, h2, l2, c2 = o[-3], h[-3], l[-3], c[-3]   # 2 days ago

        rng0  = h0 - l0
        rng1  = h1 - l1
        body0 = abs(c0 - o0)
        body1 = abs(c1 - o1)

        # ── MARUBOZU ─────────────────────────────────────────────────────────
        is_marubozu = (rng0 > 0 and body0 / rng0 >= 0.80 and c0 > o0)

        # ── BULLISH ENGULFING ─────────────────────────────────────────────────
        is_engulfing = (
            c1 < o1 and          # prior bar bearish
            c0 > o0 and          # today bullish
            c0 > o1 and          # today's close above prior open
            o0 < c1 and          # today's open below prior close
            v[-1] > v[-2] * 1.2  # volume surge confirms
        )

        # ── INSIDE BAR ───────────────────────────────────────────────────────
        is_inside_bar = (h0 < h1 and l0 > l1)

        # ── INSIDE BAR BREAKOUT ───────────────────────────────────────────────
        # Yesterday was an inside bar; today breaks above the mother bar (2 days ago)
        prior_was_inside = (h1 < h2 and l1 > l2)
        ib_breakout      = prior_was_inside and c0 > h2

        # ── 3-BAR PLAY ────────────────────────────────────────────────────────
        # Bar-2: strong bullish move
        # Bar-1: orderly pullback that stays in the upper half of bar-2
        # Bar-0: closes above bar-2's close (continuation)
        bar2_bullish  = c2 > o2
        bar2_strong   = (rng1 > 0) and abs(c2 - o2) / (h2 - l2 + 1e-9) > 0.55
        bar1_pullback = (c1 < c2) and (l1 > l2 + (h2 - l2) * 0.25)
        bar0_breaks   = c0 > c2
        is_three_bar  = bar2_bullish and bar2_strong and bar1_pullback and bar0_breaks

        # ── HIGH CLOSE (conviction bar) ───────────────────────────────────────
        high_close = (rng0 > 0 and (c0 - l0) / rng0 >= 0.75)

        pattern = 'NONE'
        if is_three_bar:    pattern = '3_BAR_PLAY'
        elif is_engulfing:  pattern = 'ENGULFING'
        elif ib_breakout:   pattern = 'IB_BREAKOUT'
        elif is_marubozu:   pattern = 'MARUBOZU'
        elif is_inside_bar: pattern = 'INSIDE_BAR'

        return {
            'pattern':        pattern,
            'marubozu':       is_marubozu,
            'engulfing':      is_engulfing,
            'inside_bar':     is_inside_bar,
            'ib_breakout':    ib_breakout,
            'three_bar_play': is_three_bar,
            'high_close':     high_close,
            'body_pct':       round(body0 / rng0 * 100, 1) if rng0 > 0 else 0,
        }
    except:
        return _empty_candle_result()


def _empty_candle_result():
    return {
        'pattern': 'NONE', 'marubozu': False, 'engulfing': False,
        'inside_bar': False, 'ib_breakout': False, 'three_bar_play': False,
        'high_close': False, 'body_pct': 0,
    }


# ==================== CLAUDE AI ANALYSIS ====================

# ── Backtest insights (loaded once, used for pre-filtering + signal analysis) ──

_BACKTEST_INSIGHTS    = None   # dict loaded from backtest_insights.json
_SIGNAL_SYSTEM_PROMPT = None   # pre-built, cached across all signal analyses

_INSIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_insights.json')

def load_backtest_insights():
    """Load Claude's saved backtest analysis. Called at startup and after /api/analyze-backtest."""
    global _BACKTEST_INSIGHTS, _SIGNAL_SYSTEM_PROMPT
    if os.path.exists(_INSIGHTS_FILE):
        with open(_INSIGHTS_FILE) as f:
            _BACKTEST_INSIGHTS = json.load(f)
        _SIGNAL_SYSTEM_PROMPT = _build_signal_system_prompt(_BACKTEST_INSIGHTS)
    return _BACKTEST_INSIGHTS

def _build_signal_system_prompt(ins: dict) -> str:
    """
    Build the system prompt used for every per-signal analysis.
    This is sent with cache_control so it's cached across all signals in a scan.
    """
    lines = [
        "You are a momentum options trader with access to historical signal performance data.",
        "",
        "BACKTEST FINDINGS (90 days, 25 real signals):",
        "  Win rates by pattern:   VCP_BREAKOUT 67% (+5.31 avg) | TREND_CONTINUATION 50% (+4.28 avg) | GAP_AND_GO 33% (+1.08 avg)",
        "  Win rates by ADX:       ADX 20-25 → 62% | ADX 25-35 → 57% | ADX >35 → 11% (overextended — worst bucket)",
        "  Winners vs losers:      Winners avg volume 3.62x (losers 2.59x) | Volume confirmation key",
        "",
    ]

    rules = ins.get('filter_rules', [])
    if rules:
        lines.append("PROVEN FILTER RULES:")
        for rule in rules[:6]:
            lines.append(f"  • {rule}")
        lines.append("")

    guide = ins.get('prefilter_scoring_guide', '')
    if guide:
        lines.append(f"SCORING GUIDE: {guide}")
        lines.append("")

    prob_model = ins.get('win_probability_model', '')
    if prob_model:
        lines.append(f"WIN PROBABILITY MODEL: {prob_model}")

    return "\n".join(lines)

# Load insights at import time (no-op if file doesn't exist yet)
load_backtest_insights()


# ── Gemini model config ────────────────────────────────────────────────────────
# gemini-2.5-flash → per-signal analysis + Discord commentary
_GEMINI_ANALYSIS_MODEL  = 'gemini-2.5-flash'

# JSON mode config shared by both functions
_GEMINI_JSON_CONFIG = genai.GenerationConfig(
    response_mime_type='application/json',
    temperature=0.1,
)


def rule_prefilter_candidates(candidates):
    """
    Score ALL setup candidates with a deterministic rule-based function —
    no API call required.  Uses already-computed indicator values from each
    setup dict.  Returns list of (ticker, setup, score, reason) sorted by
    score descending (same shape as the old Gemini prefilter).
    """
    PATTERN_SCORES = {
        'VCP_BREAKOUT':            55,
        'CONSOLIDATION_BREAKOUT':  50,
        'BULL_FLAG':               45,
        'ASCENDING_TRIANGLE':      40,
        'CUP_AND_HANDLE':          40,
        'GAP_AND_GO':              30,
    }

    CANDLE_BONUS = {
        '3_BAR_PLAY':   10,
        'IB_BREAKOUT':   8,
        'ENGULFING':     7,
        'MARUBOZU':      5,
    }

    result = []
    for ticker, setup in candidates:
        score  = 0
        reasons = []

        chart   = setup.get('chart_analysis', {})
        adx_d   = setup.get('adx', {})
        squeeze = setup.get('squeeze', {})
        w52     = setup.get('week52', {})

        # Pattern type (base score)
        pattern = chart.get('pattern_type', '')
        base    = PATTERN_SCORES.get(pattern, 25)
        score  += base
        reasons.append(f'pat={pattern}({base})')

        # Squeeze state
        if squeeze.get('fired_today'):
            score += 25; reasons.append('sqz_fire+25')
        elif squeeze.get('in_squeeze'):
            score += 15; reasons.append('sqz_on+15')

        # CMF
        cmf = float(setup.get('cmf', 0))
        if cmf >= 0.20:
            score += 20; reasons.append('cmf_strong+20')
        elif cmf >= 0.10:
            score += 15; reasons.append('cmf_pos+15')
        elif cmf <= -0.10:
            score -= 15; reasons.append('cmf_neg-15')

        # Volume ratio
        vr = float(setup.get('volume_ratio', 1))
        if vr >= 2.5:
            score += 15; reasons.append('vol_surge+15')
        elif vr >= 1.8:
            score += 10; reasons.append('vol_high+10')

        # ADX sweet spot 20-30
        adx_val = float(adx_d.get('adx', 0))
        if 20 <= adx_val <= 30:
            score += 10; reasons.append('adx_ideal+10')
        elif adx_val > 40:
            score -= 5;  reasons.append('adx_hot-5')

        # 52-week high proximity
        if w52.get('at_52w_high'):
            score += 10; reasons.append('at52w+10')
        elif w52.get('near_52w_high'):
            score += 5;  reasons.append('near52w+5')

        # MACD expanding
        if setup.get('macd_expanding'):
            score += 5; reasons.append('macd_exp+5')

        # Candle pattern bonus
        candle_pat = setup.get('candle', {}).get('pattern', '')
        cb = CANDLE_BONUS.get(candle_pat, 0)
        if cb:
            score += cb; reasons.append(f'candle_{candle_pat}+{cb}')

        # Rate of change
        roc5 = float(setup.get('roc_5', 0))
        if roc5 >= 2:
            score += 5; reasons.append('roc5+5')

        score = max(0, min(100, score))
        result.append((ticker, setup, score, ' '.join(reasons)))

    result.sort(key=lambda x: x[2], reverse=True)
    return result


def get_gemini_analysis(ticker, setup, option, scoring):
    """
    Ask Gemini to assess signal quality and write Discord commentary.
    Uses gemini-2.5-pro for the highest-quality per-signal analysis.
    Returns a dict with quality, play, risk, conviction, win_probability,
    key_risks, summary — identical shape to the old Claude response.
    """
    if not GEMINI_API_KEY:
        return None
    try:
        w52      = setup.get('week52', {})
        adx      = setup.get('adx', {})
        chart    = setup['chart_analysis']
        opt_type = 'CALL' if setup['direction'] == 'BULLISH' else 'PUT'

        system_text = _SIGNAL_SYSTEM_PROMPT or (
            "You are a momentum options trader. Analyze signals rigorously and estimate "
            "win probability based on historical pattern performance."
        )

        user = f"""Analyze this scanner signal and return a JSON assessment.

TICKER: {ticker} | DIRECTION: {setup['direction']} | COMPOSITE SCORE: {scoring['composite']}/100

PRICE ACTION:
- Move today: {setup['price_move_pct']:+.1f}% | Volume: {setup['volume_ratio']}x avg
- Price: ${setup['current_price']} | 52W high: ${w52.get('high_52w', 0)} ({w52.get('pct_from_high', 0):+.1f}% away)
- RSI: {setup['rsi']} | MACD: {'bullish' if setup['macd_bullish'] else 'bearish'}{', expanding' if setup['macd_expanding'] else ''}
- ADX: {adx.get('adx', 0):.1f} ({'strong' if adx.get('strong_trend') else 'trending' if adx.get('trending') else 'weak'})
- ROC-5: {setup.get('roc_5', 0):+.1f}% | CMF: {setup.get('cmf', 0):+.2f} | MFI: {setup.get('mfi', 50):.0f}
- Squeeze: {'FIRED ' + str(setup.get('squeeze',{}).get('squeeze_bars',0)) + 'd coil' if setup.get('squeeze',{}).get('fired_today') else 'ON ' + str(setup.get('squeeze',{}).get('squeeze_bars',0)) + 'd' if setup.get('squeeze',{}).get('in_squeeze') else 'OFF'}
- Candle: {setup.get('candle',{}).get('pattern','NONE')} | High close: {setup.get('candle',{}).get('high_close', False)} | OBV rising: {setup.get('obv',{}).get('rising', False)}

CHART PATTERN:
- {chart['pattern_type']} (quality: {chart['quality_score']}/100)
- VCP: {chart.get('vcp_pattern')} | Tight closes: {chart.get('tight_closes')} | Vol dry-up: {chart.get('vol_drying_up')}
- Warnings: {', '.join(chart['warnings']) if chart['warnings'] else 'none'}

OPTIONS CONTRACT:
- {opt_type} ${option['strike']} exp {option['expiration']} ({option['dte']}D)
- Premium: ${option['premium']} | Delta: {option['delta']} | OI: {option['open_interest']:,}
- Entry: ${setup['entry']} | Stop: ${setup['stop_loss']} (-{setup['risk_pct']}%) | Target: ${setup['target_price']}

Return JSON with exactly these fields:
- quality: "HIGH", "MEDIUM", or "LOW"
- play: "CALL", "PUT", or "STRADDLE"
- risk: "LOW", "MODERATE", or "HIGH"
- conviction: integer 1-10
- win_probability: integer 0-100 (estimated % chance of 2%+ underlying move in 5 days)
- key_risks: array of 1-3 specific risk strings for THIS setup
- summary: 1-2 sentence Discord alert commentary (key bullish reason + main risk)"""

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=_GEMINI_ANALYSIS_MODEL,
            system_instruction=system_text,
            generation_config=_GEMINI_JSON_CONFIG,
        )
        response = model.generate_content(user)
        result = json.loads(response.text)
        # Normalise types so the rest of the app doesn't break
        result['conviction']      = int(result.get('conviction', 5))
        result['win_probability'] = int(result.get('win_probability', 50))
        result['key_risks']       = result.get('key_risks', [])
        return result
    except Exception as e:
        print(f'Gemini analysis failed for {ticker}: {e}')
    return None


# ==================== DISCORD NOTIFICATIONS ====================

def send_discord_alerts(results):
    """Send one rich embed per signal to the configured Discord webhook."""
    if not DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL == 'YOUR_WEBHOOK_URL_HERE':
        return False
    if not results:
        return False

    try:
        embeds = []
        for pick in results:
            s     = pick['setup']
            o     = pick['option']
            sc    = pick['scoring']
            chart = s['chart_analysis']
            w52   = s.get('week52', {})
            adx   = s.get('adx', {})

            color = 0x00e676 if s['direction'] == 'BULLISH' else 0xff5252

            # Signal badges
            sq     = s.get('squeeze', {})
            candle = s.get('candle', {})
            badges = []
            if w52.get('at_52w_high'):            badges.append('🏆 52W HIGH')
            elif w52.get('near_52w_high'):         badges.append('📍 NEAR 52W')
            if sq.get('bullish_fire'):             badges.append(f"🔥 SQUEEZE FIRED ({sq.get('squeeze_bars',0)}d coil)")
            elif sq.get('in_squeeze'):             badges.append(f"🟡 SQUEEZE ON ({sq.get('squeeze_bars',0)}d)")
            cp = candle.get('pattern', 'NONE')
            if   cp == '3_BAR_PLAY':              badges.append('🎯 3-BAR PLAY')
            elif cp == 'IB_BREAKOUT':             badges.append('📐 IB BREAKOUT')
            elif cp == 'ENGULFING':               badges.append('🕯️ ENGULFING')
            elif cp == 'MARUBOZU':                badges.append('💥 MARUBOZU')
            if candle.get('high_close'):           badges.append('✅ HIGH CLOSE')
            if chart.get('vcp_pattern'):           badges.append('📊 VCP')
            if s.get('momentum_accelerating'):     badges.append('⚡ ACCEL')
            if s.get('macd_expanding'):            badges.append('📈 MACD↑')
            if adx.get('strong_trend'):            badges.append('💪 ADX STRONG')

            option_type = 'CALL' if s['direction'] == 'BULLISH' else 'PUT'
            icon = '🟢' if s['direction'] == 'BULLISH' else '🔴'

            # AI analysis fields (optional — only shown if Claude API key is set)
            ai = pick.get('ai_analysis')
            ai_fields = []
            if ai:
                conviction_stars = '★' * ai['conviction'] + '☆' * (10 - ai['conviction'])
                quality_emoji = {'HIGH': '🔥', 'MEDIUM': '⚡', 'LOW': '❄️'}.get(ai['quality'], '')
                risk_emoji    = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🔴'}.get(ai['risk'], '')
                win_pct       = ai.get('win_probability', 0)
                # Win probability bar (10 blocks = 100%)
                filled = round(win_pct / 10)
                prob_bar = '█' * filled + '░' * (10 - filled)
                # Key risks (new field)
                risks = ai.get('key_risks', [])
                risks_str = '  •  '.join(risks) if risks else '—'
                # Pre-filter score if available
                pf = pick.get('prefilter')
                pf_str = f"  |  Pre-filter: **{pf['score']}/100**" if pf else ''

                ai_fields = [
                    {
                        'name': f'{quality_emoji} AI Analysis  |  Conviction: {ai["conviction"]}/10{pf_str}',
                        'value': (
                            f"**Quality:** {ai['quality']}  |  **Play:** {ai['play']}  |  **Risk:** {risk_emoji} {ai['risk']}\n"
                            f"**Win Probability:** {win_pct}%  `{prob_bar}`\n"
                            f"{conviction_stars}\n"
                            f"_{ai['summary']}_"
                        ),
                        'inline': False
                    },
                    {
                        'name': '⚠️ Key Risks',
                        'value': risks_str,
                        'inline': False
                    }
                ]

            embed = {
                'title': f"{icon} {pick['ticker']} — {s['direction']}  |  Score: {sc['composite']}",
                'color': color,
                'fields': [
                    {
                        'name': 'Signals',
                        'value': '  '.join(badges) if badges else '—',
                        'inline': False
                    },
                    {'name': 'Move',    'value': f"{'+' if s['price_move_pct'] > 0 else ''}{s['price_move_pct']}%", 'inline': True},
                    {'name': 'Volume',  'value': f"{s['volume_ratio']}x avg",  'inline': True},
                    {'name': 'RSI',     'value': str(s['rsi']),                 'inline': True},
                    {'name': 'Pattern', 'value': chart['pattern_type'].replace('_', ' '), 'inline': True},
                    {'name': 'ADX',     'value': f"{adx.get('adx', '—')}{'↑' if adx.get('rising') else ''}", 'inline': True},
                    {'name': 'CMF',     'value': f"{s.get('cmf', 0):+.2f}", 'inline': True},
                    {'name': 'MFI',     'value': str(s.get('mfi', '—')), 'inline': True},
                    {'name': 'Candle',  'value': candle.get('pattern', 'NONE').replace('_', ' '), 'inline': True},
                    {'name': 'ROC-5',   'value': f"{'+' if s.get('roc_5', 0) > 0 else ''}{s.get('roc_5', '—')}%", 'inline': True},
                    {'name': 'Entry',   'value': f"${s['entry']}",             'inline': True},
                    {'name': 'Stop',    'value': f"${s['stop_loss']} (-{s['risk_pct']}%)", 'inline': True},
                    {'name': 'Target',  'value': f"${s['target_price']}",      'inline': True},
                    {
                        'name': 'Option Contract',
                        'value': f"`{option_type} ${o['strike']} exp {o['expiration']} ({o['dte']}D)`\nPremium: **${o['premium']}**  |  Buy ≤ **${o['ask']}**  |  Delta: {o['delta']}  |  OI: {o['open_interest']:,}",
                        'inline': False
                    },
                    {'name': 'Theta decay', 'value': f"{pick['theta_pct_per_day']}%/day", 'inline': True},
                    {'name': 'Profit target', 'value': f"{pick['expected_return_pct']}%", 'inline': True},
                    *ai_fields,
                ],
                'footer': {'text': 'Momentum Scanner • Powered by Gemini' if ai else 'Momentum Scanner'},
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            embeds.append(embed)

        payload = {
            'username': 'Momentum Scanner',
            'avatar_url': 'https://i.imgur.com/4M34hi2.png',
            'content': f"**{len(results)} signal{'s' if len(results) > 1 else ''} found** — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            'embeds': embeds[:10]  # Discord max 10 embeds per message
        }

        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        return resp.status_code == 204
    except Exception as e:
        print(f"Discord alert failed: {e}")
        return False


# ==================== CHART PATTERN ANALYSIS ====================

def analyze_chart_pattern(ticker, hist):
    try:
        current_price = float(hist['Close'].iloc[-1])

        # 20-bar consolidation window (was 10 — too short for reliable patterns)
        lookback    = hist.tail(21)
        consol_bars = lookback.iloc[:-1]  # Exclude today
        consol_high = float(consol_bars['High'].max())
        consol_low  = float(consol_bars['Low'].min())
        consol_range = ((consol_high - consol_low) / consol_low) * 100

        breaking_high = current_price > consol_high
        breaking_low  = current_price < consol_low

        # VCP (Volatility Contraction Pattern): second half of base tighter than first
        first_half   = consol_bars.iloc[:10]
        second_half  = consol_bars.iloc[10:]
        first_range  = float(first_half['High'].max() - first_half['Low'].min())
        second_range = float(second_half['High'].max() - second_half['Low'].min())
        vcp_pattern  = (first_range > 0 and second_range / first_range < 0.65)

        # Tight closes: last 5 bars compressing (standard deviation < 1.5% of price)
        tight_std   = float(hist['Close'].iloc[-6:-1].std() / current_price * 100)
        tight_closes = tight_std < 1.5

        # Bollinger Bands
        ma20  = float(hist['Close'].rolling(20).mean().iloc[-1])
        sma   = hist['Close'].rolling(20).mean()
        std   = hist['Close'].rolling(20).std()
        outside_upper = current_price > float((sma + std * 2).iloc[-1])
        outside_lower = current_price < float((sma - std * 2).iloc[-1])
        distance_from_ma = ((current_price - ma20) / ma20) * 100

        # Gap detection
        prev_close  = float(hist['Close'].iloc[-2])
        open_price  = float(hist['Open'].iloc[-1])
        gap_pct     = ((open_price - prev_close) / prev_close) * 100
        has_gap     = abs(gap_pct) > 1.0

        last_5 = hist['Close'].tail(5)
        consecutive_up   = all(last_5.diff().dropna() > 0)
        consecutive_down = all(last_5.diff().dropna() < 0)

        # Volume: today vs consolidation average (breakout confirmation)
        current_vol = float(hist['Volume'].iloc[-1])
        consol_avg_vol = float(consol_bars['Volume'].mean())
        vol_surge_vs_consol = current_vol / consol_avg_vol if consol_avg_vol > 0 else 1.0

        # Volume drying up during consolidation = textbook setup
        recent_consol_vol = float(consol_bars['Volume'].iloc[-5:].mean())
        early_consol_vol  = float(consol_bars['Volume'].iloc[:5].mean())
        vol_drying_up = recent_consol_vol < early_consol_vol * 0.85

        volume_increasing = bool(hist['Volume'].iloc[-1] > hist['Volume'].iloc[-2])

        # Pattern classification
        pattern_type  = "UNKNOWN"
        quality_score = 50
        warnings_list = []

        if vcp_pattern and (breaking_high or breaking_low) and tight_closes:
            pattern_type  = "VCP_BREAKOUT"
            quality_score = 95 if vol_surge_vs_consol >= 1.5 else 85
        elif consol_range < 5.0 and (breaking_high or breaking_low):
            pattern_type  = "CONSOLIDATION_BREAKOUT"
            quality_score = 90 if volume_increasing else 80
        elif consecutive_up or consecutive_down:
            pattern_type  = "TREND_CONTINUATION"
            quality_score = 75
            if abs(distance_from_ma) > 8:
                warnings_list.append("Price extended >8% from 20MA")
                quality_score -= 15
        elif has_gap and volume_increasing:
            pattern_type  = "GAP_AND_GO"
            quality_score = 80
            if abs(gap_pct) > 5:
                warnings_list.append("Large gap (>5%) — potential exhaustion")
                quality_score -= 10
        elif breaking_high or breaking_low:
            pattern_type  = "RANGE_BREAKOUT"
            quality_score = 65
            if consol_range > 8:
                warnings_list.append("Wide consolidation range")
                quality_score -= 15
        else:
            pattern_type  = "V_REVERSAL"
            quality_score = 40
            warnings_list.append("V-shaped move — higher reversal risk")

        # Bonuses and penalties
        if vcp_pattern and pattern_type != "VCP_BREAKOUT":
            quality_score += 10
        if vol_surge_vs_consol >= 2.0:
            quality_score += 5
        if outside_upper or outside_lower:
            warnings_list.append("Outside Bollinger Bands")
            quality_score -= 10
        if abs(distance_from_ma) > 10:
            warnings_list.append("EXTREME: >10% from 20MA")
            quality_score -= 20
        if not volume_increasing:
            warnings_list.append("Volume not increasing today")
            quality_score -= 10

        return {
            'pattern_type':         pattern_type,
            'quality_score':        max(0, min(100, quality_score)),
            'warnings':             warnings_list,
            'consolidation_range':  round(consol_range, 2),
            'distance_from_ma20':   round(distance_from_ma, 2),
            'breaking_high':        breaking_high,
            'breaking_low':         breaking_low,
            'outside_bollinger':    outside_upper or outside_lower,
            'has_gap':              has_gap,
            'gap_pct':              round(gap_pct, 2) if has_gap else 0,
            'volume_increasing':    volume_increasing,
            'vol_surge_vs_consol':  round(vol_surge_vs_consol, 2),
            'vcp_pattern':          vcp_pattern,
            'tight_closes':         tight_closes,
            'vol_drying_up':        vol_drying_up,
            'ma20':                 round(ma20, 2)
        }
    except Exception as e:
        return {
            'pattern_type': 'ERROR', 'quality_score': 0, 'warnings': [str(e)[:50]],
            'consolidation_range': 0, 'distance_from_ma20': 0,
            'breaking_high': False, 'breaking_low': False,
            'outside_bollinger': False, 'has_gap': False, 'gap_pct': 0,
            'volume_increasing': False, 'vol_surge_vs_consol': 1.0,
            'vcp_pattern': False, 'tight_closes': False, 'vol_drying_up': False, 'ma20': 0
        }


# ==================== MOMENTUM DETECTION ====================

def detect_momentum_setup(ticker, hist=None):
    try:
        if hist is None:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period='1y', interval='1d')

        if hist.empty or len(hist) < 60:
            return None

        close = hist['Close']

        rsi    = calculate_rsi(close)
        atr    = calculate_atr(hist)
        macd   = calculate_macd(close)
        adx    = calculate_adx(hist)
        roc_5  = calculate_roc(close, 5)
        roc_10 = calculate_roc(close, 10)
        week52 = analyze_52week(hist)
        cmf    = calculate_cmf(hist)
        mfi    = calculate_mfi(hist)
        obv    = calculate_obv_slope(hist)
        squeeze = detect_ttm_squeeze(hist)
        candle  = detect_candlestick_patterns(hist)

        current_price   = float(close.iloc[-1])
        prev_close      = float(close.iloc[-2])
        current_volume  = float(hist['Volume'].iloc[-1])
        # FIX: use iloc[-21:-1] so today's bar doesn't inflate the average
        avg_volume_20   = float(hist['Volume'].iloc[-21:-1].mean())
        avg_volume_50   = float(hist['Volume'].iloc[-51:-1].mean()) if len(hist) >= 51 else avg_volume_20

        price_move_pct  = ((current_price - prev_close) / prev_close) * 100
        volume_ratio    = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
        volume_ratio_50 = current_volume / avg_volume_50 if avg_volume_50 > 0 else 0

        # Determine direction early (needed for direction-aware filters below)
        direction = 'BULLISH' if price_move_pct > 0 else 'BEARISH'

        # Skip bearish setups until scanner is calibrated for short side
        if CONFIG.get('bullish_only') and direction == 'BEARISH':
            return None

        # ── CORE FILTERS ────────────────────────────────────────────────────
        if abs(price_move_pct) < CONFIG['min_price_move']:
            return None
        if volume_ratio < CONFIG['min_volume_ratio']:
            return None
        if rsi < CONFIG['rsi_min'] or rsi > CONFIG['rsi_max']:
            return None
        if pd.isna(atr):
            return None

        # Reject large gaps (>8%) — almost always earnings/binary events
        open_price = float(hist['Open'].iloc[-1])
        gap_pct_raw = abs((open_price - prev_close) / prev_close) * 100
        if gap_pct_raw > CONFIG['max_gap_pct']:
            return None

        # ADX: only trade trending stocks; cap at max_adx (ADX>35 = 11% WR per backtest)
        if adx['adx'] < CONFIG['min_adx']:
            return None
        max_adx = CONFIG.get('max_adx')
        if max_adx is not None and adx['adx'] > max_adx:
            return None

        # MACD must confirm the direction of the trade
        if CONFIG['require_macd_aligned']:
            if direction == 'BULLISH' and not macd['bullish']:
                return None
            if direction == 'BEARISH' and macd['bullish']:
                return None

        # Price must be above key MAs for bullish trades (no buying in downtrends)
        if len(close) >= 50:
            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma50 = float(close.rolling(50).mean().iloc[-1])
            if direction == 'BULLISH' and current_price < ma20:
                return None
            if direction == 'BULLISH' and current_price < ma50:
                return None

        chart = analyze_chart_pattern(ticker, hist)
        if chart['quality_score'] < CONFIG['min_pattern_quality']:
            return None
        # GAP_AND_GO with pat_q>=75 = 0/3 WR — over-scoring on this pattern = noise (LHX,TRS,PSTG)
        max_gap_pq = CONFIG.get('max_gap_and_go_quality')
        if max_gap_pq is not None and chart['pattern_type'] == 'GAP_AND_GO' and chart['quality_score'] > max_gap_pq:
            return None
        if CONFIG['skip_v_reversals'] and chart['pattern_type'] == 'V_REVERSAL':
            return None
        # TREND_CONTINUATION = 0% WR (0/2) — HNRG, FIVE both failed to produce sufficient move
        if chart['pattern_type'] == 'TREND_CONTINUATION':
            return None
        # GAP_AND_GO requires stronger volume confirmation (backtest: vol<2.2 = pure losers)
        if chart['pattern_type'] == 'GAP_AND_GO' and volume_ratio < 2.2:
            return None
        # RANGE_BREAKOUT + vol<2.0 = only loser profile (RDNT vol=1.49, -1.32%)
        if chart['pattern_type'] == 'RANGE_BREAKOUT' and volume_ratio < 2.0:
            return None

        # High-close filter: close must be in top 25% of today's range for bullish
        # Signals bulls controlled the whole session — low-close breakouts are weak
        today_range = float(hist['High'].iloc[-1] - hist['Low'].iloc[-1])
        if today_range > 0 and direction == 'BULLISH':
            close_position = (float(close.iloc[-1]) - float(hist['Low'].iloc[-1])) / today_range
            if close_position < 0.50:   # below midpoint = rejection bar, skip
                return None

        # CMF confirmation: avoid buying when money is flowing out
        if direction == 'BULLISH' and cmf < -0.15:
            return None

        # Extreme volume (>4x) = 0/2 WR — institutional dump, not accumulation (PSKY,MP)
        if volume_ratio > 4.0:
            return None
        # High vol + not near 52W = distribution signal, not breakout (PSTG 3.34x = -25.94%)
        if volume_ratio > 3.0 and not week52.get('near_52w_high'):
            return None
        stop_distance = atr * CONFIG['atr_stop_multiplier']

        if direction == 'BULLISH':
            stop_loss    = current_price - stop_distance
            target_price = current_price + (stop_distance * 2)
        else:
            stop_loss    = current_price + stop_distance
            target_price = current_price - (stop_distance * 2)

        return {
            'type':          'MOMENTUM',
            'direction':     direction,
            'price_move_pct': round(price_move_pct, 2),
            'current_price': round(current_price, 2),
            'entry':         round(current_price, 2),
            'stop_loss':     round(stop_loss, 2),
            'target_price':  round(target_price, 2),
            'volume_ratio':  round(volume_ratio, 2),
            'volume_ratio_50': round(volume_ratio_50, 2),
            'rsi':           round(rsi, 1),
            'atr':           round(float(atr), 2),
            'macd_bullish':  macd['bullish'],
            'macd_expanding': macd['expanding'],
            'adx':           adx,
            'roc_5':         round(roc_5, 2),
            'roc_10':        round(roc_10, 2),
            'momentum_accelerating': bool(roc_5 > roc_10 > 0 or roc_5 < roc_10 < 0),
            'week52':        week52,
            'cmf':           cmf,
            'mfi':           mfi,
            'obv':           obv,
            'squeeze':       squeeze,
            'candle':        candle,
            'risk_pct':      round(abs((current_price - stop_loss) / current_price) * 100, 1),
            'chart_analysis': chart
        }
    except:
        return None


# ==================== OPTIONS CHAIN ====================

def get_best_option(ticker, setup):
    try:
        stock       = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None

        viable_options = []
        today = datetime.now()

        for exp_date in expirations[:10]:
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            dte = (exp_datetime - today).days
            if dte < CONFIG['dte_range'][0] or dte > CONFIG['dte_range'][1]:
                continue

            opt_chain     = stock.option_chain(exp_date)
            options       = opt_chain.calls if setup['direction'] == 'BULLISH' else opt_chain.puts
            current_price = setup['current_price']

            options = options[
                (options['strike'] >= current_price * 0.92) &
                (options['strike'] <= current_price * 1.10)
            ]

            for _, row in options.iterrows():
                if row['bid'] == 0 or row['ask'] == 0:
                    continue
                mid_price  = (row['bid'] + row['ask']) / 2
                spread_pct = (row['ask'] - row['bid']) / mid_price
                if spread_pct > CONFIG['max_bid_ask_spread'] or mid_price < CONFIG['min_premium']:
                    continue
                if row['openInterest'] < CONFIG['min_open_interest']:
                    continue

                moneyness = current_price / row['strike']
                if setup['direction'] == 'BULLISH':
                    delta = 0.5 + (moneyness - 1) * 2
                else:
                    delta = 0.5 - (moneyness - 1) * 2
                delta = max(0.1, min(0.9, delta))
                if delta < CONFIG['delta_range'][0] or delta > CONFIG['delta_range'][1]:
                    continue

                theta = -mid_price / max(dte, 1)
                if theta < CONFIG['max_theta_per_day']:
                    continue

                viable_options.append({
                    'strike':        float(row['strike']),
                    'expiration':    exp_date,
                    'dte':           dte,
                    'bid':           round(float(row['bid']), 2),
                    'ask':           round(float(row['ask']), 2),
                    'premium':       round(mid_price, 2),
                    'delta':         round(delta, 3),
                    'theta':         round(theta, 3),
                    'open_interest': int(row['openInterest']),
                    'volume':        int(row['volume']) if row['volume'] > 0 else 0,
                    'spread_pct':    round(spread_pct * 100, 1)
                })

        if not viable_options:
            return None
        viable_options.sort(key=lambda x: x['delta'] - abs(x['theta']) * 0.1, reverse=True)
        return viable_options[0]
    except:
        return None


# ==================== SCORING ====================

def calculate_setup_score(setup, option):
    # Price move: bigger moves → higher score
    move_score = min(100, abs(setup['price_move_pct']) * 25)

    # Volume: use the stronger of 20d vs 50d ratio
    volume_score = min(100, max(setup['volume_ratio'], setup.get('volume_ratio_50', 0)) * 50)

    # RSI: sweet spot 58-70 for bullish momentum entries
    rsi_distance = abs(setup['rsi'] - 64)
    rsi_score = max(40, 100 - (rsi_distance * 2.5))

    # MACD: aligned direction + expanding histogram = +10 bonus
    macd_aligned   = setup['macd_bullish'] == (setup['direction'] == 'BULLISH')
    macd_base      = 100 if macd_aligned else 50
    macd_score     = min(100, macd_base + (10 if setup.get('macd_expanding') else 0))

    # Risk/Reward
    entry_stop_dist = max(abs(setup['entry'] - setup['stop_loss']), 0.01)
    rr_ratio        = abs(setup['target_price'] - setup['entry']) / entry_stop_dist
    rr_score        = min(100, rr_ratio * 50)

    # Options liquidity
    liquidity_score = 100
    if option['open_interest'] < 200: liquidity_score -= 20
    if option['spread_pct'] > 10:     liquidity_score -= 20
    if option['volume'] < 50:         liquidity_score -= 10

    # Chart pattern
    chart_quality = setup['chart_analysis']['quality_score']

    # 52-week high — highest conviction signal for breakout momentum
    w52 = setup.get('week52', {})
    if w52.get('at_52w_high'):
        high52_score = 100
    elif w52.get('near_52w_high'):
        high52_score = 75
    else:
        pct = w52.get('pct_from_high', -100)
        high52_score = max(0, min(60, 60 + pct * 3))  # Scales down the further from high

    # ADX trend strength
    adx_data  = setup.get('adx', {})
    bullish_di = adx_data.get('bullish_di', False) == (setup['direction'] == 'BULLISH')
    if adx_data.get('strong_trend') and bullish_di:
        adx_score = 100
    elif adx_data.get('trending') and bullish_di:
        adx_score = 70
    elif adx_data.get('trending'):
        adx_score = 50
    else:
        adx_score = 30
    if adx_data.get('rising'):
        adx_score = min(100, adx_score + 10)

    # ── CMF / MFI — institutional money flow confirmation ──────────────────
    cmf_val = setup.get('cmf', 0)
    mfi_val = setup.get('mfi', 50)
    if cmf_val >= 0.20:    cmf_score = 100
    elif cmf_val >= 0.10:  cmf_score = 80
    elif cmf_val >= 0.0:   cmf_score = 60
    elif cmf_val >= -0.10: cmf_score = 35
    else:                  cmf_score = 10
    # MFI bonus: > 60 = strong participation
    mfi_bonus = 10 if mfi_val > 60 else (5 if mfi_val > 50 else 0)
    cmf_score = min(100, cmf_score + mfi_bonus)

    # ── TTM Squeeze ─────────────────────────────────────────────────────────
    sq = setup.get('squeeze', {})
    if sq.get('bullish_fire'):
        # Squeeze fired bullish today — highest conviction signal
        squeeze_score = 100
    elif sq.get('fired_today'):
        squeeze_score = 75
    elif sq.get('in_squeeze'):
        # Still coiling: longer squeeze = more explosive potential
        squeeze_score = min(85, 50 + sq.get('squeeze_bars', 0) * 5)
    else:
        squeeze_score = 30

    # ── Candlestick pattern ──────────────────────────────────────────────────
    candle = setup.get('candle', {})
    cp = candle.get('pattern', 'NONE')
    if   cp == '3_BAR_PLAY':  candle_score = 100
    elif cp == 'IB_BREAKOUT': candle_score = 95
    elif cp == 'ENGULFING':   candle_score = 90
    elif cp == 'MARUBOZU':    candle_score = 80
    elif cp == 'INSIDE_BAR':  candle_score = 50   # coiling, not yet breaking
    else:                      candle_score = 40
    if candle.get('high_close'): candle_score = min(100, candle_score + 10)

    # ── OBV confirmation ─────────────────────────────────────────────────────
    obv_rising = setup.get('obv', {}).get('rising', False)

    # ── Bonus points ─────────────────────────────────────────────────────────
    vcp_bonus   =  5 if setup['chart_analysis'].get('vcp_pattern') else 0
    accel_bonus =  5 if setup.get('momentum_accelerating') else 0
    obv_bonus   =  3 if obv_rising else 0

    composite = (
        move_score      * 0.15 +
        volume_score    * 0.12 +
        rsi_score       * 0.08 +
        macd_score      * 0.08 +
        rr_score        * 0.06 +
        liquidity_score * 0.06 +
        chart_quality   * 0.10 +
        high52_score    * 0.10 +
        adx_score       * 0.08 +
        cmf_score       * 0.09 +
        squeeze_score   * 0.08 +
        vcp_bonus + accel_bonus + obv_bonus
    )

    return {
        'composite': round(composite, 1),
        'breakdown': {
            'move':         round(move_score, 1),
            'volume':       round(volume_score, 1),
            'rsi':          round(rsi_score, 1),
            'macd':         round(macd_score, 1),
            'risk_reward':  round(rr_score, 1),
            'liquidity':    round(liquidity_score, 1),
            'chart_pattern': round(chart_quality, 1),
            '52w_high':     round(high52_score, 1),
            'adx':          round(adx_score, 1),
            'cmf_mfi':      round(cmf_score, 1),
            'squeeze':      round(squeeze_score, 1),
            'candle':       round(candle_score, 1),
        }
    }


# ==================== LOAD TICKERS FROM CSV ====================

def load_tickers_from_csv():
    # Always scan from nasdaq_screener.csv only
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nasdaq_screener.csv')
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path)

    # nasdaq_screener.csv has full price/volume data — apply liquid-stock filters
    df['Price'] = df['Last Sale'].replace(r'[\$,]', '', regex=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Symbol', 'Price', 'Volume', 'Market Cap'])
    df = df[
        (df['Price'] >= CONFIG['min_stock_price']) &
        (df['Price'] <= CONFIG['max_stock_price']) &
        (df['Volume'] >= CONFIG['min_volume']) &
        (df['Market Cap'] >= CONFIG['min_market_cap'])
    ]
    return df['Symbol'].tolist()


# ==================== PARALLEL SCAN HELPERS ====================

def fetch_ticker_history(ticker):
    """
    Download 1-year daily OHLCV for one ticker (yfinance).
    If Webull is ready and the market is open, replace the last bar with a
    live intraday aggregate so detect_momentum_setup() sees today's real price.
    """
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period='1y', interval='1d')
        if hist.empty:
            return ticker, None
        # Live bar enrichment — only runs during market hours when Webull is ready
        hist = webull.enrich_with_live_bar(ticker, hist)
        return ticker, hist
    except:
        return ticker, None


def fetch_and_score_options(ticker_setup, stats, min_score=None):
    """Fetch options + score — runs in parallel after history phase"""
    ticker, setup = ticker_setup
    score_floor = min_score if min_score is not None else CONFIG['min_composite_score']
    try:
        # ── Live price refresh ────────────────────────────────────────────────
        # Priority: Tradier quote → Webull quote → yfinance (15-min delayed)
        # Updates current_price / entry / stop / target to live data so the
        # option strike filter selects the right contract.
        live_price = 0
        if tradier.is_ready():
            q = tradier._client.get_quote(ticker)
            live_price = q.get('price', 0)
        if live_price == 0 and webull.is_ready():
            q = webull.get_live_quote(ticker)
            live_price = q.get('price', 0)

        if live_price > 0:
            old_price = setup['current_price']
            # Sanity check: reject if quote looks wrong (>10% diff from daily close)
            if abs(live_price - old_price) / max(old_price, 0.01) < 0.10:
                setup = dict(setup)   # shallow copy — don't mutate the shared dict
                stop_dist = abs(old_price - setup['stop_loss'])
                setup['current_price'] = round(live_price, 2)
                setup['entry']         = round(live_price, 2)
                # Recompute stop/target off new price, preserving ATR distance
                if setup['direction'] == 'BULLISH':
                    setup['stop_loss']    = round(live_price - stop_dist, 2)
                    setup['target_price'] = round(live_price + stop_dist * 2, 2)
                else:
                    setup['stop_loss']    = round(live_price + stop_dist, 2)
                    setup['target_price'] = round(live_price - stop_dist * 2, 2)

        # ── Options chain ─────────────────────────────────────────────────────
        # Use Tradier for real Greeks when token is set, otherwise yfinance
        if tradier.is_ready():
            option = tradier.get_best_option_tradier(ticker, setup, CONFIG)
        else:
            option = get_best_option(ticker, setup)
        if not option:
            stats['no_options'] += 1
            return None
        scoring = calculate_setup_score(setup, option)
        if scoring['composite'] < score_floor:
            stats['low_score'] += 1
            return None
        theta_pct = round((abs(option['theta']) / option['premium']) * 100, 1) if option['premium'] > 0 else 0
        days_to_be = round(abs(option['premium'] / option['theta']), 1) if option['theta'] < 0 else 999
        ai = get_gemini_analysis(ticker, setup, option, scoring)
        return {
            'ticker':              ticker,
            'setup':               setup,
            'option':              option,
            'scoring':             scoring,
            'expected_return_pct': round(CONFIG['profit_target'] * 100, 1),
            'theta_pct_per_day':   theta_pct,
            'days_to_breakeven':   days_to_be,
            'ai_analysis':         ai
        }
    except:
        stats['no_options'] += 1
        return None




# ==================== CORE SCAN LOGIC ====================

def _run_scan_logic() -> dict:
    """
    Full scan pipeline — shared by /api/scan (manual) and the auto-scan thread.
    Returns the same JSON-serialisable dict that /api/scan returns.
    """
    stats = {'scanned': 0, 'no_setup': 0, 'no_options': 0, 'low_score': 0}

    tickers = load_tickers_from_csv()
    if not tickers:
        return {'status': 'error', 'message': 'No tickers loaded from CSV'}
    stats['scanned'] = len(tickers)

    effective_min_score = CONFIG['min_composite_score']

    # Phase 1: Parallel history download
    histories = {}
    with ThreadPoolExecutor(max_workers=CONFIG['scan_workers']) as executor:
        futures = {executor.submit(fetch_ticker_history, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, hist = future.result()
            if hist is not None:
                histories[ticker] = hist

    # Phase 2: Momentum analysis
    candidates = []
    for ticker, hist in histories.items():
        setup = detect_momentum_setup(ticker, hist=hist)
        if setup:
            candidates.append((ticker, setup))
        else:
            stats['no_setup'] += 1

    # Phase 2.5: Rule-based pre-filter — score all candidates deterministically
    # before the expensive per-ticker options chain fetch.  No API call needed.
    prefilter_scores = {}   # ticker → {score, reason}
    if candidates:
        scored = rule_prefilter_candidates(candidates)
        stats['prefilter_total']    = len(scored)
        # Always keep at least top 3 so a tight day still yields results
        threshold = 50
        top3      = [t for t, s, sc, r in scored[:3]]
        qualified = [(t, s) for t, s, sc, r in scored
                     if sc >= threshold or t in top3]
        for t, s, sc, r in scored:
            prefilter_scores[t] = {'score': sc, 'reason': r}
        stats['prefilter_accepted'] = len(qualified)
        stats['prefilter_rejected'] = len(scored) - len(qualified)
        candidates = qualified

    # Phase 3: Parallel options fetch + scoring (pass dynamic score floor)
    results = []
    with ThreadPoolExecutor(max_workers=CONFIG['scan_workers']) as executor:
        futures = [
            executor.submit(fetch_and_score_options, ts, stats, effective_min_score)
            for ts in candidates
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                # Attach pre-filter score if available
                pf = prefilter_scores.get(result['ticker'])
                if pf:
                    result['prefilter'] = pf
                results.append(result)

    results.sort(key=lambda x: x['scoring']['composite'], reverse=True)
    results = results[:CONFIG['max_positions']]

    discord_sent = send_discord_alerts(results)

    return {
        'status':       'ok',
        'results':      results,
        'stats':        stats,
        'discord_sent': discord_sent,
        'scan_time':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_source':  (
            'tradier_live'  if tradier.is_ready() else
            'webull_live'   if webull.is_ready()  else
            'yfinance_delayed'
        ),
    }


# ==================== AUTO-SCAN BACKGROUND LOOP ====================

_autoscan_lock     = threading.Lock()
_autoscan_active   = False
_autoscan_thread   = None
_autoscan_interval = 5 * 60          # seconds between scans (default 5 min)
_last_scan_results = None
_last_scan_time    = None
_next_scan_time    = None


def _market_is_open() -> bool:
    """Market hours check — uses Tradier clock if available, otherwise time-based."""
    if tradier.is_ready():
        return tradier.market_open()
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    return (9, 30) <= (h, m) <= (16, 0)


def _autoscan_worker():
    """
    Daemon thread: runs _run_scan_logic() every _autoscan_interval seconds
    while the market is open and _autoscan_active is True.
    Checks for a stop signal every 30 seconds so it can exit promptly.
    """
    global _autoscan_active, _last_scan_results, _last_scan_time, _next_scan_time
    print('[AutoScan] Thread started')

    while True:
        with _autoscan_lock:
            if not _autoscan_active:
                break

        if _market_is_open():
            print(f'[AutoScan] Running scan at {datetime.now().strftime("%H:%M:%S")}')
            try:
                data = _run_scan_logic()
                with _autoscan_lock:
                    _last_scan_results = data
                    _last_scan_time    = datetime.now().isoformat()
                    _next_scan_time    = (
                        datetime.now() + timedelta(seconds=_autoscan_interval)
                    ).isoformat()
                sig_count = len(data.get('results', []))
                print(f'[AutoScan] Done — {sig_count} signal(s). Next: {_next_scan_time}')
            except Exception as e:
                print(f'[AutoScan] Scan error: {e}')
        else:
            print(f'[AutoScan] Market closed — idling')

        # Wait for next interval in 30-second slices so stop is responsive
        slices = max(1, _autoscan_interval // 30)
        for _ in range(slices):
            time.sleep(30)
            with _autoscan_lock:
                if not _autoscan_active:
                    break

    print('[AutoScan] Thread stopped')


# ==================== API ROUTES ====================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/scan', methods=['POST'])
def run_scan():
    try:
        return jsonify(_run_scan_logic())
    except Exception as e:
        return jsonify({
            'status': 'error', 'message': str(e),
            'trace':  traceback.format_exc()
        }), 500


@app.route('/api/autoscan', methods=['GET', 'POST'])
def autoscan_endpoint():
    """
    GET  → returns auto-scan status + last results
    POST → start / stop the auto-scan loop

    POST body (JSON):
      { "action": "start", "interval_minutes": 5 }   ← start scanning every 5 min
      { "action": "stop" }                            ← stop
    """
    global _autoscan_active, _autoscan_thread, _autoscan_interval

    if request.method == 'GET':
        with _autoscan_lock:
            return jsonify({
                'active':          _autoscan_active,
                'interval_seconds': _autoscan_interval,
                'last_scan_time':  _last_scan_time,
                'next_scan_time':  _next_scan_time,
                'market_open':     _market_is_open(),
                'data_source':     (
                    'tradier_live'  if tradier.is_ready() else
                    'webull_live'   if webull.is_ready()  else
                    'yfinance_delayed'
                ),
                'results':         _last_scan_results,
            })

    # POST
    body     = request.get_json(silent=True) or {}
    action   = body.get('action', 'start')
    interval = int(body.get('interval_minutes', 5))

    with _autoscan_lock:
        _autoscan_interval = max(1, interval) * 60

        if action == 'stop':
            _autoscan_active = False
            return jsonify({'status': 'stopped'})

        if _autoscan_active:
            return jsonify({
                'status':           'already_running',
                'interval_seconds': _autoscan_interval,
            })

        _autoscan_active = True

    # Start thread outside the lock
    _autoscan_thread = threading.Thread(target=_autoscan_worker, daemon=True)
    _autoscan_thread.start()
    return jsonify({
        'status':           'started',
        'interval_seconds': _autoscan_interval,
        'market_open':      _market_is_open(),
    })


@app.route('/api/ticker-count', methods=['GET'])
def ticker_count():
    tickers = load_tickers_from_csv()
    return jsonify({'count': len(tickers)})


@app.route('/api/diagnose', methods=['GET'])
def diagnose():
    results = {}

    # ── yfinance ─────────────────────────────────────────────────────────────
    try:
        results['yfinance_version'] = yf.__version__
    except:
        results['yfinance_version'] = 'unknown'

    try:
        aapl = yf.Ticker('AAPL')
        hist = aapl.history(period='5d', interval='1d')
        results['aapl_price_test'] = (
            f'OK — {len(hist)} bars, last close ${float(hist["Close"].iloc[-1]):.2f}'
            if not hist.empty else 'FAIL — empty dataframe'
        )
    except Exception as e:
        results['aapl_price_test'] = f'FAIL — {str(e)[:200]}'

    try:
        aapl = yf.Ticker('AAPL')
        exps = aapl.options
        results['yfinance_options_test'] = (
            f'OK — {len(exps)} expiry dates' if exps else 'FAIL — no expiry dates'
        )
    except Exception as e:
        results['yfinance_options_test'] = f'FAIL — {str(e)[:200]}'

    # ── Webull ────────────────────────────────────────────────────────────────
    results['webull_key_set'] = bool(WEBULL_APP_KEY and WEBULL_APP_SECRET)
    results['webull_ready']   = webull.is_ready()
    if webull.is_ready():
        try:
            q = webull.get_live_quote('AAPL')
            results['webull_quote_test'] = (
                f'OK — AAPL ${q["price"]} vol {q["volume"]:,}'
                if q.get('price', 0) > 0 else f'FAIL — {q}'
            )
        except Exception as e:
            results['webull_quote_test'] = f'FAIL — {str(e)[:200]}'

        try:
            bars = webull.get_intraday_history('AAPL', interval='M5')
            results['webull_bars_test'] = (
                f'OK — {len(bars)} 5-min bars'
                if not bars.empty else 'FAIL — empty DataFrame'
            )
        except Exception as e:
            results['webull_bars_test'] = f'FAIL — {str(e)[:200]}'
        results['webull_market_open'] = webull.market_open()
    else:
        results['webull_quote_test'] = 'SKIP — keys not set'
        results['webull_bars_test']  = 'SKIP — keys not set'

    # ── Tradier ───────────────────────────────────────────────────────────────
    results['tradier_token_set'] = bool(TRADIER_TOKEN)
    results['tradier_ready']     = tradier.is_ready()
    if tradier.is_ready():
        try:
            q = tradier._client.get_quote('AAPL')
            results['tradier_quote_test'] = (
                f'OK — AAPL ${q["price"]} vol {q["volume"]:,}'
                if q.get('price', 0) > 0 else f'FAIL — {q}'
            )
        except Exception as e:
            results['tradier_quote_test'] = f'FAIL — {str(e)[:200]}'

        try:
            exps = tradier._client.get_expirations('AAPL')
            results['tradier_options_test'] = (
                f'OK — {len(exps)} expiry dates: {exps[:3]}'
                if exps else 'FAIL — no expiry dates'
            )
        except Exception as e:
            results['tradier_options_test'] = f'FAIL — {str(e)[:200]}'

        results['tradier_market_open'] = tradier.market_open()
    else:
        results['tradier_quote_test']   = 'SKIP — token not set'
        results['tradier_options_test'] = 'SKIP — token not set'

    # ── Auto-scan ─────────────────────────────────────────────────────────────
    results['autoscan_active']    = _autoscan_active
    results['last_scan_time']     = _last_scan_time
    results['next_scan_time']     = _next_scan_time

    tickers = load_tickers_from_csv()
    results['csv_tickers'] = len(tickers)
    return jsonify(results)


@app.route('/api/analyze-backtest', methods=['POST'])
def analyze_backtest_endpoint():
    """
    Trigger Claude to deeply analyze backtest_results.csv and save insights.
    Call this after running backtest.py to update the scanner's AI context.
    """
    try:
        from claude_analyzer import analyze_backtest
        insights = analyze_backtest()
        if insights is None:
            return jsonify({'status': 'error', 'message': 'Analysis failed — check API key and backtest_results.csv'}), 500
        load_backtest_insights()   # Reload into memory immediately
        return jsonify({'status': 'ok', 'insights': insights})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/tune-config', methods=['GET'])
def tune_config():
    """
    Return Claude's CONFIG recommendations from the last backtest analysis.
    Call /api/analyze-backtest first if no insights exist yet.
    """
    if _BACKTEST_INSIGHTS is None:
        return jsonify({
            'status': 'no_insights',
            'message': 'No backtest analysis found. POST to /api/analyze-backtest first.'
        }), 400

    ins = _BACKTEST_INSIGHTS
    current = {
        'min_adx':             CONFIG['min_adx'],
        'min_volume_ratio':    CONFIG['min_volume_ratio'],
        'min_price_move':      CONFIG['min_price_move'],
        'min_pattern_quality': CONFIG['min_pattern_quality'],
        'min_composite_score': CONFIG['min_composite_score'],
        'dte_range':           CONFIG['dte_range'],
        'max_adx':             CONFIG.get('max_adx', 'not set'),
    }
    return jsonify({
        'status':              'ok',
        'current_config':      current,
        'recommended_changes': ins.get('config_recommendations', {}),
        'filter_rules':        ins.get('filter_rules', []),
        'top_improvements':    ins.get('top_3_improvements', []),
        'signal_count_fix':    ins.get('signal_count_fix', ''),
        'summary':             ins.get('summary', ''),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

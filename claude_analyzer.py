"""
Claude Backtest Analyzer
========================
Sends backtest_results.csv to Claude (with adaptive thinking) for deep analysis.
Outputs structured recommendations that the scanner uses to:
  1. Pre-filter signal candidates before expensive options lookup
  2. Enrich per-signal commentary with win probability estimates
  3. Guide CONFIG parameter tuning

Usage:
    python claude_analyzer.py                  # analyze backtest_results.csv
    python claude_analyzer.py --csv my_bt.csv  # custom file
    python claude_analyzer.py --print-only     # print insights already saved
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import anthropic
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flask_app import ANTHROPIC_API_KEY, CONFIG

INSIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_insights.json')

# ── Output schema ──────────────────────────────────────────────────────────────

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "2-3 sentence plain-English summary of key findings"
        },
        "top_3_improvements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The 3 highest-impact changes to make, in order of impact"
        },
        "feature_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature":  {"type": "string"},
                    "finding":  {"type": "string"},
                    "direction": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]}
                },
                "required": ["feature", "finding", "direction"],
                "additionalProperties": False
            }
        },
        "config_recommendations": {
            "type": "object",
            "properties": {
                "max_adx":              {"type": ["number", "null"]},
                "min_adx":              {"type": ["number", "null"]},
                "min_volume_ratio":     {"type": ["number", "null"]},
                "min_price_move":       {"type": ["number", "null"]},
                "min_pattern_quality":  {"type": ["number", "null"]},
                "min_composite_score":  {"type": ["number", "null"]},
                "skip_gap_and_go":      {"type": ["boolean", "null"]},
                "dte_range_min":        {"type": ["number", "null"]},
                "dte_range_max":        {"type": ["number", "null"]}
            },
            "additionalProperties": False
        },
        "filter_rules": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific conditional rules, e.g. 'reject GAP_AND_GO when ADX > 32'"
        },
        "signal_count_diagnosis": {
            "type": "string",
            "description": "Why signal count is low (0.28/day) and which specific filter is most restrictive"
        },
        "signal_count_fix": {
            "type": "string",
            "description": "Specific filter to loosen and by how much to reach ~1 signal/day without hurting win rate"
        },
        "prefilter_scoring_guide": {
            "type": "string",
            "description": "Concise guide for scoring new candidates 0-100. Claude uses this at scan time."
        },
        "win_probability_model": {
            "type": "string",
            "description": "How to estimate win probability for a new signal given its features"
        }
    },
    "required": [
        "summary", "top_3_improvements", "feature_findings",
        "config_recommendations", "filter_rules",
        "signal_count_diagnosis", "signal_count_fix",
        "prefilter_scoring_guide", "win_probability_model"
    ],
    "additionalProperties": False
}


# ── Data formatting ────────────────────────────────────────────────────────────

def format_backtest_data(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"BACKTEST: {len(df)} signals | {df['date'].min()} → {df['date'].max()} | ~{df['date'].nunique()} trading days")
    lines.append(f"Directions: {df['direction'].value_counts().to_dict()}")
    lines.append("")

    for d in [3, 5, 10]:
        col = f'ret_{d}d'
        v = df[col].dropna()
        wins = (v >= 2.0).sum()
        lines.append(f"{d}d: WR={wins/len(v)*100:.1f}%  avg={v.mean():+.2f}%  median={v.median():+.2f}%  n={len(v)}")

    lines.append("")
    lines.append("SIGNAL DATA (pipe-delimited):")
    lines.append("ticker|date|direction|pattern|pat_q|rsi|adx|vol_ratio|near52w|macd_exp|ret_3d|ret_5d|ret_10d|mfe_5d|mae_5d")

    for _, r in df.sort_values('ret_5d', ascending=False).iterrows():
        lines.append(
            f"{r['ticker']}|{r['date']}|{r['direction']}|{r['pattern']}|"
            f"{r['pat_quality']}|{r['rsi']:.1f}|{r['adx']:.1f}|{r['volume_ratio']:.2f}|"
            f"{'Y' if r['near_52w'] else 'N'}|"
            f"{'Y' if r['macd_expanding'] else 'N'}|"
            f"{r['ret_3d']:+.2f}|{r['ret_5d']:+.2f}|{r['ret_10d']:+.2f}|"
            f"{r['mfe_5d']:+.2f}|{r['mae_5d']:+.2f}"
        )

    lines.append("")
    lines.append("CURRENT CONFIG (what the scanner is already using):")
    lines.append(
        f"  min_adx={CONFIG['min_adx']}  min_volume_ratio={CONFIG['min_volume_ratio']}  "
        f"min_price_move={CONFIG['min_price_move']}  "
        f"min_pattern_quality={CONFIG['min_pattern_quality']}  min_composite_score={CONFIG['min_composite_score']}  "
        f"dte_range={CONFIG['dte_range']}  skip_v_reversals={CONFIG['skip_v_reversals']}"
    )

    return "\n".join(lines)


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze_backtest(csv_path: str = None) -> dict:
    """
    Deep analysis of backtest results using Claude with adaptive thinking.
    Saves structured insights to backtest_insights.json.
    """
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return None

    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_results.csv')

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run backtest.py first.")
        return None

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} signals from {csv_path}")

    backtest_str = format_backtest_data(df)

    system = (
        "You are a quantitative trading analyst specializing in momentum breakout options systems. "
        "You analyze historical signal data with statistical rigor and produce specific, actionable "
        "recommendations. Base every conclusion on the actual numbers in the data — cite win rates, "
        "specific tickers, and feature values. Do not generalize beyond what the data supports."
    )

    user = f"""Analyze this momentum scanner backtest and return structured JSON recommendations.

{backtest_str}

CRITICAL QUESTIONS (answer all with specific numbers from the data):

1. FEATURE ANALYSIS: For each feature (adx, volume_ratio, pattern, near_52w, macd_exp),
   what is the win rate difference between high vs low values? Which feature best separates winners
   from losers on 5-day returns?

2. CONFIG TUNING: What specific parameter values should change? Give exact numbers.
   E.g., if ADX > 35 signals lose, recommend max_adx=35.

3. CONDITIONAL RULES: What conditional filter rules should be added?
   E.g., "reject GAP_AND_GO when ADX > 32 AND volume_ratio < 2.0".

4. SIGNAL COUNT: The scanner produces only 0.28 signals/day (target: 1+/day).
   Which single filter is most responsible? What value should it be relaxed to in order
   to 3x the signal count WITHOUT significantly hurting win rate?

5. PRE-FILTER SCORING: Write a concise 3-4 sentence scoring guide that can be used at runtime
   to rate a new candidate 0-100. Include the exact features and thresholds that matter most.

6. WIN PROBABILITY MODEL: How should win probability be estimated for a new signal?
   Reference the specific feature values associated with wins vs losses.

Return ONLY a JSON object — no markdown fences, no prose before or after — matching this exact schema:
""" + json.dumps(_ANALYSIS_SCHEMA, indent=2) + """

All required keys must be present."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print("Sending to Claude (adaptive thinking enabled — may take 30-60 seconds)...")
    print("  ", end='', flush=True)

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 8000},
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        thinking_shown = False
        for event in stream:
            if hasattr(event, 'type'):
                if event.type == 'content_block_start':
                    cb = getattr(event, 'content_block', None)
                    if cb and cb.type == 'thinking' and not thinking_shown:
                        print("Thinking", end='', flush=True)
                        thinking_shown = True
                elif event.type == 'content_block_delta':
                    delta = getattr(event, 'delta', None)
                    if delta and getattr(delta, 'type', '') == 'thinking_delta':
                        print('.', end='', flush=True)
        response = stream.get_final_message()

    print(" done.\n")

    text = next((b.text for b in response.content if b.type == 'text'), None)
    if not text:
        print("ERROR: No text in Claude response.")
        print("Content blocks:", [b.type for b in response.content])
        return None

    # Strip markdown fences if present
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1]
        text = text.rsplit('```', 1)[0].strip()

    insights = json.loads(text)

    # Save for use by scanner
    with open(INSIGHTS_FILE, 'w') as f:
        json.dump(insights, f, indent=2)
    print(f"Saved to: {INSIGHTS_FILE}\n")

    _print_insights(insights)
    return insights


def _safe_print(text: str):
    """Print, replacing any unencodable chars with '?' for Windows terminals."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace'))


def _print_insights(ins: dict):
    w = 65
    _safe_print("=" * w)
    _safe_print("  CLAUDE'S ANALYSIS")
    _safe_print("=" * w)

    _safe_print(f"\n{ins['summary']}\n")

    _safe_print("TOP 3 IMPROVEMENTS:")
    for i, item in enumerate(ins['top_3_improvements'], 1):
        _safe_print(f"  {i}. {item}")

    _safe_print(f"\nSIGNAL COUNT DIAGNOSIS:\n  {ins['signal_count_diagnosis']}")
    _safe_print(f"\nSIGNAL COUNT FIX:\n  {ins['signal_count_fix']}")

    _safe_print("\nRECOMMENDED CONFIG CHANGES:")
    cfg = ins.get('config_recommendations', {})
    current_map = {
        'max_adx': CONFIG.get('max_adx', 'not set'),
        'min_adx': CONFIG['min_adx'],
        'min_volume_ratio': CONFIG['min_volume_ratio'],
        'min_price_move': CONFIG['min_price_move'],
        'min_pattern_quality': CONFIG['min_pattern_quality'],
        'min_composite_score': CONFIG['min_composite_score'],
    }
    for k, v in cfg.items():
        if v is not None:
            cur = current_map.get(k, 'N/A')
            marker = ' <--' if cur != v and cur != 'N/A' else ''
            _safe_print(f"  {k:28s}  {str(cur):>8}  ->  {v}{marker}")

    _safe_print("\nCONDITIONAL FILTER RULES:")
    for rule in ins.get('filter_rules', []):
        _safe_print(f"  * {rule}")

    _safe_print(f"\nPRE-FILTER SCORING GUIDE:\n  {ins['prefilter_scoring_guide']}")
    _safe_print(f"\nWIN PROBABILITY MODEL:\n  {ins['win_probability_model']}")
    _safe_print("=" * w)


# ── Entry point ────────────────────────────────────────────────────────────────

def load_insights() -> dict:
    """Load saved insights. Returns None if not yet generated."""
    if os.path.exists(INSIGHTS_FILE):
        with open(INSIGHTS_FILE) as f:
            return json.load(f)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze backtest results with Claude')
    parser.add_argument('--csv', type=str, default=None, help='Path to backtest CSV')
    parser.add_argument('--print-only', action='store_true', help='Print saved insights without re-running')
    args = parser.parse_args()

    if args.print_only:
        ins = load_insights()
        if ins:
            _print_insights(ins)
        else:
            print("No saved insights found. Run without --print-only first.")
    else:
        analyze_backtest(args.csv)

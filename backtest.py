"""
Signal Backtester — tests momentum scanner signals on historical data.

How it works:
  For each ticker, it walks through the last N trading days and asks:
  "Would the scanner have fired a signal on this day?"
  If yes, it records the signal and checks what the stock did over the
  next 3, 5, and 10 days.

Since we can't get historical options prices from yfinance, we use
a stock-return proxy:
  - WIN  = stock moved 2%+ in the right direction within 5 days
  - LOSS = stock moved < 2% or against you

This maps well to options reality: a 2% underlying move with a 0.5-delta
option at $2.50 gives roughly 50% gain — your profit target.

Usage:
  python backtest.py               # runs with default settings
  python backtest.py --days 60     # backtest last 60 trading days
  python backtest.py --tickers 100 # test first 100 tickers
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# -- Pull in scanner logic ----------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flask_app import (
    detect_momentum_setup,
    load_tickers_from_csv,
    CONFIG,
)

# -- Settings -----------------------------------------------------------------
BACKTEST_DAYS   = 90       # How many recent trading days to test
WIN_THRESHOLD   = 2.0      # % move in right direction = WIN (proxy for 50% option gain)
BIG_WIN         = 4.0      # % move = BIG WIN (proxy for 100%+ option gain)
FORWARD_WINDOWS = [3, 5, 10]   # Days to check forward returns

# -- Data download -------------------------------------------------------------

def download_history(ticker):
    """Download 2 years of daily OHLCV — enough for backtest + indicator warmup."""
    try:
        hist = yf.Ticker(ticker).history(period='2y', interval='1d')
        if hist.empty or len(hist) < 120:
            return ticker, None
        return ticker, hist
    except:
        return ticker, None


def download_all(tickers, workers=12):
    print(f"  Downloading {len(tickers)} tickers... ", end='', flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_history, t): t for t in tickers}
        for f in as_completed(futures):
            t, hist = f.result()
            if hist is not None:
                results[t] = hist
    print(f"{len(results)} succeeded")
    return results


# -- Signal replay -------------------------------------------------------------

def replay_signals(ticker, hist, backtest_days):
    """
    Walk through the last `backtest_days` of data.
    For each day, feed the scanner only data UP TO that day (no look-ahead).
    Record signals and their forward returns.
    """
    signals = []
    total_bars = len(hist)

    # We need at least 60 bars of history for indicator warmup
    min_warmup = 65
    start_idx  = max(min_warmup, total_bars - backtest_days)

    for i in range(start_idx, total_bars):
        hist_slice = hist.iloc[:i+1]   # Data available on day i (no look-ahead)

        setup = detect_momentum_setup(ticker, hist=hist_slice)
        if not setup:
            continue

        signal_date  = hist_slice.index[-1]
        entry_price  = float(hist_slice['Close'].iloc[-1])
        direction    = setup['direction']

        # Forward returns — data AFTER the signal day
        future_data = hist.iloc[i+1 : i+1+max(FORWARD_WINDOWS)+1]
        if len(future_data) < 3:
            continue

        fwd = {}
        for d in FORWARD_WINDOWS:
            if len(future_data) >= d:
                fp  = float(future_data['Close'].iloc[d-1])
                ret = (fp - entry_price) / entry_price * 100
                fwd[f'ret_{d}d'] = round(ret if direction == 'BULLISH' else -ret, 2)
            else:
                fwd[f'ret_{d}d'] = None

        # Max favorable / adverse excursion in 5-day window
        if len(future_data) >= 5:
            fp5 = future_data['Close'].iloc[:5]
            if direction == 'BULLISH':
                mfe = (fp5.max() - entry_price) / entry_price * 100
                mae = (fp5.min() - entry_price) / entry_price * 100
            else:
                mfe = (entry_price - fp5.min()) / entry_price * 100
                mae = (entry_price - fp5.max()) / entry_price * 100
        else:
            mfe = mae = 0.0

        signals.append({
            'ticker':        ticker,
            'date':          signal_date.strftime('%Y-%m-%d'),
            'direction':     direction,
            'pattern':       setup['chart_analysis']['pattern_type'],
            'pat_quality':   setup['chart_analysis']['quality_score'],
            'rsi':           setup['rsi'],
            'adx':           setup['adx']['adx'],
            'volume_ratio':  setup['volume_ratio'],
            'roc_5':         setup.get('roc_5', 0),
            'near_52w':      setup['week52']['near_52w_high'],
            'at_52w':        setup['week52']['at_52w_high'],
            'macd_bullish':  setup['macd_bullish'],
            'macd_expanding': setup['macd_expanding'],
            'entry':         round(entry_price, 2),
            'mfe_5d':        round(mfe, 2),
            'mae_5d':        round(mae, 2),
            **fwd,
        })

    return signals


# -- Reporting -----------------------------------------------------------------

def win_stats(series, threshold):
    """Return (win_rate%, count, avg_return) for a return series."""
    valid = series.dropna()
    if len(valid) == 0:
        return 0.0, 0, 0.0
    wins = (valid >= threshold).sum()
    return round(wins / len(valid) * 100, 1), len(valid), round(valid.mean(), 2)


def print_report(df):
    total = len(df)
    print(f"\n{'='*65}")
    print(f"  BACKTEST RESULTS  —  {total} total signals")
    print(f"{'='*65}")

    # -- Overall win rates -----------------------------------------------------
    print(f"\n{'-'*65}")
    print(f"  OVERALL WIN RATES  (win = stock moved {WIN_THRESHOLD}%+ in right direction)")
    print(f"{'-'*65}")
    for d in FORWARD_WINDOWS:
        col = f'ret_{d}d'
        wr, n, avg = win_stats(df[col], WIN_THRESHOLD)
        bwr, _, _  = win_stats(df[col], BIG_WIN)
        print(f"  {d:2d}-day  |  Win: {wr:5.1f}%  |  Big win ({BIG_WIN}%+): {bwr:5.1f}%  |  Avg return: {avg:+.2f}%  |  n={n}")

    # -- Signals per day -------------------------------------------------------
    unique_days = df['date'].nunique()
    print(f"\n  Signals per trading day:  {total/max(unique_days,1):.1f}  "
          f"({total} signals over {unique_days} days)")

    # -- By direction ---------------------------------------------------------
    print(f"\n{'-'*65}")
    print(f"  BY DIRECTION")
    print(f"{'-'*65}")
    for dir_, grp in df.groupby('direction'):
        wr, n, avg = win_stats(grp['ret_5d'], WIN_THRESHOLD)
        print(f"  {dir_:8s}  |  Win: {wr:5.1f}%  |  Avg 5d: {avg:+.2f}%  |  n={n}")

    # -- By pattern type -------------------------------------------------------
    print(f"\n{'-'*65}")
    print(f"  BY PATTERN TYPE  (5-day returns)")
    print(f"{'-'*65}")
    pattern_stats = []
    for pattern, grp in df.groupby('pattern'):
        wr, n, avg = win_stats(grp['ret_5d'], WIN_THRESHOLD)
        if n >= 3:
            pattern_stats.append((wr, n, avg, pattern))
    for wr, n, avg, pattern in sorted(pattern_stats, reverse=True):
        bar = '#' * int(wr / 5)
        print(f"  {pattern:30s}  WR:{wr:5.1f}%  avg:{avg:+.2f}%  n={n:3d}  {bar}")

    # -- By ADX bucket ---------------------------------------------------------
    print(f"\n{'-'*65}")
    print(f"  BY ADX STRENGTH  (5-day returns)  — is trend filter working?")
    print(f"{'-'*65}")
    df['adx_bucket'] = pd.cut(df['adx'], bins=[0, 20, 25, 35, 100],
                               labels=['<20 (no trend)', '20-25', '25-35', '>35 (strong)'])
    for bucket, grp in df.groupby('adx_bucket', observed=True):
        wr, n, avg = win_stats(grp['ret_5d'], WIN_THRESHOLD)
        if n >= 3:
            print(f"  ADX {str(bucket):18s}  WR:{wr:5.1f}%  avg:{avg:+.2f}%  n={n:3d}")

    # -- Volume ratio ---------------------------------------------------------
    print(f"\n{'-'*65}")
    print(f"  BY VOLUME RATIO  (5-day returns)  — is volume filter working?")
    print(f"{'-'*65}")
    df['vol_bucket'] = pd.cut(df['volume_ratio'], bins=[0, 2, 3, 5, 100],
                               labels=['1-2x', '2-3x', '3-5x', '>5x'])
    for bucket, grp in df.groupby('vol_bucket', observed=True):
        wr, n, avg = win_stats(grp['ret_5d'], WIN_THRESHOLD)
        if n >= 3:
            print(f"  Volume {str(bucket):10s}  WR:{wr:5.1f}%  avg:{avg:+.2f}%  n={n:3d}")

    # -- 52W high proximity ----------------------------------------------------
    print(f"\n{'-'*65}")
    print(f"  52-WEEK HIGH PROXIMITY  (5-day returns)")
    print(f"{'-'*65}")
    for label, mask in [('At 52W high', df['at_52w']),
                         ('Near 52W high', df['near_52w'] & ~df['at_52w']),
                         ('Far from 52W high', ~df['near_52w'])]:
        grp = df[mask]
        wr, n, avg = win_stats(grp['ret_5d'], WIN_THRESHOLD)
        if n >= 3:
            print(f"  {label:22s}  WR:{wr:5.1f}%  avg:{avg:+.2f}%  n={n:3d}")

    # -- MACD alignment --------------------------------------------------------
    print(f"\n{'-'*65}")
    print(f"  MACD  (5-day returns)  — does MACD alignment predict wins?")
    print(f"{'-'*65}")
    for label, mask in [('MACD bullish + expanding', df['macd_bullish'] & df['macd_expanding']),
                         ('MACD bullish only',        df['macd_bullish'] & ~df['macd_expanding']),
                         ('MACD bearish',             ~df['macd_bullish'])]:
        grp = df[mask]
        wr, n, avg = win_stats(grp['ret_5d'], WIN_THRESHOLD)
        if n >= 3:
            print(f"  {label:28s}  WR:{wr:5.1f}%  avg:{avg:+.2f}%  n={n:3d}")

    # -- Best individual signals -----------------------------------------------
    print(f"\n{'-'*65}")
    print(f"  TOP 10 SIGNALS BY 5-DAY RETURN")
    print(f"{'-'*65}")
    top = df.nlargest(10, 'ret_5d')[['ticker','date','direction','pattern','rsi','adx','ret_3d','ret_5d','ret_10d']]
    print(top.to_string(index=False))

    print(f"\n{'='*65}\n")


# -- Main ----------------------------------------------------------------------

def run_backtest(backtest_days=BACKTEST_DAYS, max_tickers=200):
    print(f"\n{'='*65}")
    print(f"  MOMENTUM SCANNER BACKTEST")
    print(f"  Testing last {backtest_days} trading days  |  {max_tickers} tickers")
    print(f"{'='*65}\n")

    # Load tickers
    tickers = load_tickers_from_csv()[:max_tickers]
    print(f"Loaded {len(tickers)} tickers from CSV")

    # Download all ticker data
    print("Downloading ticker histories:")
    all_data = download_all(tickers, workers=CONFIG['scan_workers'])

    # Replay signals
    print(f"\nReplaying signals (walking {backtest_days} days per ticker)...")
    all_signals = []
    for i, (ticker, hist) in enumerate(all_data.items()):
        sigs = replay_signals(ticker, hist, backtest_days)
        all_signals.extend(sigs)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_data)} tickers done  —  {len(all_signals)} signals so far")

    if not all_signals:
        print("\nNo signals found. Try loosening CONFIG filters or increasing backtest_days.")
        return None

    df = pd.DataFrame(all_signals).sort_values('date').reset_index(drop=True)
    print(f"\nTotal signals found: {len(df)}")

    # Print report
    print_report(df)

    # Save CSV
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_results.csv')
    df.to_csv(out_path, index=False)
    print(f"Full results saved to:  {out_path}\n")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Momentum Scanner Backtester')
    parser.add_argument('--days',    type=int, default=BACKTEST_DAYS, help='Trading days to backtest')
    parser.add_argument('--tickers', type=int, default=200,           help='Max tickers to test')
    args = parser.parse_args()

    run_backtest(backtest_days=args.days, max_tickers=args.tickers)

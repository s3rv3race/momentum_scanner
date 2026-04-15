# -*- coding: utf-8 -*-
"""
tradier.py -- Real-time market data via Tradier brokerage API.

Drop-in replacement for yfinance options chains + intraday bars.
Set TRADIER_TOKEN in flask_app.py and everything switches automatically.

Tradier free account:
  https://tradier.com/individual/api
  Gives you: real-time quotes, real-time options chains, intraday bars,
             real Greeks (actual delta/IV -- not estimated like yfinance).
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# ── Tradier endpoints ──────────────────────────────────────────────────────────
_LIVE_BASE    = 'https://api.tradier.com/v1'
_SANDBOX_BASE = 'https://sandbox.tradier.com/v1'   # for testing without real account


class TradierClient:
    """
    Thin wrapper around the Tradier REST API.
    Use live=True for your real brokerage token.
    Use live=False (sandbox) for testing -- returns fake quotes but real structure.
    """

    def __init__(self, token: str, live: bool = True):
        self.token   = token
        self.base    = _LIVE_BASE if live else _SANDBOX_BASE
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Accept':        'application/json'
        }

    def _get(self, path: str, params: dict = None):
        resp = requests.get(
            f'{self.base}{path}',
            headers=self.headers,
            params=params or {},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    # ── QUOTES ────────────────────────────────────────────────────────────────

    def get_quote(self, ticker: str) -> dict:
        """Real-time quote for one ticker. Returns price, volume, change %."""
        try:
            data = self._get('/markets/quotes', {'symbols': ticker, 'greeks': 'false'})
            q = data['quotes']['quote']
            return {
                'ticker':    ticker,
                'price':     float(q.get('last') or q.get('close') or 0),
                'change':    float(q.get('change') or 0),
                'change_pct': float(q.get('change_percentage') or 0),
                'volume':    int(q.get('volume') or 0),
                'avg_volume': int(q.get('average_volume') or 0),
            }
        except Exception as e:
            return {'ticker': ticker, 'price': 0, 'change': 0,
                    'change_pct': 0, 'volume': 0, 'avg_volume': 0, 'error': str(e)}

    # ── INTRADAY BARS ─────────────────────────────────────────────────────────

    def get_intraday_bars(self, ticker: str, interval: str = '5min',
                          session_filter: str = 'open') -> pd.DataFrame:
        """
        Fetch intraday OHLCV bars for today (and recent history for warmup).

        interval options: '1min', '5min', '15min'
        session_filter:   'open' (regular hours only) | 'all' (pre/post market)

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (timezone-aware)
        """
        try:
            data = self._get('/markets/timesales', {
                'symbol':         ticker,
                'interval':       interval,
                'session_filter': session_filter,
            })
            series = data.get('series', {})
            if not series or series == 'null':
                return pd.DataFrame()

            rows = series.get('data', [])
            if not rows:
                return pd.DataFrame()
            if isinstance(rows, dict):
                rows = [rows]   # single-bar edge case

            df = pd.DataFrame(rows)
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time').sort_index()
            df = df.rename(columns={
                'open':   'Open',
                'high':   'High',
                'low':    'Low',
                'close':  'Close',
                'volume': 'Volume',
            })
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')
            return df.dropna()
        except Exception as e:
            print(f'Tradier intraday bars failed for {ticker}: {e}')
            return pd.DataFrame()

    # ── OPTIONS EXPIRATIONS ───────────────────────────────────────────────────

    def get_expirations(self, ticker: str) -> list:
        """Return list of expiration date strings: ['2025-04-17', '2025-04-24', ...]"""
        try:
            data = self._get('/markets/options/expirations', {
                'symbol':           ticker,
                'includeAllRoots':  'true',
                'strikes':          'false',
            })
            exps = data.get('expirations', {}).get('date', [])
            if isinstance(exps, str):
                exps = [exps]
            return exps or []
        except Exception as e:
            print(f'Tradier expirations failed for {ticker}: {e}')
            return []

    # ── OPTIONS CHAIN ─────────────────────────────────────────────────────────

    def get_options_chain(self, ticker: str, expiration: str,
                          option_type: str = 'call') -> pd.DataFrame:
        """
        Fetch full options chain for one expiration.
        option_type: 'call' | 'put'

        Returns DataFrame with columns matching what the scanner expects:
          strike, bid, ask, volume, openInterest, impliedVolatility,
          delta, gamma, theta, vega  (REAL Greeks -- not estimated)
        """
        try:
            data = self._get('/markets/options/chains', {
                'symbol':     ticker,
                'expiration': expiration,
                'greeks':     'true',
            })
            opts = data.get('options', {}).get('option', [])
            if not opts:
                return pd.DataFrame()
            if isinstance(opts, dict):
                opts = [opts]

            df = pd.DataFrame(opts)

            # Filter to requested type
            df = df[df['option_type'] == option_type].copy()
            if df.empty:
                return pd.DataFrame()

            # Normalise column names to match what get_best_option() expects
            rename = {
                'strike':           'strike',
                'bid':              'bid',
                'ask':              'ask',
                'volume':           'volume',
                'open_interest':    'openInterest',
                'greeks':           '_greeks_raw',  # parsed below
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

            # Extract Greeks from nested dict
            if '_greeks_raw' in df.columns:
                greeks = df['_greeks_raw'].apply(
                    lambda g: g if isinstance(g, dict) else {}
                )
                df['delta'] = greeks.apply(lambda g: g.get('delta', None))
                df['gamma'] = greeks.apply(lambda g: g.get('gamma', None))
                df['theta'] = greeks.apply(lambda g: g.get('theta', None))
                df['vega']  = greeks.apply(lambda g: g.get('vega',  None))
                df['impliedVolatility'] = greeks.apply(lambda g: g.get('smv_vol', None))
                df.drop(columns=['_greeks_raw'], inplace=True)

            # Numeric coerce
            for col in ['strike', 'bid', 'ask', 'volume', 'openInterest',
                        'delta', 'gamma', 'theta', 'vega', 'impliedVolatility']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['openInterest'] = df['openInterest'].fillna(0).astype(int)
            df['volume']       = df['volume'].fillna(0).astype(int)
            return df.reset_index(drop=True)

        except Exception as e:
            print(f'Tradier options chain failed for {ticker} {expiration}: {e}')
            return pd.DataFrame()

    # ── MARKET STATUS ─────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Returns True if the US equity market is currently open."""
        try:
            data  = self._get('/markets/clock')
            state = data.get('clock', {}).get('state', '')
            return state == 'open'
        except:
            # Fall back to time check: 9:30-16:00 ET Mon-Fri
            now = datetime.now()
            if now.weekday() >= 5:
                return False
            h, m = now.hour, now.minute
            return (9, 30) <= (h, m) <= (16, 0)


# ── Module-level helpers called from flask_app.py ─────────────────────────────

_client: TradierClient = None


def init(token: str, live: bool = True):
    """Call once at startup with your Tradier token."""
    global _client
    _client = TradierClient(token, live=live)
    print(f'Tradier client initialised ({"LIVE" if live else "SANDBOX"})')


def is_ready() -> bool:
    """True if a token has been set."""
    return _client is not None


def get_best_option_tradier(ticker: str, setup: dict, config: dict) -> dict | None:
    """
    Tradier replacement for get_best_option().
    Same return shape -- the rest of the scanner doesn't need to change.
    Uses REAL Greeks instead of approximated delta.
    """
    if not _client:
        return None

    try:
        expirations = _client.get_expirations(ticker)
        if not expirations:
            return None

        today         = datetime.now()
        current_price = setup['current_price']
        direction     = setup['direction']
        opt_type      = 'call' if direction == 'BULLISH' else 'put'
        viable        = []

        for exp_date in expirations[:10]:
            dte = (datetime.strptime(exp_date, '%Y-%m-%d') - today).days
            if dte < config['dte_range'][0] or dte > config['dte_range'][1]:
                continue

            chain = _client.get_options_chain(ticker, exp_date, opt_type)
            if chain.empty:
                continue

            # Strike filter: 92%-110% of spot
            chain = chain[
                (chain['strike'] >= current_price * 0.92) &
                (chain['strike'] <= current_price * 1.10)
            ]

            for _, row in chain.iterrows():
                bid = float(row.get('bid') or 0)
                ask = float(row.get('ask') or 0)
                if bid <= 0 or ask <= 0:
                    continue

                mid_price  = (bid + ask) / 2
                spread_pct = (ask - bid) / mid_price
                if spread_pct > config['max_bid_ask_spread']:
                    continue
                if mid_price < config.get('min_premium', 0.20):
                    continue
                if int(row.get('openInterest') or 0) < config['min_open_interest']:
                    continue

                # Use REAL delta from Tradier Greeks
                delta = float(row.get('delta') or 0)
                if delta == 0:
                    # Fallback estimate if Greeks not returned
                    moneyness = current_price / float(row['strike'])
                    delta = 0.5 + (moneyness - 1) * 2
                    delta = max(0.1, min(0.9, delta))

                # For puts, delta is negative -- use abs for range check
                delta_check = abs(delta)
                if delta_check < config['delta_range'][0] or delta_check > config['delta_range'][1]:
                    continue

                theta = float(row.get('theta') or (-mid_price / max(dte, 1)))
                iv    = float(row.get('impliedVolatility') or 0)

                viable.append({
                    'strike':        float(row['strike']),
                    'expiration':    exp_date,
                    'dte':           dte,
                    'bid':           round(bid, 2),
                    'ask':           round(ask, 2),
                    'premium':       round(mid_price, 2),
                    'delta':         round(delta, 3),
                    'theta':         round(theta, 3),
                    'gamma':         round(float(row.get('gamma') or 0), 4),
                    'vega':          round(float(row.get('vega')  or 0), 4),
                    'iv':            round(iv * 100, 1) if iv else 0,   # as %
                    'open_interest': int(row.get('openInterest') or 0),
                    'volume':        int(row.get('volume') or 0),
                    'spread_pct':    round(spread_pct * 100, 1),
                    'data_source':   'tradier_live',
                })

        if not viable:
            return None

        # Sort: prefer higher delta within range, penalise theta decay
        viable.sort(key=lambda x: abs(x['delta']) - abs(x['theta']) * 0.1, reverse=True)
        return viable[0]

    except Exception as e:
        print(f'Tradier get_best_option failed for {ticker}: {e}')
        return None


def get_intraday_history(ticker: str, interval: str = '5min') -> pd.DataFrame:
    """
    Return intraday bars for a ticker.
    Used by detect_momentum_setup() during market hours.
    """
    if not _client:
        return pd.DataFrame()
    return _client.get_intraday_bars(ticker, interval=interval)


def market_open() -> bool:
    if not _client:
        # Simple time-based fallback
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        h, m = now.hour, now.minute
        return (9, 30) <= (h, m) <= (16, 0)
    return _client.is_market_open()

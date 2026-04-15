# -*- coding: utf-8 -*-
"""
webull.py -- Real-time stock data via the official Webull OpenAPI.

Implements HMAC-SHA1 request signing directly using Python stdlib — no
third-party SDK required (the official SDK is incompatible with Python 3.13+
because it bundles the removed `cgi` module and requires grpcio 1.51.1).

Signing algorithm reverse-engineered from:
  webullsdkcore/auth/composer/default_signature_composer.py
  webullsdkcore/auth/algorithm/sha_hmac1.py
  webullsdkcore/headers.py

Credentials: webull.com/center#openApiManagement → API Management → My Application
Docs:        https://developer.webull.com/apis/docs/market-data-api/data-api/
"""

import hashlib
import hmac
import base64
import uuid
import json
import socket
import requests
import pandas as pd
from datetime import datetime
from urllib.parse import quote, urlencode
import warnings
warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────
_BASE_URL  = 'https://api.webull.com'
_API_VER   = 'v1'

# Header names (from webullsdkcore/headers.py)
_H_APP_KEY  = 'x-app-key'
_H_TS       = 'x-timestamp'
_H_SIG_ALG  = 'x-signature-algorithm'
_H_SIG_VER  = 'x-signature-version'
_H_NONCE    = 'x-signature-nonce'
_H_SIG      = 'x-signature'
_H_VERSION  = 'x-version'


# ── HMAC-SHA1 signing (pure stdlib) ───────────────────────────────────────────

def _iso8601_utc() -> str:
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

def _nonce() -> str:
    name = socket.gethostname() + str(uuid.uuid1())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

def _b64_hmac_sha1(secret: str, message: str) -> str:
    key = (secret + '&').encode('utf-8')
    msg = message.encode('utf-8')
    sig = hmac.new(key, msg, hashlib.sha1).digest()
    return base64.b64encode(sig).decode('utf-8').strip()

def _sign_request(app_key: str, app_secret: str,
                  uri: str, query_params: dict,
                  body_params: dict = None) -> dict:
    """
    Build signed headers for one Webull API request.

    Algorithm (from SDK source):
    1. sign_params = {lowercase(header_name): value}  for all auth headers + host
       plus all query params
    2. Sort sign_params alphabetically by key
    3. sign_string = uri + '&' + 'k1=v1&k2=v2&...'
       If body exists: append '&' + MD5_UPPERCASE(compact_json_body)
    4. url_encode(sign_string)  using RFC 3986 (safe='')
    5. signature = base64(HMAC-SHA1(app_secret+'&', encoded_sign_string))
    """
    ts    = _iso8601_utc()
    nonce = _nonce()
    host  = 'api.webull.com'

    # Auth headers that participate in signing (x-version is sent but NOT signed)
    # Source: webullsdkcore/auth/composer/default_signature_composer.py
    # _refresh_sign_headers only adds: app_key, timestamp, sign_version, sign_algorithm, nonce, host
    sign_headers = {
        _H_APP_KEY:  app_key,
        _H_TS:       ts,
        _H_SIG_ALG:  'HMAC-SHA1',
        _H_SIG_VER:  '1.0',
        _H_NONCE:    nonce,
        'host':      host,          # host goes into sign_params but NOT sent as header
    }

    # Combine sign_headers + query_params (all lowercase keys)
    sign_params = {k.lower(): str(v) for k, v in sign_headers.items()}
    for k, v in (query_params or {}).items():
        sign_params[k.lower()] = str(v)

    # Build sign string: uri + & + sorted k=v pairs
    sorted_pairs = sorted(sign_params.items())
    kv_str = '&'.join(f'{k}={v}' for k, v in sorted_pairs)
    sign_string = f'{uri}&{kv_str}'

    # Append body MD5 if present
    if body_params:
        compact_body = json.dumps(body_params, separators=(',', ':'))
        body_md5 = hashlib.md5(compact_body.encode('utf-8')).hexdigest().upper()
        sign_string = f'{sign_string}&{body_md5}'

    # URL-encode (RFC 3986: safe='' encodes everything except unreserved chars)
    encoded = quote(sign_string, safe='')

    # HMAC-SHA1 signature
    signature = _b64_hmac_sha1(app_secret, encoded)

    # Return actual request headers (host excluded — requests adds it automatically)
    # x-version is sent with the request but was NOT included in the signature
    return {
        _H_APP_KEY:  app_key,
        _H_TS:       ts,
        _H_SIG_ALG:  'HMAC-SHA1',
        _H_SIG_VER:  '1.0',
        _H_NONCE:    nonce,
        _H_SIG:      signature,
        _H_VERSION:  _API_VER,     # sent but not signed
        'Accept':    'application/json',
        'Accept-Encoding': 'gzip',
    }


# ── WebullClient ───────────────────────────────────────────────────────────────

class WebullClient:
    """
    Webull OpenAPI client — no SDK required, Python 3.14 compatible.

    Credentials come from the Webull developer portal:
      webull.com/center#openApiManagement → API Management → My Application
    """

    def __init__(self, app_key: str, app_secret: str):
        self.app_key    = app_key
        self.app_secret = app_secret
        self._session   = requests.Session()

    def _get(self, path: str, params: dict) -> dict:
        """Make a signed GET request and return parsed JSON."""
        uri     = path                             # e.g. /market-data/snapshot
        headers = _sign_request(self.app_key, self.app_secret, uri, params)
        url     = f'{_BASE_URL}{path}'
        resp    = self._session.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ── QUOTE ─────────────────────────────────────────────────────────────────

    def get_quote(self, ticker: str) -> dict:
        """Real-time snapshot for one ticker."""
        try:
            data = self._get('/market-data/snapshot', {
                'symbols':  ticker,
                'category': 'US_STOCK',
            })
            items = data if isinstance(data, list) else ([data] if data else [])
            q = next((x for x in items if x.get('symbol') == ticker), items[0] if items else {})
            return {
                'ticker':     ticker,
                'price':      float(q.get('price') or q.get('close') or 0),
                'change':     float(q.get('change') or 0),
                'change_pct': float(q.get('change_ratio') or 0),
                'volume':     int(float(q.get('volume') or 0)),
            }
        except Exception as e:
            return {'ticker': ticker, 'price': 0, 'change': 0,
                    'change_pct': 0, 'volume': 0, 'error': str(e)}

    # ── INTRADAY BARS ─────────────────────────────────────────────────────────

    def get_intraday_bars(self, ticker: str, interval: str = 'M5',
                          count: int = 200) -> pd.DataFrame:
        """
        Intraday OHLCV bars.
        interval: 'M1' | 'M5' | 'M15' | 'M30' | 'M60'
        """
        try:
            data = self._get('/market-data/bars', {
                'symbol':   ticker,
                'category': 'US_STOCK',
                'timespan': interval,
                'count':    str(min(count, 1200)),
            })
            items = data if isinstance(data, list) else []
            if not items:
                return pd.DataFrame()

            df = pd.DataFrame(items)

            # Column name normalisation (SDK uses various casings)
            col_map = {
                'time': 'time', 't': 'time',
                'open': 'Open', 'o': 'Open',
                'high': 'High', 'h': 'High',
                'low':  'Low',  'l': 'Low',
                'close':'Close','c': 'Close',
                'volume':'Volume','v':'Volume',
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if 'time' not in df.columns:
                return pd.DataFrame()

            # Parse timestamps (Unix ms or ISO string)
            if pd.api.types.is_numeric_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
            else:
                df['time'] = pd.to_datetime(df['time'], utc=True)

            df = df.set_index('time').sort_index()
            needed = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in needed if c not in df.columns]
            if missing:
                return pd.DataFrame()

            return df[needed].apply(pd.to_numeric, errors='coerce').dropna()
        except Exception as e:
            print(f'Webull intraday bars failed for {ticker}: {e}')
            return pd.DataFrame()

    # ── MARKET STATUS ─────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        h, m = now.hour, now.minute
        return (9, 30) <= (h, m) <= (16, 0)


# ── Module-level state + helpers ──────────────────────────────────────────────

_client: WebullClient = None


def init(app_key: str, app_secret: str, region: str = 'us'):
    """Call once at startup with your Webull credentials."""
    global _client
    _client = WebullClient(app_key, app_secret)
    print(f'Webull client initialised (stdlib signing, Python 3.14 compatible)')


def is_ready() -> bool:
    return _client is not None


def market_open() -> bool:
    if _client:
        return _client.is_market_open()
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    return (9, 30) <= (h, m) <= (16, 0)


def get_live_quote(ticker: str) -> dict:
    if not _client:
        return {}
    return _client.get_quote(ticker)


def get_intraday_history(ticker: str, interval: str = 'M5') -> pd.DataFrame:
    if not _client:
        return pd.DataFrame()
    return _client.get_intraday_bars(ticker, interval=interval)


def enrich_with_live_bar(ticker: str, hist: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the last daily bar in `hist` with a live intraday aggregate.
    Called from fetch_ticker_history() during market hours when Webull is ready.
    """
    if not _client or not _client.is_market_open():
        return hist
    try:
        intraday = _client.get_intraday_bars(ticker, interval='M5', count=200)
        if intraday.empty:
            return hist

        today_utc = pd.Timestamp.utcnow().normalize()
        today_bars = intraday[intraday.index.normalize() >= today_utc]
        if today_bars.empty:
            return hist

        live_bar = pd.DataFrame([{
            'Open':   float(today_bars['Open'].iloc[0]),
            'High':   float(today_bars['High'].max()),
            'Low':    float(today_bars['Low'].min()),
            'Close':  float(today_bars['Close'].iloc[-1]),
            'Volume': int(today_bars['Volume'].sum()),
        }], index=[today_bars.index[-1]])

        hist_copy = hist.copy()
        if len(hist_copy) > 0:
            last = hist_copy.index[-1]
            last_utc = (last.tz_convert('UTC') if getattr(last, 'tz', None)
                        else pd.Timestamp(last, tz='UTC')).normalize()
            if last_utc >= today_utc:
                hist_copy = hist_copy.iloc[:-1]

        return pd.concat([hist_copy, live_bar])
    except Exception as e:
        print(f'Webull live bar enrichment failed for {ticker}: {e}')
        return hist

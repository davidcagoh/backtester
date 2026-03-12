"""
Lightweight web app for natural-language backtesting on local CSV data.

Run with: python main.py
Then open http://localhost:8000
"""

import json
import math
import os
import re
import textwrap
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from google import genai
from google.genai import types as genai_types
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PORT = 8000

# Load .env if present
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

GEMINI_KEY = os.getenv("GEMINI_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ----------------------------
# Data utilities
# ----------------------------


def available_datasets() -> Dict[str, Path]:
    """Return mapping of dataset key -> csv path (stem = key)."""
    return {p.stem: p for p in DATA_DIR.glob("*.csv")}


def load_price_data(symbol: str) -> pd.DataFrame:
    datasets = available_datasets()
    if symbol not in datasets:
        raise ValueError(f"Unknown dataset '{symbol}'. Options: {', '.join(datasets)}")

    df = pd.read_csv(datasets[symbol])
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if "close" not in df.columns:
        raise ValueError("CSV must contain a 'Close' column")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "close"]].dropna()


def dataset_summary(symbol: str) -> Dict:
    df = load_price_data(symbol)
    return {
        "symbol": symbol,
        "rows": len(df),
        "start": df["date"].iloc[0].date().isoformat(),
        "end": df["date"].iloc[-1].date().isoformat(),
        "min_close": round(float(df["close"].min()), 4),
        "max_close": round(float(df["close"].max()), 4),
    }


# ----------------------------
# Strategy building
# ----------------------------


@dataclass
class StrategySpec:
    name: str
    code_preview: str
    explanation: str
    builder: Callable[[pd.DataFrame], pd.DataFrame]


def moving_average_strategy(fast: int, slow: int) -> StrategySpec:
    def build(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["fast"] = data["close"].rolling(fast, min_periods=1).mean()
        data["slow"] = data["close"].rolling(slow, min_periods=1).mean()
        data["signal"] = 0
        data.loc[data["fast"].shift(1) < data["slow"].shift(1), "cross_up"] = (
            data["fast"] >= data["slow"]
        )
        data.loc[data["fast"].shift(1) > data["slow"].shift(1), "cross_down"] = (
            data["fast"] <= data["slow"]
        )
        data.loc[data["cross_up"] == True, "signal"] = 1
        data.loc[data["cross_down"] == True, "signal"] = -1
        return data

    code = f"""
    # Moving-average crossover
    fast = {fast}
    slow = {slow}
    df['fast'] = df['Close'].rolling(fast, min_periods=1).mean()
    df['slow'] = df['Close'].rolling(slow, min_periods=1).mean()
    df['signal'] = 0
    df.loc[df['fast'].shift(1) < df['slow'].shift(1) & (df['fast'] >= df['slow']), 'signal'] = 1
    df.loc[df['fast'].shift(1) > df['slow'].shift(1) & (df['fast'] <= df['slow']), 'signal'] = -1
    """
    explanation = (
        f"Buy when the {fast}-day average crosses above the {slow}-day average; "
        f"sell when it crosses back below."
    )
    return StrategySpec(
        name=f"MA crossover {fast}/{slow}", code_preview=textwrap.dedent(code).strip(), explanation=explanation, builder=build
    )


def bounce_strategy(pct: float, lookback: int = 60, stop_pct: float = 0.1, time_stop: int = 90) -> StrategySpec:
    def build(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["low"] = data["close"].rolling(lookback, min_periods=1).min()
        trigger = (1 + pct / 100)
        data["signal"] = 0
        data.loc[data["close"] >= data["low"] * trigger, "signal"] = 1

        # trailing stop once in trade
        data["peak"] = data["close"].cummax()
        data.loc[data["close"] <= data["peak"] * (1 - stop_pct), "signal"] = -1

        # time-based exit
        data["days_in_trade"] = 0
        in_pos = False
        counter = 0
        signals = []
        for sig in data["signal"].tolist():
            if sig == 1 and not in_pos:
                in_pos = True
                counter = 0
            elif in_pos:
                counter += 1
                if counter >= time_stop:
                    sig = -1
                    in_pos = False
            signals.append(sig)
            if sig == -1:
                in_pos = False
                counter = 0
        data["signal"] = signals
        return data

    code = f"""
    # "Buy X% off the bottom" bounce idea
    lookback = {lookback}
    bounce = {pct}%
    stop_loss = {stop_pct*100}% trailing from peak
    time_stop = {time_stop} days
    df['low'] = df['Close'].rolling(lookback, min_periods=1).min()
    df['signal'] = (df['Close'] >= df['low'] * (1 + bounce/100)).astype(int)
    # trailing stop & time stop applied in simulator
    """
    explanation = (
        f"Look back {lookback} days, buy once price is {pct}% above that window's low. "
        f"Exit if price falls {stop_pct*100:.0f}% from the post-entry peak or after {time_stop} days."
    )
    return StrategySpec(
        name=f"Bounce +{pct}%", code_preview=textwrap.dedent(code).strip(), explanation=explanation, builder=build
    )


def breakout_strategy(window: int = 50, stop_pct: float = 0.08) -> StrategySpec:
    def build(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["high"] = data["close"].rolling(window, min_periods=1).max()
        data["signal"] = 0
        data.loc[data["close"] > data["high"].shift(1), "signal"] = 1
        data.loc[data["close"] < data["high"] * (1 - stop_pct), "signal"] = -1
        return data

    code = f"""
    # Breakout strategy
    window = {window}
    stop_loss = {stop_pct*100}% below breakout high
    df['high'] = df['Close'].rolling(window, min_periods=1).max()
    df['signal'] = (df['Close'] > df['high'].shift(1)).astype(int)
    df.loc[df['Close'] < df['high'] * (1 - {stop_pct}), 'signal'] = -1
    """
    explanation = (
        f"Buy on new {window}-day highs; exit if price falls {stop_pct*100:.0f}% under that breakout level."
    )
    return StrategySpec(
        name=f"{window}-day breakout", code_preview=textwrap.dedent(code).strip(), explanation=explanation, builder=build
    )


def buy_and_hold_strategy() -> StrategySpec:
    def build(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["signal"] = 0
        if len(data) > 0:
            data.loc[data.index[0], "signal"] = 1
            data.loc[data.index[-1], "signal"] = -1
        return data

    code = """
    # Buy the first close, sell the last close
    df['signal'] = 0
    df.loc[df.index[0], 'signal'] = 1
    df.loc[df.index[-1], 'signal'] = -1
    """
    explanation = "Simple buy-and-hold across the whole sample."
    return StrategySpec(
        name="Buy and hold", code_preview=textwrap.dedent(code).strip(), explanation=explanation, builder=build
    )


def _gemini_client():
    if not GEMINI_KEY:
        return None
    return genai.Client(api_key=GEMINI_KEY)


LLM_SYSTEM_PROMPT = """
You are a cautious trading-strategy compiler. Given a natural-language description, you produce a Python function `build(df)` that operates on a pandas DataFrame with columns: 'date', 'close'.
Rules:
- `pd` (pandas) is already available in scope. Do NOT write any import statements.
- Only use pandas/vectorized operations; no external data, no network, no file I/O, no randomness.
- Add a column `signal` to df with 1 for buy, -1 for sell/exit, 0 otherwise, and return df.
- Do not use while/for loops except simple cumulative counters if unavoidable.
- Never call eval/exec.
- Keep code concise and deterministic.
Return JSON with keys: name (string), explanation (1-2 sentences), code (the full function including def build(df): ...).
"""


def build_from_code(code: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    # Strip any bare import lines the LLM may have emitted
    cleaned = "\n".join(
        line for line in code.splitlines()
        if not re.match(r"^\s*(import |from \S+ import )", line)
    )
    local_vars: Dict = {}
    safe_builtins = {"min": min, "max": max, "abs": abs, "len": len, "range": range, "int": int, "float": float, "round": round, "zip": zip, "enumerate": enumerate}
    try:
        exec(cleaned, {"pd": pd, "__builtins__": safe_builtins}, local_vars)
    except Exception as exc:
        raise ValueError(f"Generated code failed to compile: {exc}")
    if "build" not in local_vars:
        raise ValueError("Generated code must define build(df)")
    return local_vars["build"]


def llm_strategy(text: str) -> tuple:
    """Returns (StrategySpec, {"in": int, "out": int})."""
    client = _gemini_client()
    if not client:
        raise RuntimeError("GEMINI_KEY not set; cannot call LLM")

    prompt = f'Natural language request: "{text}"\nProduce the JSON.'
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=LLM_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=16384,
        ),
    )
    content = resp.text
    usage = {
        "in": getattr(resp.usage_metadata, "prompt_token_count", 0) or 0,
        "out": getattr(resp.usage_metadata, "candidates_token_count", 0) or 0,
    }

    try:
        parsed = json.loads(content)
        code = parsed.get("code", "")
        name = parsed.get("name", "Custom strategy")
        explanation = parsed.get("explanation", "")
    except Exception as exc:
        raise ValueError(f"LLM response parse error: {exc}\nRaw: {content[:300]}")

    builder = build_from_code(code)
    return StrategySpec(name=name, code_preview=code, explanation=explanation, builder=builder), usage


def heuristic_strategy(text: str) -> StrategySpec:
    # previous rule-based fallback
    if not text:
        return buy_and_hold_strategy()
    cleaned = text.lower()
    ma_numbers = re.findall(r"(\d+)\s*(?:day)?\s*(?:ma|ema|moving average)", cleaned)
    if len(ma_numbers) >= 2:
        fast, slow = sorted(int(n) for n in ma_numbers[:2])
        if fast == slow:
            slow = fast * 2
        return moving_average_strategy(fast, slow)
    bounce_match = re.search(r"(\d+\.?\d*)%[^.]*bottom|bounce", cleaned)
    if bounce_match:
        pct = float(bounce_match.group(1)) if bounce_match.group(1) else 20.0
        return bounce_strategy(pct)
    breakout_match = re.search(r"breakout|momentum|high", cleaned)
    if breakout_match:
        return breakout_strategy()
    dip_match = re.search(r"dip|pullback", cleaned)
    if dip_match:
        return bounce_strategy(10.0, lookback=30, stop_pct=0.07, time_stop=45)
    return buy_and_hold_strategy()


def interpret_strategy(text: str) -> tuple:
    """Returns (StrategySpec, usage_dict)."""
    if GEMINI_KEY:
        return llm_strategy(text)
    return heuristic_strategy(text), {"in": 0, "out": 0}


# ----------------------------
# Backtest engine
# ----------------------------


def simulate(df: pd.DataFrame, spec: StrategySpec) -> Dict:
    data = spec.builder(df)
    if "signal" not in data.columns:
        raise ValueError("Strategy builder must add a 'signal' column")

    cash = 1.0
    shares = 0.0
    equity_curve = []
    trades: List[Dict] = []

    signals = data["signal"].fillna(0).tolist()
    closes = data["close"].tolist()
    dates = data["date"].tolist()

    for idx, (sig, price, date) in enumerate(zip(signals, closes, dates)):
        if sig == 1 and shares == 0:
            shares = cash / price
            cash = 0.0
            trades.append({"action": "BUY", "date": date.date().isoformat(), "price": round(price, 4)})
        elif sig == -1 and shares > 0:
            cash = shares * price
            shares = 0.0
            trades.append({"action": "SELL", "date": date.date().isoformat(), "price": round(price, 4)})
        equity_curve.append(cash + shares * price)

    # Liquidate at end if still holding
    if shares > 0:
        cash = shares * closes[-1]
        trades.append({"action": "SELL", "date": dates[-1].date().isoformat(), "price": round(closes[-1], 4)})
        shares = 0.0
    final_equity = cash
    equity_curve[-1] = final_equity

    total_return = (final_equity - 1) * 100
    days = (dates[-1] - dates[0]).days or 1
    cagr = (final_equity) ** (365 / days) - 1

    # max drawdown
    peak = -1
    max_dd = 0
    for val in equity_curve:
        peak = val if peak < 0 else max(peak, val)
        dd = (val - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    returns = pd.Series(equity_curve).pct_change().dropna()
    vol = returns.std() * math.sqrt(252)
    sharpe = returns.mean() / vol * math.sqrt(252) if vol > 0 else 0

    win_trades = 0
    realized_trades = []
    for i in range(0, len(trades) - 1, 2):
        buy = trades[i]
        sell = trades[i + 1] if i + 1 < len(trades) else None
        if sell:
            pnl_pct = (sell["price"] - buy["price"]) / buy["price"] * 100
            win_trades += int(pnl_pct > 0)
            realized_trades.append({
                "entry": buy,
                "exit": sell,
                "pnl_pct": round(pnl_pct, 2),
            })

    win_rate = (win_trades / len(realized_trades)) * 100 if realized_trades else 0

    return {
        "strategy": spec.name,
        "code": spec.code_preview,
        "explanation": spec.explanation,
        "metrics": {
            "total_return_pct": round(total_return, 2),
            "cagr_pct": round(cagr * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe": round(sharpe, 2),
            "trades": len(trades),
            "wins": int(win_trades),
            "win_rate_pct": round(win_rate, 2),
            "period_days": days,
        },
        "trades": realized_trades[:50],
        "equity_curve": equity_curve[-500:],  # trim to keep payload small
    }


# ----------------------------
# HTTP layer
# ----------------------------


def json_response(handler: BaseHTTPRequestHandler, payload: Dict, status: int = 200):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parsed.path == "/api/datasets":
            data = [dataset_summary(sym) for sym in available_datasets().keys()]
            json_response(self, {"datasets": data})
            return

        if parsed.path == "/api/dataset_info":
            params = parse_qs(parsed.query)
            symbol = params.get("symbol", [None])[0]
            if not symbol:
                json_response(self, {"error": "symbol required"}, status=400)
                return
            try:
                json_response(self, dataset_summary(symbol))
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=400)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            json_response(self, {"error": "Invalid JSON"}, status=400)
            return

        if parsed.path == "/api/generate":
            description = payload.get("description", "")
            try:
                spec, usage = interpret_strategy(description)
                json_response(
                    self,
                    {
                        "strategy": spec.name,
                        "code": spec.code_preview,
                        "explanation": spec.explanation,
                        "llm": bool(GEMINI_KEY),
                        "usage": usage,
                    },
                )
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/run":
            symbols = payload.get("symbols") or []
            description = payload.get("description", "")
            date_from = payload.get("date_from") or None
            date_to = payload.get("date_to") or None
            if not symbols:
                json_response(self, {"error": "at least one symbol required"}, status=400)
                return
            try:
                spec, usage = interpret_strategy(description)
                results = []
                for sym in symbols:
                    df = load_price_data(sym)
                    if date_from:
                        df = df[df["date"] >= pd.Timestamp(date_from)]
                    if date_to:
                        df = df[df["date"] <= pd.Timestamp(date_to)]
                    df = df.reset_index(drop=True)
                    if len(df) < 2:
                        raise ValueError(f"{sym}: date range produced fewer than 2 rows")
                    res = simulate(df, spec)
                    res["symbol"] = sym
                    results.append(res)
                json_response(self, {"strategy": spec.name, "results": results, "code": spec.code_preview, "explanation": spec.explanation, "usage": usage})
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=400)
            return

        json_response(self, {"error": "Not found"}, status=404)


def run_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Backtester UI running on http://localhost:{PORT}")
    server.serve_forever()


# ----------------------------
# Front-end
# ----------------------------


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Backtester</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0d1117;
      --surface:  #161b22;
      --border:   #30363d;
      --muted:    #8b949e;
      --text:     #e6edf3;
      --accent:   #3fb950;
      --accent-d: #238636;
      --red:      #f85149;
      --yellow:   #e3b341;
      --blue:     #58a6ff;
      --radius:   10px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      font-size: 14px;
      line-height: 1.5;
    }

    body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }

    /* ── Token bar ── */
    .token-bar {
      display: flex; align-items: center; gap: 16px;
      padding: 6px 24px;
      background: #0d1117;
      border-bottom: 1px solid var(--border);
      font-size: 12px; color: var(--muted);
    }
    .token-bar span { color: var(--text); font-variant-numeric: tabular-nums; }
    .token-bar .sep { color: var(--border); }

    /* ── Date row ── */
    .date-row { display: flex; gap: 10px; margin-top: 10px; align-items: center; flex-wrap: wrap; }
    .date-row label { font-size: 12px; color: var(--muted); white-space: nowrap; }
    input[type=date] {
      background: var(--bg); color: var(--text);
      border: 1px solid var(--border); border-radius: var(--radius);
      padding: 6px 10px; font-size: 13px; font-family: inherit;
      outline: none; transition: border-color 0.15s;
    }
    input[type=date]:focus { border-color: var(--blue); }

    /* ── Header ── */
    header {
      display: flex; align-items: center; gap: 12px;
      padding: 14px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
    }
    header svg { flex-shrink: 0; }
    header h1 { font-size: 16px; font-weight: 600; letter-spacing: 0.2px; }
    header .badge {
      margin-left: auto; font-size: 11px; padding: 2px 8px;
      border-radius: 20px; background: #1f2b1f; color: var(--accent);
      border: 1px solid var(--accent-d);
    }

    /* ── Layout ── */
    .layout {
      display: grid;
      grid-template-columns: 280px 1fr;
      flex: 1;
    }

    /* ── Sidebar ── */
    .sidebar {
      background: var(--surface);
      border-right: 1px solid var(--border);
      padding: 20px 16px;
      display: flex; flex-direction: column; gap: 20px;
    }
    .sidebar-title {
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.8px; color: var(--muted); margin-bottom: 10px;
    }

    .ds-item {
      display: flex; align-items: center; gap: 10px;
      padding: 8px 10px; border-radius: var(--radius);
      cursor: pointer; transition: background 0.15s;
      border: 1px solid transparent;
    }
    .ds-item:hover { background: #1c2128; }
    .ds-item.selected { background: #1a2f1a; border-color: var(--accent-d); }
    .ds-item input[type=checkbox] { accent-color: var(--accent); width: 14px; height: 14px; cursor: pointer; }
    .ds-symbol { font-weight: 700; font-size: 13px; }
    .ds-meta { font-size: 11px; color: var(--muted); }

    /* ── Main ── */
    .main { padding: 24px; display: flex; flex-direction: column; gap: 20px; overflow-y: auto; }

    /* ── Cards ── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
    }
    .card-title {
      font-size: 12px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.7px; color: var(--muted); margin-bottom: 14px;
    }

    /* ── Textarea / inputs ── */
    textarea {
      width: 100%; resize: vertical;
      min-height: 90px;
      background: var(--bg); color: var(--text);
      border: 1px solid var(--border); border-radius: var(--radius);
      padding: 10px 12px; font-size: 14px; font-family: inherit;
      transition: border-color 0.15s;
      outline: none;
    }
    textarea:focus { border-color: var(--blue); }

    /* ── Buttons ── */
    .btn-row { display: flex; gap: 8px; margin-top: 12px; }
    button {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 8px 16px; border: none; border-radius: var(--radius);
      cursor: pointer; font-size: 13px; font-weight: 600;
      transition: opacity 0.15s, filter 0.15s;
    }
    button:hover { filter: brightness(1.1); }
    button:active { filter: brightness(0.9); }
    button:disabled { opacity: 0.45; cursor: default; filter: none; }
    .btn-primary { background: var(--accent); color: #0d1117; }
    .btn-secondary { background: #21262d; color: var(--text); border: 1px solid var(--border); }

    /* ── Spinner ── */
    @keyframes spin { to { transform: rotate(360deg); } }
    .spinner { width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.2); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; }

    /* ── Metric grid ── */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .metric {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 14px;
    }
    .metric-label { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
    .metric-value { font-size: 20px; font-weight: 700; }
    .pos { color: var(--accent); }
    .neg { color: var(--red); }
    .neu { color: var(--text); }

    /* ── Strategy preview ── */
    .explanation { color: var(--muted); font-size: 13px; margin-bottom: 14px; }
    pre {
      background: var(--bg); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 14px 16px;
      overflow-x: auto; font-size: 12px; line-height: 1.6;
      font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace;
      color: #c9d1d9;
    }

    /* ── Trade table ── */
    details { margin-top: 16px; }
    summary { cursor: pointer; font-size: 13px; color: var(--blue); user-select: none; margin-bottom: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { color: var(--muted); font-weight: 600; text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
    td { padding: 7px 8px; border-bottom: 1px solid #21262d; }
    tr:last-child td { border-bottom: none; }
    .win { color: var(--accent); }
    .loss { color: var(--red); }

    /* ── Error ── */
    .error-box { background: #2d1515; border: 1px solid #5a1d1d; border-radius: var(--radius); padding: 14px 16px; color: var(--red); font-size: 13px; }

    /* ── Chart ── */
    .chart-wrap { position: relative; height: 220px; margin-top: 16px; }

    /* ── Result symbol header ── */
    .result-header { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
    .result-symbol { font-size: 18px; font-weight: 700; }
    .result-sub { font-size: 12px; color: var(--muted); }

    /* ── Hidden ── */
    .hidden { display: none !important; }
  </style>
</head>
<body>

<header>
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
  <h1>Backtester</h1>
  <span class="badge" id="llm-badge">Gemini</span>
</header>

<div class="token-bar" id="token-bar" style="display:none;">
  Gemini tokens this session &nbsp;|&nbsp;
  In: <span id="tok-in">0</span> &nbsp;<span class="sep">|</span>&nbsp;
  Out: <span id="tok-out">0</span> &nbsp;<span class="sep">|</span>&nbsp;
  Total: <span id="tok-total">0</span>
</div>

<div class="layout">
  <!-- Sidebar -->
  <nav class="sidebar">
    <div>
      <div class="sidebar-title">Datasets</div>
      <div id="datasets"></div>
    </div>
    <div id="dataset-info-panel" class="hidden">
      <div class="sidebar-title">Info</div>
      <div id="dataset-info-content"></div>
    </div>
  </nav>

  <!-- Main -->
  <div class="main">

    <!-- Input card -->
    <div class="card">
      <div class="card-title">Strategy</div>
      <textarea id="strategy-text" placeholder="Describe your strategy in plain English — e.g. &quot;Buy when the 10-day MA crosses above the 50-day MA, sell when it crosses back below&quot;"></textarea>
      <div class="date-row">
        <label>From</label>
        <input type="date" id="date-from" />
        <label>To</label>
        <input type="date" id="date-to" />
        <button class="btn-secondary" style="padding:6px 10px;font-size:12px;margin-top:0;" onclick="clearDates()">Clear</button>
      </div>
      <div class="btn-row">
        <button class="btn-secondary" id="btn-translate" onclick="generateStrategy()">
          <span>Translate to code</span>
        </button>
        <button class="btn-primary" id="btn-run" onclick="runBacktest()">
          <span>Run backtest</span>
        </button>
      </div>
    </div>

    <!-- Generated strategy preview -->
    <div class="card hidden" id="generated-card">
      <div class="card-title">Generated strategy</div>
      <div class="explanation" id="generated-explanation"></div>
      <pre id="generated-code"></pre>
    </div>

    <!-- Results -->
    <div id="results-area"></div>

  </div>
</div>

<script>
  // ── Helpers ──────────────────────────────────────────────────────────────────

  async function fetchJSON(url, options = {}) {
    const res = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options });
    return res.json();
  }

  function setLoading(btn, loading) {
    if (loading) {
      btn.disabled = true;
      btn._prev = btn.innerHTML;
      btn.innerHTML = '<span class="spinner"></span><span>Working…</span>';
    } else {
      btn.disabled = false;
      btn.innerHTML = btn._prev;
    }
  }

  // ── Token tracking ────────────────────────────────────────────────────────────

  const _tok = { in: 0, out: 0 };

  function addTokens(usage) {
    if (!usage) return;
    _tok.in  += usage.in  || 0;
    _tok.out += usage.out || 0;
    document.getElementById('tok-in').textContent    = _tok.in.toLocaleString();
    document.getElementById('tok-out').textContent   = _tok.out.toLocaleString();
    document.getElementById('tok-total').textContent = (_tok.in + _tok.out).toLocaleString();
    document.getElementById('token-bar').style.display = 'flex';
  }

  // ── Date helpers ──────────────────────────────────────────────────────────────

  function clearDates() {
    document.getElementById('date-from').value = '';
    document.getElementById('date-to').value   = '';
  }

  function metricColor(label, value) {
    const v = parseFloat(value);
    if (isNaN(v)) return 'neu';
    if (label.toLowerCase().includes('drawdown')) return v < 0 ? 'neg' : 'neu';
    if (label.toLowerCase().includes('return') || label.toLowerCase().includes('cagr') || label.toLowerCase().includes('sharpe')) {
      return v > 0 ? 'pos' : v < 0 ? 'neg' : 'neu';
    }
    return 'neu';
  }

  // ── Datasets ─────────────────────────────────────────────────────────────────

  async function loadDatasets() {
    const data = await fetchJSON('/api/datasets');
    const list = document.getElementById('datasets');
    list.innerHTML = '';
    data.datasets.forEach((ds, idx) => {
      const el = document.createElement('label');
      el.className = 'ds-item' + (idx === 0 ? ' selected' : '');
      el.innerHTML = `
        <input type="checkbox" value="${ds.symbol}" ${idx === 0 ? 'checked' : ''} onchange="onDatasetChange(this)" />
        <div>
          <div class="ds-symbol">${ds.symbol}</div>
          <div class="ds-meta">${ds.start} → ${ds.end}</div>
        </div>`;
      list.appendChild(el);
    });
    refreshDatasetInfo();
  }

  function onDatasetChange(cb) {
    const label = cb.closest('.ds-item');
    if (cb.checked) label.classList.add('selected');
    else label.classList.remove('selected');
    refreshDatasetInfo();
  }

  function selectedSymbols() {
    return Array.from(document.querySelectorAll('#datasets input[type=checkbox]:checked')).map(el => el.value);
  }

  async function refreshDatasetInfo() {
    const symbols = selectedSymbols();
    const panel = document.getElementById('dataset-info-panel');
    const content = document.getElementById('dataset-info-content');
    if (!symbols.length) { panel.classList.add('hidden'); return; }
    const infoList = await Promise.all(symbols.map(s => fetchJSON('/api/dataset_info?symbol=' + s)));
    content.innerHTML = infoList.map(info => `
      <div style="margin-bottom:12px;font-size:12px;">
        <div style="font-weight:700;margin-bottom:4px;">${info.symbol}</div>
        <div style="color:var(--muted);">${info.rows} rows</div>
        <div style="color:var(--muted);">$${info.min_close} – $${info.max_close}</div>
      </div>`).join('');
    panel.classList.remove('hidden');
  }

  // ── Generate ──────────────────────────────────────────────────────────────────

  async function generateStrategy() {
    const btn = document.getElementById('btn-translate');
    const description = document.getElementById('strategy-text').value.trim();
    if (!description) return;
    setLoading(btn, true);
    const data = await fetchJSON('/api/generate', { method: 'POST', body: JSON.stringify({ description }) });
    setLoading(btn, false);
    addTokens(data.usage);
    const card = document.getElementById('generated-card');
    if (data.error) {
      card.classList.remove('hidden');
      card.innerHTML = `<div class="card-title">Error</div><div class="error-box">${data.error}</div>`;
      return;
    }
    card.classList.remove('hidden');
    card.innerHTML = `
      <div class="card-title">${escHtml(data.strategy)}</div>
      <div class="explanation">${escHtml(data.explanation)}</div>
      <pre>${escHtml(data.code)}</pre>`;
  }

  // ── Run ───────────────────────────────────────────────────────────────────────

  let charts = [];

  async function runBacktest() {
    const btn = document.getElementById('btn-run');
    const symbols = selectedSymbols();
    if (!symbols.length) { alert('Select at least one dataset.'); return; }
    const description = document.getElementById('strategy-text').value.trim();
    const date_from = document.getElementById('date-from').value || null;
    const date_to   = document.getElementById('date-to').value   || null;
    setLoading(btn, true);
    const res = await fetchJSON('/api/run', { method: 'POST', body: JSON.stringify({ symbols, description, date_from, date_to }) });
    setLoading(btn, false);
    addTokens(res.usage);

    // destroy old charts
    charts.forEach(c => c.destroy());
    charts = [];

    const area = document.getElementById('results-area');

    if (res.error) {
      area.innerHTML = `<div class="error-box">${escHtml(res.error)}</div>`;
      return;
    }

    // strategy header card
    let html = `
      <div class="card">
        <div class="card-title">Results — ${escHtml(res.strategy)}</div>
        <div class="explanation">${escHtml(res.explanation)}</div>
      </div>`;

    area.innerHTML = html;

    res.results.forEach((r, i) => {
      const m = r.metrics;
      const canvasId = 'chart-' + i;

      const metricsHtml = [
        ['Total return', m.total_return_pct + '%'],
        ['CAGR', m.cagr_pct + '%'],
        ['Max drawdown', m.max_drawdown_pct + '%'],
        ['Sharpe', m.sharpe],
        ['Trades', m.trades],
        ['Win rate', m.win_rate_pct + '%'],
        ['Period', m.period_days + 'd'],
      ].map(([label, value]) => `
        <div class="metric">
          <div class="metric-label">${label}</div>
          <div class="metric-value ${metricColor(label, value)}">${value}</div>
        </div>`).join('');

      const tradeRows = (r.trades || []).map(t => `
        <tr>
          <td>${t.entry.date}</td>
          <td>$${t.entry.price}</td>
          <td>${t.exit.date}</td>
          <td>$${t.exit.price}</td>
          <td class="${t.pnl_pct >= 0 ? 'win' : 'loss'}">${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct}%</td>
        </tr>`).join('');

      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <div class="result-header">
          <span class="result-symbol">${r.symbol}</span>
          <span class="result-sub">${m.period_days} days</span>
        </div>
        <div class="metrics-grid">${metricsHtml}</div>
        <div class="chart-wrap"><canvas id="${canvasId}"></canvas></div>
        <details>
          <summary>${(r.trades || []).length} round-trip trades</summary>
          <table>
            <thead><tr><th>Entry</th><th>Entry px</th><th>Exit</th><th>Exit px</th><th>P&amp;L</th></tr></thead>
            <tbody>${tradeRows || '<tr><td colspan="5" style="color:var(--muted)">No completed round trips</td></tr>'}</tbody>
          </table>
        </details>`;
      area.appendChild(card);

      // draw chart after DOM insertion
      const ctx = document.getElementById(canvasId).getContext('2d');
      const curve = r.equity_curve || [];
      const isPositive = curve.length > 1 && curve[curve.length - 1] >= curve[0];
      const color = isPositive ? '#3fb950' : '#f85149';
      const chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: curve.map((_, i) => i),
          datasets: [{
            data: curve,
            borderColor: color,
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            backgroundColor: (ctx) => {
              const g = ctx.chart.ctx.createLinearGradient(0, 0, 0, 220);
              g.addColorStop(0, color + '33');
              g.addColorStop(1, color + '00');
              return g;
            },
            tension: 0.3,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: {
            callbacks: { label: ctx => ' ' + (ctx.parsed.y * 100 - 100).toFixed(1) + '% equity' }
          }},
          scales: {
            x: { display: false },
            y: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', font: { size: 11 } } },
          },
        },
      });
      charts.push(chart);
    });

    // Generated code accordion at bottom
    const codeCard = document.createElement('div');
    codeCard.className = 'card';
    codeCard.innerHTML = `
      <details>
        <summary style="color:var(--muted);font-size:12px;">View generated code</summary>
        <pre style="margin-top:12px;">${escHtml(res.code)}</pre>
      </details>`;
    area.appendChild(codeCard);
  }

  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  loadDatasets();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    run_server()

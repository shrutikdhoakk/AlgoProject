# backtest_swing.py
# ---------------------------------------------------------------------
# Daily backtest for a breakout + momentum swing strategy:
# - Universe from CSV (e.g., NIFTY500.csv), cleaned of junk tickers
# - Rank by breakout strength + ADX + RSI
# - Max positions & max invested cap
# - ATR-based stops & trailing; exits on stop/SMA20/RSI<30
# - Optional regime filter using ^NSEI EMA50 > EMA200
# - Batched yfinance downloads for speed with large universes (e.g., 500)
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


# ----------------------- helpers & indicators -------------------------

def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def _first_numeric_column(df: pd.DataFrame) -> pd.Series:
    num = df.select_dtypes(include=["number"])
    if num.shape[1] == 0:
        num = df.apply(pd.to_numeric, errors="coerce")
    return num.iloc[:, 0]

def _pick_col(df: pd.DataFrame, name: str) -> pd.Series:
    lname = name.lower()
    if not isinstance(df.columns, pd.MultiIndex):
        if name in df.columns:
            return _to_float_series(df[name]).dropna()
        lowmap = {str(c).lower(): c for c in df.columns}
        if lname in lowmap:
            return _to_float_series(df[lowmap[lname]]).dropna()
        return _to_float_series(_first_numeric_column(df)).dropna()
    # MultiIndex handling
    if lname in [str(x).lower() for x in df.columns.get_level_values(0)]:
        sub = df.xs(key=name, axis=1, level=0, drop_level=False)
        s = sub.iloc[:, 0]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return _to_float_series(s).dropna()
    for col in df.columns:
        parts = col if isinstance(col, tuple) else (col,)
        if any(str(p).lower() == lname for p in parts):
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return _to_float_series(s).dropna()
    return _to_float_series(_first_numeric_column(df)).dropna()

def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    high = _pick_col(df, "High").rename("High")
    low = _pick_col(df, "Low").rename("Low")
    close = _pick_col(df, "Close").rename("Close")
    out = pd.concat([high, low, close], axis=1).dropna()
    out["High"] = out["High"].astype(float)
    out["Low"] = out["Low"].astype(float)
    out["Close"] = out["Close"].astype(float)
    return out

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    series = _to_float_series(series)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0)

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    df = _normalize_ohlc(df)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]),
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()

def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    df = _normalize_ohlc(df)
    high, low = df["High"], df["Low"]
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0.0; minus_dm[minus_dm < 0] = 0.0
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0.0; minus_dm[~mask] = 0.0
    tr = (high - low).abs()
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    atr_ = tr.rolling(window, min_periods=window).mean().replace(0.0, np.nan)
    plus_di = 100 * (plus_dm.rolling(window, min_periods=window).sum() / atr_)
    minus_di = 100 * (minus_dm.rolling(window, min_periods=window).sum() / atr_)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window, min_periods=window).mean()


# ----------------------- universe & downloads -------------------------

YAHOO_OK = re.compile(r"^[A-Z0-9][A-Z0-9\.\-&]*$")

def load_universe(csv_path: str, limit: int) -> List[str]:
    df = pd.read_csv(csv_path)
    for cand in ["SYMBOL", "Symbol", "symbol", "TICKER", "Ticker", "Name"]:
        if cand in df.columns:
            syms = df[cand].astype(str)
            break
    else:
        syms = df.iloc[:, 0].astype(str)

    cleaned: List[str] = []
    for raw in syms:
        s = raw.strip().upper()
        if not s:
            continue
        if s.isdigit():
            continue
        if s in {"NIFTY", "NIFTY50", "NIFTY 50", "NIFTY-50", "^NSEI"}:
            continue
        if not YAHOO_OK.match(s):
            continue
        cleaned.append(s)

    # De-duplicate preserving order
    seen = set()
    uniq = []
    for s in cleaned:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:limit]

def to_yahoo_symbol(sym: str, suffix: str) -> str:
    if sym.endswith(suffix):
        return sym
    s = sym.replace("-EQ", "").replace(" ", "")
    return s + suffix

def chunked(seq: List[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def fetch_many_daily(symbols: List[str], start: str, end: str, suffix: str) -> Dict[str, pd.DataFrame]:
    """
    Multi-ticker daily OHLC download with yfinance (batched).
    Returns dict: symbol -> normalized OHLC (may be empty if no data).
    """
    out: Dict[str, pd.DataFrame] = {}
    if yf is None or not symbols:
        return out

    ymap = {s: to_yahoo_symbol(s, suffix) for s in symbols}
    BATCH = 50  # 25–75 is reasonable; 50 is a good balance

    for batch_syms in chunked(list(ymap.keys()), BATCH):
        ybatch = [ymap[s] for s in batch_syms]
        try:
            df = yf.download(
                tickers=ybatch,
                start=start, end=end,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception:
            for s in batch_syms:
                out[s] = pd.DataFrame(columns=["High", "Low", "Close"])
            continue

        # If multi-ticker, columns are MultiIndex (ticker, field)
        if isinstance(df.columns, pd.MultiIndex):
            for s in batch_syms:
                ysym = ymap[s]
                if ysym in df.columns.get_level_values(0):
                    try:
                        sub = df[ysym]
                        out[s] = _normalize_ohlc(sub)
                    except Exception:
                        out[s] = pd.DataFrame(columns=["High","Low","Close"])
                else:
                    out[s] = pd.DataFrame(columns=["High","Low","Close"])
        else:
            # Single ticker path
            s = batch_syms[0]
            try:
                out[s] = _normalize_ohlc(df)
            except Exception:
                out[s] = pd.DataFrame(columns=["High","Low","Close"])

    return out


# ----------------------- backtest core --------------------------------

@dataclass
class Position:
    qty: int
    entry: float
    stop: float
    last_close: float

def compute_score(close: float, rh20: float, rh50: float, adx14: float, rsi14: float) -> float:
    eps = 1e-9
    breakout_ratio = max(close / max(rh20, eps), close / max(rh50, eps))
    return 0.6 * breakout_ratio + 0.3 * (adx14 / 25.0) + 0.1 * (rsi14 / 50.0)

def backtest(
    symbols: List[str],
    start: str,
    end: str,
    max_positions: int = 5,
    total_capital: float = 100000.0,
    max_invested: float = 50000.0,
    max_risk_per_trade: float = 2000.0,
    regime_filter: int = 1,
    exchange_suffix: str = ".NS",
    slippage_bps: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Index for regime check
    idx_df = fetch_many_daily(["^NSEI"], start, end, suffix="")  # suffix "" for index
    idx = idx_df.get("^NSEI", pd.DataFrame(columns=["High","Low","Close"]))
    if idx.empty:
        # No index data -> treat as always risk-on
        ema50 = pd.Series(dtype=float)
        ema200 = pd.Series(dtype=float)
    else:
        idx_close = idx["Close"]
        ema50 = idx_close.ewm(span=50, adjust=False, min_periods=50).mean()
        ema200 = idx_close.ewm(span=200, adjust=False, min_periods=200).mean()

    # Download all symbols in batches
    data_raw = fetch_many_daily(symbols, start, end, exchange_suffix)

    # Precompute indicators
    data: Dict[str, pd.DataFrame] = {}
    for s, df in data_raw.items():
        if len(df) < 220:
            continue
        df = df.copy()
        df["RSI7"]  = rsi(df["Close"], 7)
        df["RSI14"] = rsi(df["Close"], 14)
        df["RSI21"] = rsi(df["Close"], 21)
        df["ATR14"] = atr(df, 14)
        df["ADX14"] = adx(df, 14)
        df["RH20"]  = df["Close"].rolling(20, min_periods=20).max()
        df["RH50"]  = df["Close"].rolling(50, min_periods=50).max()
        df["SMA20"] = df["Close"].rolling(20, min_periods=20).mean()
        data[s] = df

    # Trading state
    positions: Dict[str, Position] = {}
    cash = float(total_capital)
    trades: List[Dict] = []

    # Calendar (union of all dates from index; if empty, union of data)
    if not idx.empty:
        all_dates = list(idx.index)
    else:
        # fallback: use union of all data dates
        all_dates = sorted({d for df in data.values() for d in df.index})

    per_slot_budget = max_invested / float(max_positions)

    def invested_notional(now_date) -> float:
        tot = 0.0
        for sym, pos in positions.items():
            df = data.get(sym)
            if df is None or now_date not in df.index:
                price = pos.last_close
            else:
                price = float(df.at[now_date, "Close"])
            tot += pos.qty * price
        return float(tot)

    for dt in all_dates:
        # Regime check
        risk_on = True
        if regime_filter == 1 and not ema50.empty and not ema200.empty:
            e50 = float(ema50.get(dt, np.nan))
            e200 = float(ema200.get(dt, np.nan))
            if not np.isnan(e50) and not np.isnan(e200):
                risk_on = e50 > e200

        # --------- Update positions (trail & exits) -----------
        for sym in list(positions.keys()):
            df = data.get(sym)
            if df is None or dt not in df.index:
                continue
            row = df.loc[dt]
            c = float(row["Close"])
            a14 = float(row["ATR14"]) if not np.isnan(row["ATR14"]) else np.nan
            sma20 = float(row["SMA20"]) if not np.isnan(row["SMA20"]) else np.nan
            rsi14 = float(row["RSI14"]) if not np.isnan(row["RSI14"]) else np.nan

            pos = positions[sym]
            pos.last_close = c

            # trail stop to Close - 2*ATR if higher
            if not np.isnan(a14):
                new_stop = c - 2.0 * a14
                if new_stop > pos.stop:
                    pos.stop = new_stop

            # exit conditions: stop OR below SMA20 OR RSI14<30
            exit_reason = None
            if not np.isnan(c) and c <= pos.stop:
                exit_reason = "STOP"
            elif not np.isnan(c) and not np.isnan(sma20) and c < sma20:
                exit_reason = "SMA20_BREAK"
            elif not np.isnan(rsi14) and rsi14 < 30.0:
                exit_reason = "RSI_WEAK"

            # Regime purge
            if exit_reason is None and regime_filter == 1 and not risk_on:
                exit_reason = "RISK_OFF"

            if exit_reason is not None:
                price = c * (1.0 - slippage_bps / 10000.0)
                cash += pos.qty * price
                trades.append(
                    dict(date=dt.strftime("%Y-%m-%d"), symbol=sym, side="SELL",
                         qty=pos.qty, price=round(price, 4), stop=round(pos.stop, 4), reason=exit_reason)
                )
                del positions[sym]

        if regime_filter == 1 and not risk_on:
            # don’t add new entries when risk-off
            pass
        else:
            # --------- Entries (ranked) ----------
            slots = max(0, max_positions - len(positions))
            if slots > 0:
                remaining_budget = max(0.0, max_invested - invested_notional(dt))
                if remaining_budget > 0:
                    # candidates
                    candidates: List[Tuple[str, float, float, float, float]] = []
                    for sym, df in data.items():
                        if sym in positions or dt not in df.index:
                            continue
                        row = df.loc[dt]
                        c = float(row["Close"]); a14 = float(row["ATR14"])
                        adx14 = float(row["ADX14"])
                        r7, r14, r21 = float(row["RSI7"]), float(row["RSI14"]), float(row["RSI21"])
                        rh20, rh50 = float(row["RH20"]), float(row["RH50"])

                        if any(np.isnan(x) for x in (c, a14, adx14, r7, r14, r21, rh20, rh50)):
                            continue

                        cond_breakout = (c >= rh20) or (c >= rh50)
                        cond_rsi = (r7 > 40.0) and (r14 > 40.0) and (r21 > 40.0)
                        cond_adx = adx14 > 20.0
                        if not (cond_breakout and cond_rsi and cond_adx):
                            continue

                        score = compute_score(c, rh20, rh50, adx14, r14)
                        candidates.append((sym, c, a14, c - 2.0 * a14, score))

                    # rank by score
                    candidates.sort(key=lambda x: x[-1], reverse=True)

                    per_slot_budget = max_invested / float(max_positions)

                    for sym, price, a14, stop, _score in candidates:
                        if slots == 0 or remaining_budget <= 0:
                            break
                        if a14 <= 0 or price <= 0:
                            continue

                        budget_for_this = min(per_slot_budget, remaining_budget)
                        qty_cash = int(budget_for_this // price)
                        qty_risk = int(max(1.0, max_risk_per_trade / a14))
                        qty = max(1, min(qty_cash, qty_risk))

                        notional = qty * price
                        if notional > remaining_budget:
                            qty = int(remaining_budget // price)
                            notional = qty * price
                        if qty < 1 or notional <= 0:
                            continue

                        price_eff = price * (1.0 + slippage_bps / 10000.0)
                        notional_eff = qty * price_eff
                        if cash < notional_eff:
                            continue

                        cash -= notional_eff
                        positions[sym] = Position(qty=qty, entry=price_eff, stop=stop, last_close=price)
                        trades.append(
                            dict(date=dt.strftime("%Y-%m-%d"), symbol=sym, side="BUY",
                                 qty=qty, price=round(price_eff, 4), stop=round(stop, 4), reason="ENTRY")
                        )
                        remaining_budget -= qty * price
                        slots -= 1

        # (no daily equity write here; we’ll compute after the loop)

    # --------- finalize equity curve ----------
    equity_rows = []
    # use all_dates for NAV series
    for dt in all_dates:
        mtm = 0.0
        for sym, pos in positions.items():
            df = data.get(sym)
            if df is not None and dt in df.index:
                px = float(df.at[dt, "Close"])
            else:
                px = pos.last_close
            mtm += pos.qty * px
        nav = cash + mtm
        equity_rows.append(dict(date=dt.strftime("%Y-%m-%d"), cash=round(cash, 2), mtm=round(mtm, 2), equity=round(nav, 2)))

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return trades_df, equity_df


# ----------------------- CLI -----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Swing Strategy Backtest")
    ap.add_argument("--csv", default=os.getenv("NIFTY500_PATH", "NIFTY500.csv"), help="Universe CSV path")
    ap.add_argument("--limit", type=int, default=int(os.getenv("UNIVERSE_LIMIT", "500")))
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD; default: today")

    # Full config (you can override from CLI)
    ap.add_argument("--max_positions", type=int, default=5)
    ap.add_argument("--total_capital", type=float, default=500000.0)
    ap.add_argument("--max_invested", type=float, default=20000.0)
    ap.add_argument("--max_risk", type=float, default=20000.0, help="Max risk per trade (₹)")
    ap.add_argument("--regime_filter", type=int, default=int(os.getenv("REGIME_FILTER", "0")), choices=[0, 1])
    ap.add_argument("--suffix", default=".NS", help="Exchange suffix, e.g., .NS for NSE tickers")
    ap.add_argument("--slippage_bps", type=float, default=0.0, help="Per trade slippage in basis points")

    args = ap.parse_args()

    if args.end is None:
        args.end = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Load universe (cleaned)
    symbols = load_universe(args.csv, args.limit)
    print(f"[signals] Loaded {len(symbols)} cleaned symbols")

    # Run backtest
    trades_df, equity_df = backtest(
        symbols=symbols,
        start=args.start,
        end=args.end,
        max_positions=args.max_positions,
        total_capital=args.total_capital,
        max_invested=args.max_invested,
        max_risk_per_trade=args.max_risk,
        regime_filter=args.regime_filter,
        exchange_suffix=args.suffix,
        slippage_bps=args.slippage_bps,
    )

    # Save outputs
    trades_df.to_csv("trades.csv", index=False)
    equity_df.to_csv("equity_curve.csv", index=False)

    # Quick summary
    if not trades_df.empty:
        buys = (trades_df["side"] == "BUY").sum()
        sells = (trades_df["side"] == "SELL").sum()
        print(f"Buys: {buys}, Sells: {sells}")
        print("Saved: trades.csv, equity_curve.csv")
    else:
        print("No trades produced. Try disabling regime filter or widening the date range.")

if __name__ == "__main__":
    main()

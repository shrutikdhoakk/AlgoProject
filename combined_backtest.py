# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

"""
combined_backtest.py
====================

Consolidated breakout + momentum backtesting engine (single file).
"""

import argparse
import os
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, cast, Any

import numpy as np
import pandas as pd


# Lazy import yfinance
try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # Will be checked at runtime


# --------------------------- Parameters -----------------------------

@dataclass
class StrategyParameters:
    # Risk / portfolio
    total_capital: float = 100_000.0
    max_invested: float = 50_000.0
    per_trade_risk: float = 2_000.0
    max_positions: int = 5
    slippage: float = 0.001
    use_gate: bool = True
    use_trend: bool = True

    # Entry filters
    breakout_atr_buf: float = 0.45
    adx_min: float = 25.0
    rsi_min: float = 55.0
    vol_mult_min: float = 1.5
    vol_z_min: float = 1.5
    pattern_score_min: float = 30.0

    # Stops / trailing
    trail_atr_mult: float = 1.30
    early_sma20_exit: int = 2
    early_rsi14_below: float = 35.0
    early_rsi14_bars: int = 2

    # Compression gate
    bb_window: int = 20
    bb_std_mult: float = 2.0
    atr_window: int = 14
    pct_window: int = 252
    squeeze_pct: float = 0.10

    # Composite weights
    w_breakout: float = 0.5
    w_adx: float = 0.3
    w_rsi: float = 0.1
    w_pattern: float = 0.1

    # Declared private flags so Pylance is happy
    _force_first_entry: bool = field(default=False, repr=False)
    _no_breakout: bool = field(default=False, repr=False)


# ------------------------- Helpers / Indicators ---------------------

# --- Pylance-friendly numeric helper ---
def _to_float_series(s: pd.Series) -> pd.Series:
    # Keep runtime behavior but quiet Pylance's to_numeric stub mismatch
    return pd.to_numeric(s, errors="coerce").astype("float64")  # type: ignore[call-overload]


def _ensure_float_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce OHLCV columns to float to avoid Series[object] issues."""
    out = df.copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in out.columns:
            s = _to_float_series(out[col])
            out[col] = pd.Series(s, index=out.index, dtype="float64")
    return out


def rolling_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    df = _ensure_float_ohlcv(df)
    high = cast(pd.Series, df["High"])
    low = cast(pd.Series, df["Low"])
    close = cast(pd.Series, df["Close"])
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window, min_periods=1).mean()
    return cast(pd.Series, atr)


def rolling_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    s = _to_float_series(series)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = up.rolling(window, min_periods=1).mean()
    avg_loss = down.rolling(window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return cast(pd.Series, rsi).fillna(50.0)


def rolling_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    df = _ensure_float_ohlcv(df)
    high = df["High"].astype("float64", copy=False)
    low = df["Low"].astype("float64", copy=False)
    close = df["Close"].astype("float64", copy=False)

    up = high.diff().astype("float64", copy=False)
    dn = (-low.diff()).astype("float64", copy=False)

    plus_dm = up.where((up > dn) & (up > 0.0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0.0), 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window, min_periods=1).mean().replace(0.0, np.nan)
    plus_di = 100 * (plus_dm.rolling(window, min_periods=1).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window, min_periods=1).mean() / atr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100
    return dx.rolling(window, min_periods=1).mean().fillna(0.0)


# ---------------------- Pattern / Gate (vector) ---------------------

def compute_features_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_float_ohlcv(df)
    if df.empty:
        return df
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Indicators
    df["ATR14"] = rolling_atr(df, 14)
    df["ADX14"] = rolling_adx(df, 14)
    df["RSI7"] = rolling_rsi(cast(pd.Series, df["Close"]), 7)
    df["RSI14"] = rolling_rsi(cast(pd.Series, df["Close"]), 14)
    df["RSI21"] = rolling_rsi(cast(pd.Series, df["Close"]), 21)
    df["SMA20"] = cast(pd.Series, df["Close"]).rolling(20, min_periods=1).mean()
    df["SMA50"] = cast(pd.Series, df["Close"]).rolling(50, min_periods=1).mean()
    df["SMA200"] = cast(pd.Series, df["Close"]).rolling(200, min_periods=1).mean()

    # Prior highs (shift to avoid lookahead)
    df["HH20"] = cast(pd.Series, df["High"]).rolling(20, min_periods=20).max().shift(1)
    df["HH50"] = cast(pd.Series, df["High"]).rolling(50, min_periods=50).max().shift(1)

    # Volume stats
    if "Volume" in df.columns:
        vol = cast(pd.Series, df["Volume"]).astype("float64", copy=False)
        df["VolAvg20"] = vol.rolling(20, min_periods=1).mean()
        vol_std = vol.rolling(20, min_periods=1).std(ddof=0)
        df["VolZ20"] = (vol - df["VolAvg20"]) / vol_std.replace({0.0: np.nan})
    else:
        df["VolAvg20"] = np.nan
        df["VolZ20"] = np.nan

    # Simple bullish pattern proxy
    high = df["High"].astype("float64", copy=False)
    low = df["Low"].astype("float64", copy=False)
    close = df["Close"].astype("float64", copy=False)
    open_ = df["Open"].astype("float64", copy=False)

    rng = (high - low).abs().replace(0.0, np.nan)
    body_up = (close - open_).clip(lower=0.0)
    df["BullCandle"] = close > open_
    df["PatternScore"] = np.where(df["BullCandle"], 100.0 * body_up / rng, 0.0)
    df["PatternScore"] = cast(pd.Series, df["PatternScore"]).fillna(0.0).clip(0.0, 100.0)

    # --- Vectorized squeeze/breakout gate ---
    _bb_win = 20; _bb_std_mult = 2.0
    _atr_win = 14; _pct_win = 252; _squeeze_pct = 0.10

    close_f = df["Close"].astype("float64", copy=False)
    high_f  = df["High"].astype("float64", copy=False)
    low_f   = df["Low"].astype("float64", copy=False)

    _mid = close_f.rolling(_bb_win, min_periods=_bb_win).mean()
    _std = close_f.rolling(_bb_win, min_periods=_bb_win).std(ddof=0)
    _bb_upper = _mid + _bb_std_mult * _std
    _bb_lower = _mid - _bb_std_mult * _std
    _bb_width = (_bb_upper - _bb_lower) / _mid

    import numpy as _np
    def _pct_of_last(arr) -> float:
        last = arr[-1]
        valid = arr[~_np.isnan(arr)]
        if valid.size == 0 or _np.isnan(last):
            return _np.nan
        return float((valid <= last).mean())

    df["BBWidthPctile"] = _bb_width.rolling(_pct_win, min_periods=_bb_win).apply(_pct_of_last, raw=True)

    _atr = df["ATR14"]
    if _atr.isna().all():
        _prev_close = close_f.shift(1)
        _tr = _np.maximum.reduce([
            (high_f - low_f).abs().to_numpy(),
            (high_f - _prev_close).abs().to_numpy(),
            (low_f - _prev_close).abs().to_numpy(),
        ])
        _atr = pd.Series(_tr, index=df.index).rolling(_atr_win, min_periods=_atr_win).mean()
    df["ATRPctile"] = _atr.rolling(_pct_win, min_periods=_atr_win).apply(_pct_of_last, raw=True)

    _hh20 = high_f.rolling(20, min_periods=20).max()
    _hh50 = high_f.rolling(50, min_periods=50).max()

    _bo = (close_f > _hh20) | (close_f > _hh50)
    _squeeze = (df["BBWidthPctile"] <= _squeeze_pct) | (df["ATRPctile"] <= _squeeze_pct * 2.0)
    df["CandidateOK"] = (_squeeze & _bo).fillna(False)

    return df


def pattern_gate(
    df: pd.DataFrame,
    bb_window: int = 20,
    bb_std_mult: float = 2.0,
    atr_window: int = 14,
    pct_window: int = 252,
    squeeze_pct: float = 0.10,
) -> Dict[str, Optional[float]]:
    """Identify volatility compression and breakout readiness for a single symbol."""
    if not isinstance(df, pd.DataFrame) or len(df) < max(bb_window, atr_window) + 2:
        return {"candidate_ok": False, "bb_width_pctile": None, "atr_pctile": None}
    df = _ensure_float_ohlcv(df)
    close = df["Close"].astype("float64", copy=False)
    high  = df["High"].astype("float64", copy=False)
    low   = df["Low"].astype("float64", copy=False)
    # Bollinger Bands
    mid = close.rolling(bb_window, min_periods=bb_window).mean()
    std = close.rolling(bb_window, min_periods=bb_window).std(ddof=0)
    upper = mid + bb_std_mult * std
    lower = mid - bb_std_mult * std
    bb_width = (upper - lower) / mid.replace(0.0, np.nan)
    # ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean()
    # Percentile rank helper
    def pct_rank(s: pd.Series) -> float:
        if len(s) == 0 or pd.isna(s.iloc[-1]):
            return np.nan
        return float(s.rank(pct=True).iloc[-1])
    bb_pctile = bb_width.rolling(pct_window, min_periods=min(bb_window * 3, pct_window)).apply(pct_rank, raw=False)
    atr_pctile = atr.rolling(pct_window, min_periods=min(atr_window * 3, pct_window)).apply(pct_rank, raw=False)
    # Breakouts
    hh20 = high.shift(1).rolling(20, min_periods=20).max()
    hh50 = high.shift(1).rolling(50, min_periods=50).max()
    breakout_hh20 = close > hh20   # float-vs-float
    breakout_hh50 = close > hh50
    # Latest values
    last_bb_pct = bb_pctile.dropna().iloc[-1] if len(bb_pctile.dropna()) else np.nan
    last_atr_pct = atr_pctile.dropna().iloc[-1] if len(atr_pctile.dropna()) else np.nan
    is_squeeze = False
    try:
        if last_bb_pct <= squeeze_pct or last_atr_pct <= squeeze_pct * 2:
            is_squeeze = True
    except Exception:
        pass
    bo20 = bool(breakout_hh20.dropna().iloc[-1]) if len(breakout_hh20.dropna()) else False
    bo50 = bool(breakout_hh50.dropna().iloc[-1]) if len(breakout_hh50.dropna()) else False
    candidate_ok = bool(is_squeeze and (bo20 or bo50))
    return {
        "candidate_ok": candidate_ok,
        "bb_width_pctile": None if pd.isna(last_bb_pct) else float(last_bb_pct),
        "atr_pctile": None if pd.isna(last_atr_pct) else float(last_atr_pct),
    }


def compute_composite_score(row: pd.Series, params: StrategyParameters) -> float:
    hh_prev = max(row.get("HH20", np.nan), row.get("HH50", np.nan))
    atr14 = row.get("ATR14", np.nan)
    close = row.get("Close", np.nan)
    if np.isnan(close) or np.isnan(atr14) or np.isnan(hh_prev):
        breakout_ratio = 0.0
    else:
        target = hh_prev + params.breakout_atr_buf * atr14
        breakout_ratio = max(0.0, (close - target) / (abs(atr14) + 1e-9))
    adx_norm = max(0.0, (row.get("ADX14", 0.0) - params.adx_min) / max(params.adx_min, 1e-9))
    rsi_norm = max(0.0, (row.get("RSI14", 0.0) - params.rsi_min) / max(params.rsi_min, 1e-9))
    pat_norm = max(0.0, row.get("PatternScore", 0.0) / 100.0)
    score = (
        params.w_breakout * breakout_ratio
        + params.w_adx * adx_norm
        + params.w_rsi * rsi_norm
        + params.w_pattern * pat_norm
    )
    return float(score)


# --------------------------- Data Download --------------------------

def download_prices(symbols: List[str], start: str, end: str, suffix: str = ".NS") -> Dict[str, pd.DataFrame]:
    if yf is None:
        raise ImportError("yfinance is required to download price data. Install via `pip install yfinance`.")
    results: Dict[str, pd.DataFrame] = {}

    def pick_col(frame: pd.DataFrame, want: str) -> pd.Series:
        """
        Robustly extract a 1-D Series for the requested field name from a DataFrame
        that may have plain columns or a MultiIndex with 2–3 levels
        (e.g. ('Price','Close','RELIANCE.NS')).
        """
        assert isinstance(frame, pd.DataFrame)
        want_l = want.lower()
        aliases = {"close": ["close", "adj close", "price"], "open": ["open"], "high": ["high"], "low": ["low"], "volume": ["volume"]}
        targets = aliases.get(want_l, [want_l])

        # 1) Simple columns
        if not isinstance(frame.columns, pd.MultiIndex):
            for t in targets:
                exact = [c for c in frame.columns if str(c).lower() == t]
                if exact:
                    s = frame[exact[0]]
                    return _to_float_series(s)
                sw = [c for c in frame.columns if str(c).lower().startswith(t)]
                if sw:
                    s = frame[sw[0]]
                    return _to_float_series(s)
            if frame.shape[1] == 1:
                s = frame.iloc[:, 0]
                return _to_float_series(s)
            return pd.Series(index=frame.index, dtype="float64")

        # 2) MultiIndex columns (2–3 levels). Match if ANY level equals/starts-with target.
        def col_matches(col, t: str) -> bool:
            parts = [str(x).lower() for x in (col if isinstance(col, tuple) else (col,))]
            return any(p == t or p.startswith(t) for p in parts)

        # Try each target in priority order
        for t in targets:
            # (a) direct column pick if tuple/col matches
            matches = [col for col in frame.columns if col_matches(col, t)]
            if matches:
                s = frame[matches[0]]
                if isinstance(s, pd.DataFrame):
                    # if one ticker, squeeze to Series; else pick first column
                    s = s.iloc[:, 0] if s.shape[1] >= 1 else s.squeeze()
                return _to_float_series(s)

            # (b) try xs() by level if that level equals the field
            for lvl in range(frame.columns.nlevels):
                level_vals = [str(v).lower() for v in frame.columns.get_level_values(lvl)]
                if t in level_vals:
                    try:
                        s = frame.xs(key=t, axis=1, level=lvl, drop_level=False)
                        if isinstance(s, pd.DataFrame):
                            col0 = s.columns[0]
                            s = s[col0]
                            if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                                s = s.iloc[:, 0]
                        return _to_float_series(s)
                    except Exception:
                        pass

            # (c) special case: ('price','close',ticker) style
            try:
                if frame.columns.nlevels >= 2:
                    level0 = [str(v).lower() for v in frame.columns.get_level_values(0)]
                    level1 = [str(v).lower() for v in frame.columns.get_level_values(1)]
                    if "price" in level0 and t in level1:
                        subcols = [col for col in frame.columns if str(col[0]).lower() == "price" and str(col[1]).lower().startswith(t)]
                        if subcols:
                            s = frame[subcols[0]]
                            if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                                s = s.iloc[:, 0]
                            return _to_float_series(s)
            except Exception:
                pass

        if frame.shape[1] == 1:
            s = frame.iloc[:, 0]
            return _to_float_series(s)

        return pd.Series(index=frame.index, dtype="float64")

    for sym in symbols:
        ticker = sym if sym.endswith(suffix) or suffix == "" else sym + suffix
        df: pd.DataFrame = pd.DataFrame()
        for _try in range(3):
            try:
                dl = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                )
                if isinstance(dl, pd.DataFrame) and not dl.empty:
                    df = dl
                    break
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            results[sym] = pd.DataFrame()
            continue

        op = pick_col(df, "Open")
        hi = pick_col(df, "High")
        lo = pick_col(df, "Low")
        cl = pick_col(df, "Close")
        vo = pick_col(df, "Volume")

        out = pd.DataFrame(
            {"Open": op, "High": hi, "Low": lo, "Close": cl, "Volume": vo},
            index=df.index,
        ).dropna(subset=["Close"])
        results[sym] = _ensure_float_ohlcv(out)

    return results


# ------------------------------ Engine ------------------------------

def backtest(
    symbols: List[str],
    start: str,
    end: str,
    params: StrategyParameters,
    index_symbol: str = "^NSEI",
    suffix: str = ".NS",
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Index regime data
    indices: Dict[str, pd.DataFrame] = {}
    if index_symbol:
        try:
            indices = download_prices([index_symbol], start, end, suffix="")
        except Exception:
            indices = {}
    idx_df = indices.get(index_symbol, pd.DataFrame())

    if not idx_df.empty:
        try:
            if isinstance(idx_df.columns, pd.MultiIndex):
                close_cols = [c for c in idx_df.columns
                              if (isinstance(c, tuple) and (str(c[-1]).lower() in ("close", "adj close")))]
                if close_cols:
                    s = idx_df[close_cols[0]]
                    if isinstance(s, pd.DataFrame):
                        s = s.squeeze()
                else:
                    flat = idx_df.droplevel(0, axis=1)
                    if "Close" in flat.columns:
                        s = flat.loc[:, "Close"]
                    elif "Adj Close" in flat.columns:
                        s = flat.loc[:, "Adj Close"]
                    else:
                        s = flat.squeeze()
            else:
                if "Close" in idx_df.columns:
                    s = idx_df.loc[:, "Close"]
                elif "Adj Close" in idx_df.columns:
                    s = idx_df.loc[:, "Adj Close"]
                else:
                    s = idx_df.squeeze()

            s = _to_float_series(pd.Series(s)) if not isinstance(s, pd.Series) else _to_float_series(s)
            idx_close = s.dropna()
        except Exception:
            idx_close = pd.Series(dtype=float)

        if idx_close.empty:
            ema50 = pd.Series(dtype=float)
            ema200 = pd.Series(dtype=float)
        else:
            ema50 = idx_close.ewm(span=50, adjust=False, min_periods=50).mean()
            ema200 = idx_close.ewm(span=200, adjust=False, min_periods=200).mean()
    else:
        ema50 = pd.Series(dtype=float)
        ema200 = pd.Series(dtype=float)

    # Download & features
    raw_data = download_prices(symbols, start, end, suffix=suffix)
    if debug:
        total_rows = sum((0 if (df is None or df.empty) else len(df)) for df in raw_data.values())
        for _sym, _df in raw_data.items():
            rows = 0 if _df is None or _df.empty else len(_df)
            print(f"RAW DATA {_sym}: rows={rows}")
        print(f"TOTAL RAW ROWS: {total_rows}")

    data: Dict[str, pd.DataFrame] = {}
    for sym, df in raw_data.items():
        try:
            fdf = compute_features_for_symbol(df)
            fdf = fdf.dropna(subset=["HH20", "HH50", "ATR14", "ADX14", "RSI14"])
            if not fdf.empty:
                data[sym] = fdf
        except Exception:
            continue
    if debug:
        total_rows = sum((0 if (df is None or df.empty) else len(df)) for df in data.values())
        print(f"FEATURED DATA symbols={len(data)} | TOTAL FEATURE ROWS: {total_rows}")

    # Trading state
    positions: Dict[str, Dict[str, float]] = {}
    cash = float(params.total_capital)
    trades: List[Dict[str, Any]] = []

    # Trading calendar
    all_dates_set = set()
    for df in data.values():
        all_dates_set.update(df.index.tolist())
    if not idx_df.empty:
        all_dates_set.update(idx_df.index.tolist())
    if not all_dates_set:
        return pd.DataFrame(), pd.DataFrame()
    all_dates = sorted(all_dates_set)

    def invested_notional(at_date: pd.Timestamp) -> float:
        total = 0.0
        for sym, pos in positions.items():
            df = data.get(sym)
            if df is None or at_date not in df.index:
                price = float(pos.get("last_price", pos.get("entry_price", 0.0)))
            else:
                price = float(df.at[at_date, "Close"])
                positions[sym]["last_price"] = price
            total += pos["qty"] * price
        return float(total)

    equity_records: List[Tuple[pd.Timestamp, float]] = []

    # Main loop
    for dt in all_dates:
        # Regime
        risk_on = True
        if not ema50.empty and not ema200.empty:
            e50 = float(ema50.get(dt, np.nan))
            e200 = float(ema200.get(dt, np.nan))
            if not np.isnan(e50) and not np.isnan(e200):
                risk_on = e50 > e200

        # ---- Manage open positions ----
        exited: List[str] = []
        for sym in list(positions.keys()):
            df = data.get(sym)
            if df is None or dt not in df.index:
                continue
            row = df.loc[dt]
            atr = float(row.get("ATR14", np.nan))
            if not np.isnan(atr):
                new_stop = float(row["Close"]) - params.trail_atr_mult * atr
                if new_stop > positions[sym]["stop"]:
                    positions[sym]["stop"] = new_stop
            exit_reason: Optional[str] = None
            price_now = float(row.get("Close", positions[sym].get("last_price", 0.0)))
            if price_now <= positions[sym]["stop"]:
                exit_reason = "STOP"
            sma20 = float(row.get("SMA20", np.nan))
            if not np.isnan(sma20) and price_now < sma20:
                positions[sym]["sma_below_count"] += 1
            else:
                positions[sym]["sma_below_count"] = 0
            if positions[sym]["sma_below_count"] >= params.early_sma20_exit:
                exit_reason = exit_reason or "SMA20_BREAK"
            rsi14 = float(row.get("RSI14", np.nan))
            if not np.isnan(rsi14) and rsi14 < params.early_rsi14_below:
                positions[sym]["rsi_below_count"] += 1
            else:
                positions[sym]["rsi_below_count"] = 0
            if positions[sym]["rsi_below_count"] >= params.early_rsi14_bars:
                exit_reason = exit_reason or "RSI_WEAK"
            if exit_reason is None and not risk_on:
                exit_reason = "RISK_OFF"
            if exit_reason:
                qty = positions[sym]["qty"]
                exit_price = price_now * (1.0 - params.slippage)
                cash += qty * exit_price
                trades.append(
                    {
                        "date": dt.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "side": "SELL",
                        "qty": qty,
                        "price": round(exit_price, 4),
                        "reason": exit_reason,
                    }
                )
                exited.append(sym)
        for sym in exited:
            positions.pop(sym, None)

        # ---- Entries ----
        if risk_on:
            available_slots = max(0, params.max_positions - len(positions))
            remaining_budget = max(0.0, params.max_invested - invested_notional(dt))
            if debug:
                print(f"DT {dt.date()} | risk_on={risk_on} | slots={available_slots} | budget={remaining_budget:.2f} | open={len(positions)}")

            # Diagnostic path
            if params._force_first_entry:
                for sym, df in data.items():
                    if sym in positions or dt not in df.index:
                        continue
                    price = float(df.at[dt, "Close"])
                    if remaining_budget >= price and available_slots > 0:
                        positions[sym] = {
                            "qty": 1,
                            "entry_date": dt,
                            "entry_price": price,
                            "stop": price * (1.0 - 0.05),
                            "sma_below_count": 0,
                            "rsi_below_count": 0,
                            "last_price": price,
                        }
                        cash -= price * (1.0 + params.slippage)
                        trades.append({"date": dt.strftime("%Y-%m-%d"), "symbol": sym, "side": "BUY", "qty": 1, "price": round(price, 4), "reason": "FORCE"})
                        available_slots -= 1
                        remaining_budget -= price

            if available_slots > 0 and remaining_budget > 0:
                candidates: List[Tuple[str, float]] = []
                for sym, df in data.items():
                    if sym in positions or dt not in df.index:
                        continue
                    row = df.loc[dt]
                    if params.use_gate and not bool(row.get("CandidateOK", False)):
                        continue
                    if params.use_trend:
                        if not (
                            float(row["Close"]) > float(row.get("SMA50", np.nan))
                            and float(row.get("SMA50", np.nan)) > float(row.get("SMA200", np.nan))
                            and float(row.get("SMA20", np.nan)) > float(row.get("SMA50", np.nan))
                        ):
                            continue
                    hh = max(row.get("HH20", np.nan), row.get("HH50", np.nan))
                    if np.isnan(hh):
                        continue
                    atr = row.get("ATR14", np.nan)
                    close_now = row.get("Close", np.nan)
                    if np.isnan(atr) or np.isnan(close_now):
                        continue
                    if (not params._no_breakout) and close_now <= (hh + params.breakout_atr_buf * atr):
                        continue
                    if row.get("ADX14", 0.0) < params.adx_min:
                        continue
                    if row.get("RSI14", 0.0) < params.rsi_min:
                        continue
                    volavg = row.get("VolAvg20", np.nan)
                    vol = row.get("Volume", np.nan)
                    volz = row.get("VolZ20", np.nan)
                    if not np.isnan(vol) and not np.isnan(volavg):
                        if vol < params.vol_mult_min * volavg:
                            continue
                        if not np.isnan(volz) and volz < params.vol_z_min:
                            continue
                    if row.get("PatternScore", 0.0) < params.pattern_score_min:
                        continue
                    sc = compute_composite_score(row, params)
                    candidates.append((sym, sc))

                candidates.sort(key=lambda x: x[1], reverse=True)
                per_slot_cap = params.max_invested / float(params.max_positions)

                for sym, _sc in candidates:
                    if available_slots == 0 or remaining_budget <= 0:
                        break
                    df = data[sym]
                    row = df.loc[dt]
                    price = float(row["Close"])
                    atr = float(row["ATR14"])
                    qty_from_cap = int(per_slot_cap // price)
                    qty_from_risk = int(max(1.0, params.per_trade_risk / (params.trail_atr_mult * atr + 1e-9)))
                    qty = max(1, min(qty_from_cap, qty_from_risk))
                    notional = qty * price
                    if notional > remaining_budget or qty < 1:
                        continue
                    buy_price = price * (1.0 + params.slippage)
                    cash -= qty * buy_price
                    positions[sym] = {
                        "qty": qty,
                        "entry_price": buy_price,
                        "stop": price - params.trail_atr_mult * atr,
                        "sma_below_count": 0,
                        "rsi_below_count": 0,
                        "last_price": price,
                    }
                    trades.append(
                        {
                            "date": dt.strftime("%Y-%m-%d"),
                            "symbol": sym,
                            "side": "BUY",
                            "qty": qty,
                            "price": round(buy_price, 4),
                            "reason": "ENTRY",
                        }
                    )
                    available_slots -= 1
                    remaining_budget -= notional

        equity = cash + invested_notional(dt)
        equity_records.append((dt, equity))

    trades_df = pd.DataFrame(trades)
    equity_curve_df = pd.DataFrame(equity_records, columns=["date", "equity"])
    equity_curve_df["date"] = pd.to_datetime(equity_curve_df["date"])
    return trades_df, equity_curve_df


# ------------------------- Metrics & CLI ----------------------------

def compute_performance_metrics(equity_curve: pd.DataFrame) -> Dict[str, float]:
    if equity_curve.empty:
        return {}
    eq = cast(pd.Series, equity_curve.set_index("date").sort_index()["equity"])
    returns = cast(pd.Series, eq.pct_change().dropna())
    if len(returns) == 0:
        return {}
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-9)) - 1.0)
    vol = float(returns.std() * np.sqrt(252))
    rolling_max = eq.cummax()
    dd = eq / rolling_max - 1.0
    max_dd = float(dd.min())
    return {"total_return": total_return, "cagr": cagr, "volatility": vol, "max_drawdown": max_dd}


# ------------------------------ CLI --------------------------------

def _load_symbols_from_universe(path: str) -> List[str]:
    """Read symbols from CSV/TXT.
    - CSV with header 'symbol' (preferred), or any first column.
    - TXT: one symbol per line (comma allowed; first token used).
    """
    syms: List[str] = []
    if not path or not os.path.exists(path):
        return syms
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".txt"):
            with open(path, "r", encoding="utf-8-sig") as f:
                rdr = csv.reader(f)
                rows = list(rdr)
                if not rows:
                    return []
                # If header has 'symbol', drop header
                header = [c.strip().lower() for c in rows[0]]
                if "symbol" in header:
                    rows = rows[1:]
                # Take first column/token
                for r in rows:
                    if not r:
                        continue
                    s = str(r[0]).strip().upper()
                    if not s:
                        continue
                    # If a line like "AAA, BBB", take 'AAA'
                    if "," in s:
                        s = s.split(",")[0].strip().upper()
                    syms.append(s)
        else:
            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    s = line.strip().split(",")[0].strip().upper()
                    if s:
                        syms.append(s)
    except Exception:
        return []
    # De-dupe, preserve order
    seen: set[str] = set()
    out: List[str] = []
    for s in syms:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main():
    parser = argparse.ArgumentParser(description="Backtest combined breakout+momentum strategy")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated tickers (no suffix) OR CSV path with a 'symbol' column")
    parser.add_argument("--universe", type=str, default=None, help="Path to CSV/TXT with symbols (header 'symbol' or one per line)")
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--index", type=str, default="^NSEI", help="Benchmark index ('' to disable regime)")
    parser.add_argument("--suffix", type=str, default=".NS", help="Yahoo suffix (default .NS)")
    parser.add_argument("--output", type=str, default="", help="Base CSV path to write *_trades.csv and *_equity.csv")
    # Overrides
    parser.add_argument("--breakout_atr_buf", type=float)
    parser.add_argument("--trail_atr_mult", type=float)
    parser.add_argument("--rsi_min", type=float)
    parser.add_argument("--adx_min", type=float)
    parser.add_argument("--max_positions", type=int)
    parser.add_argument("--per_trade_risk", type=float)
    parser.add_argument("--max_invested", type=float)
    parser.add_argument("--slippage", type=float)
    parser.add_argument("--pattern_score_min", type=float)
    parser.add_argument("--vol_mult_min", type=float)
    parser.add_argument("--vol_z_min", type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_breakout", action="store_true")
    parser.add_argument("--force_first_entry", action="store_true")
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--no_trend", action="store_true")
    args = parser.parse_args()

    # Parse symbols from either --symbols (string or CSV path) and/or --universe
    syms: List[str] = []

    _p = (args.symbols or "").strip().strip('"').strip("'")
    if _p:
        if os.path.isfile(_p) and _p.lower().endswith(".csv"):
            try:
                df_symbols = pd.read_csv(_p)
                col = "symbol" if "symbol" in df_symbols.columns else df_symbols.columns[0]
                s = cast(pd.Series, df_symbols[col]).astype(str).str.upper().tolist()
                syms.extend([x.strip().upper() for x in s if x and isinstance(x, str)])
            except Exception as e:
                raise SystemExit(f"Failed to read symbols from {_p}: {e}")
        else:
            syms.extend([s.strip().upper() for s in _p.split(",") if s.strip()])

    if args.universe:
        syms.extend(_load_symbols_from_universe(args.universe))

    # De-dupe while preserving order
    seen = set()
    syms = [s for s in syms if s and not (s in seen or seen.add(s))]

    if not syms:
        raise SystemExit("No symbols provided. Use --symbols and/or --universe.")

    # Params
    params = StrategyParameters()
    params._no_breakout = bool(args.no_breakout)
    params._force_first_entry = bool(args.force_first_entry)
    for k in [
        "breakout_atr_buf", "trail_atr_mult", "rsi_min", "adx_min", "max_positions",
        "per_trade_risk", "max_invested", "slippage", "pattern_score_min", "vol_mult_min", "vol_z_min",
    ]:
        val = getattr(args, k)
        if val is not None:
            setattr(params, k, val)
    if args.no_gate:
        params.use_gate = False
    if args.no_trend:
        params.use_trend = False

    trades_df, equity_curve_df = backtest(
        symbols=syms,
        start=args.start,
        end=args.end,
        params=params,
        index_symbol=args.index,
        suffix=args.suffix,
        debug=args.debug,
    )
    if trades_df.empty or equity_curve_df.empty:
        print("Backtest produced no trades or equity curve—check your inputs.")
        return
    metrics = compute_performance_metrics(equity_curve_df)
    print("Performance metrics:\n", metrics)
    if args.output:
        base = args.output.rsplit(".csv", 1)[0]
        trades_path = base + "_trades.csv"
        eq_path = base + "_equity.csv"
        trades_df.to_csv(trades_path, index=False)
        equity_curve_df.to_csv(eq_path, index=False)
        print(f"Saved trades to {trades_path} and equity curve to {eq_path}")
    else:
        print("\nLast 5 trades:")
        print(trades_df.tail())
        print("\nEquity curve (last 5 rows):")
        print(equity_curve_df.tail())


if __name__ == "__main__":  # pragma: no cover
    main()

# Full Bearish Swing Strategy Backtest
# ---------------------------------------------------------------------
# Implements a mirrored version of the bullish swing strategy for short trades
# - Entry: Breakdown below recent lows (RL10)
# - Conditions: ADX > 15, RSI < 70
# - Stop: Price + 2 * ATR
# - Target: Recent low (RL50) - ATR
# - Exit: price >= stop, RSI > 70, or price above SMA
# ---------------------------------------------------------------------

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple
from dataclasses import dataclass

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs)).fillna(100.0)

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff().where(high.diff() > low.diff(), 0.0)
    minus_dm = -low.diff().where(low.diff() > high.diff(), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.rolling(window).mean()
    plus_di = 100 * plus_dm.rolling(window).sum() / atr_
    minus_di = 100 * minus_dm.rolling(window).sum() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(window).mean()

def reward_to_risk(entry: float, target: float, stop: float) -> float:
    risk = max(stop - entry, 1e-9)
    reward = max(entry - target, 0.0)
    return reward / risk

@dataclass
class Position:
    qty: int
    entry: float
    stop: float
    last_close: float

def ai_score(features: Dict[str, float]) -> float:
    return (
        0.4 * features.get('breakdown_ratio', 1.0) +
        0.3 * features.get('adx14', 20) / 25.0 +
        0.2 * (100 - features.get('rsi14', 50)) / 50.0 +
        0.1 * features.get('r2r', 1.0)
    )

def backtest(
    data: Dict[str, pd.DataFrame],
    dates: List[pd.Timestamp],
    max_positions: int = 5,
    total_capital: float = 100000.0,
    max_invested: float = 50000.0,
    min_rr_ratio: float = 1.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    positions: Dict[str, Position] = {}
    cash = float(total_capital)
    trades: List[Dict] = []
    per_slot_budget = max_invested / max_positions

    for dt in dates:
        for sym in list(positions.keys()):
            df = data[sym]
            if dt not in df.index:
                continue
            row = df.loc[dt]
            c = float(row['Close'])
            sma20 = float(row['SMA20'])
            rsi14 = float(row['RSI14'])
            pos = positions[sym]
            pos.last_close = c
            new_stop = c + 2.0 * row['ATR14']
            if new_stop < pos.stop:
                pos.stop = new_stop
            exit_reason = None
            if c >= pos.stop:
                exit_reason = 'STOP'
            elif c > sma20:
                exit_reason = 'SMA_BREAK'
            elif rsi14 > 70.0:
                exit_reason = 'RSI_WEAK'
            if exit_reason:
                trades.append(dict(date=dt, symbol=sym, side='COVER', qty=pos.qty, price=round(c,2), reason=exit_reason))
                cash -= pos.qty * c
                del positions[sym]

        available_slots = max_positions - len(positions)
        if available_slots > 0:
            candidates = []
            for sym, df in data.items():
                if sym in positions or dt not in df.index:
                    continue
                row = df.loc[dt]
                close, a14 = row['Close'], row['ATR14']
                adx14, rsi14 = row['ADX14'], row['RSI14']
                rl10, rl50 = row['RL10'], row['RL50']
                if close <= rl10 and adx14 > 15 and rsi14 < 70:
                    stop = close + 2.0 * a14
                    target = rl50 - a14
                    r2r = reward_to_risk(close, target, stop)
                    if r2r < min_rr_ratio:
                        continue
                    score = ai_score(dict(breakdown_ratio=rl10/close, adx14=adx14, rsi14=rsi14, r2r=r2r))
                    candidates.append((score, sym, close, stop))
            candidates.sort(reverse=True)
            for score, sym, price, stop in candidates[:available_slots]:
                budget = min(per_slot_budget, max_invested - sum(p.qty*p.last_close for p in positions.values()))
                qty = int(budget // price)
                if qty < 1:
                    continue
                cost = qty * price
                if cash >= cost:
                    positions[sym] = Position(qty=qty, entry=price, stop=stop, last_close=price)
                    cash += cost
                    trades.append(dict(date=dt, symbol=sym, side='SHORT', qty=qty, price=round(price,2), reason='ENTRY'))

    equity = []
    for dt in dates:
        mtm = sum(p.qty * (p.entry - (data[sym].at[dt,'Close'] if dt in data[sym].index else p.last_close)) for sym, p in positions.items())
        equity.append(dict(date=dt, cash=round(cash,2), mtm=round(mtm,2), equity=round(cash+mtm,2)))
    return pd.DataFrame(trades), pd.DataFrame(equity)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='NIFTY500.csv')
    ap.add_argument('--start', default='2022-01-01')
    ap.add_argument('--end', default=pd.Timestamp.today().strftime('%Y-%m-%d'))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    symbols = df.iloc[:,0].astype(str).unique().tolist()
    raw = yf.download(symbols, start=args.start, end=args.end, group_by='ticker', auto_adjust=False, threads=True)
    data = {}
    for sym in symbols:
        try:
            if sym not in raw.columns.get_level_values(0):
                continue
            d = raw[sym][['High','Low','Close']].dropna().copy()
            d['RSI14'] = rsi(d['Close'])
            d['ATR14'] = atr(d)
            d['ADX14'] = adx(d)
            d['RL10'] = d['Close'].rolling(10).min()
            d['RL50'] = d['Close'].rolling(50).min()
            d['SMA20'] = d['Close'].rolling(20).mean()
            data[sym] = d
        except:
            continue
    all_dates = [set(df.index) for df in data.values() if not df.empty]
    dates = sorted(set.intersection(*all_dates)) if all_dates else []
    trades, equity = backtest(data, dates)
    trades.to_csv('bear_trades.csv', index=False)
    equity.to_csv('bear_equity_curve.csv', index=False)
    print(f"Backtest complete. {len(trades)} short trades. Saved to CSV.")

if __name__ == '__main__':
    main()
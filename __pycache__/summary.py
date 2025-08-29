import pandas as pd, numpy as np

# --- load files in current folder ---
tr = pd.read_csv("trades.csv")           # expects: date,symbol,side,qty,price,stop,reason
eq = pd.read_csv("equity_curve.csv")     # expects: date,cash,mtm,equity

# --- build round trips (BUY -> next SELL per symbol, FIFO) ---
rounds = []
for sym, g in tr.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    fifo = []
    for _, row in g.iterrows():
        if row["side"] == "BUY":
            fifo.append(row)
        elif row["side"] == "SELL" and fifo:
            buy = fifo.pop(0)
            qty = int(min(row["qty"], buy["qty"]))
            pnl = (float(row["price"]) - float(buy["price"])) * qty
            rounds.append({
                "symbol": sym,
                "buy_date": buy["date"],
                "sell_date": row["date"],
                "buy_px": float(buy["price"]),
                "sell_px": float(row["price"]),
                "qty": qty,
                "pnl": float(pnl),
            })

rt = pd.DataFrame(rounds)

# --- summary metrics ---
gross_pnl = float(rt["pnl"].sum()) if not rt.empty else 0.0
wins = int((rt["pnl"] > 0).sum()) if not rt.empty else 0
losses = int((rt["pnl"] <= 0).sum()) if not rt.empty else 0
win_rate = (wins / len(rt) * 100.0) if len(rt) else 0.0

eq = eq.copy()
eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
eq = eq.dropna(subset=["equity"])
start_val = float(eq["equity"].iloc[0]) if len(eq) else np.nan
end_val = float(eq["equity"].iloc[-1]) if len(eq) else np.nan
days = len(eq)                                  # rows in equity csv
years = max(1e-9, days / 252.0)                 # ~252 trading days/year
cagr = ((end_val / start_val) ** (1.0/years) - 1.0) * 100.0 if start_val and start_val>0 else np.nan

roll_max = eq["equity"].cummax()
dd = (eq["equity"] / roll_max - 1.0) * 100.0
max_dd = float(dd.min()) if len(dd) else np.nan

print(f"Trades (round trips): {len(rt)}  |  Wins: {wins}  Losses: {losses}  Win rate: {win_rate:.1f}%")
print(f"Gross PnL: {gross_pnl:,.2f}")
print(f"CAGR: {cagr:.2f}%")
print(f"Max Drawdown: {max_dd:.2f}%")

# --- Top winners / losers (optional) ---
if not rt.empty:
    print("\nTop 10 winners:")
    print(rt.sort_values("pnl", ascending=False).head(10)[["symbol","buy_date","sell_date","qty","buy_px","sell_px","pnl"]].to_string(index=False))
    print("\nTop 10 losers:")
    print(rt.sort_values("pnl", ascending=True).head(10)[["symbol","buy_date","sell_date","qty","buy_px","sell_px","pnl"]].to_string(index=False))
else:
    print("\nNo completed round trips found.")

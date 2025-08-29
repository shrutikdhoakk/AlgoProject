import pandas as pd

# Load trades
tr = pd.read_csv("trades.csv")   # expects: date,symbol,side,qty,price,stop,reason

# --- Build round trips (BUY -> next SELL per symbol) ---
rounds = []
for sym, g in tr.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    buys = []
    for _, row in g.iterrows():
        if row["side"] == "BUY":
            buys.append(row)
        elif row["side"] == "SELL" and buys:
            buy = buys.pop(0)
            qty = min(row["qty"], buy["qty"])
            pnl = (row["price"] - buy["price"]) * qty
            rounds.append({
                "symbol": sym,
                "buy_date": buy["date"],
                "sell_date": row["date"],
                "qty": qty,
                "buy_px": buy["price"],
                "sell_px": row["price"],
                "pnl": pnl
            })

rt = pd.DataFrame(rounds)

if rt.empty:
    print("⚠️ No completed round trips found.")
else:
    wins = (rt["pnl"] > 0).sum()
    losses = (rt["pnl"] <= 0).sum()
    total = len(rt)
    win_rate = wins / total * 100
    loss_rate = losses / total * 100
    avg_win = rt.loc[rt["pnl"] > 0, "pnl"].mean() if wins > 0 else 0
    avg_loss = rt.loc[rt["pnl"] <= 0, "pnl"].mean() if losses > 0 else 0

    print(f"Total closed trades: {total}")
    print(f"Wins: {wins} ({win_rate:.1f}%) | Avg win: {avg_win:,.2f}")
    print(f"Losses: {losses} ({loss_rate:.1f}%) | Avg loss: {avg_loss:,.2f}")
    print(f"Net PnL: {rt['pnl'].sum():,.2f}")

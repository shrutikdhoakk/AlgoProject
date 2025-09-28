import os, sys
import pandas as pd

infile  = sys.argv[1] if len(sys.argv) > 1 else "nifty500.csv"
outfile = sys.argv[2] if len(sys.argv) > 2 else os.path.join("data","symbols_nifty500_clean.csv")

df = pd.read_csv(infile)
col = next((c for c in df.columns if c.strip().lower() in ("symbol","symbols","ticker","tickers")), df.columns[0])

# 1) normalize
syms = df[col].astype(str).str.strip().str.upper()

# 2) drop common Yahoo suffixes if present (keep left of first dot)
base = syms.str.split(".", n=1, expand=True)[0]

# 3) keep only sane tickers: letters/digits/&/-
def ok_sym(s: str) -> bool:
    return len(s) > 0 and all(ch.isalnum() or ch in "&-" for ch in s)

clean = base[base.apply(ok_sym)].drop_duplicates().sort_values()

os.makedirs(os.path.dirname(outfile), exist_ok=True)
pd.DataFrame({"symbol": clean}).to_csv(outfile, index=False)
print(f"Wrote {outfile} with {clean.shape[0]} symbols.")

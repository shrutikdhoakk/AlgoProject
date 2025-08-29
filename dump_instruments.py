# dump_instruments.py
import os, pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect

if os.path.exists(".env"):
    load_dotenv(".env")

api_key = os.getenv("KITE_API_KEY")
access_token = os.getenv("KITE_ACCESS_TOKEN")
if not api_key or not access_token:
    raise SystemExit("Set KITE_API_KEY and KITE_ACCESS_TOKEN in .env or env vars.")

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

rows = kite.instruments()  # all exchanges; or kite.instruments("NSE")
df = pd.DataFrame(rows)
need = {"instrument_token","tradingsymbol","exchange"}
missing = need - set(map(str, df.columns))
if missing:
    raise SystemExit(f"Missing columns in instruments dump: {missing}")

df.to_csv("kite_instruments.csv", index=False)
print(f"Saved {len(df)} instruments to kite_instruments.csv")

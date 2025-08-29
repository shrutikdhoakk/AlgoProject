# tv_webhook.py
from __future__ import annotations
import os, json, uvicorn
from typing import Optional
from fastapi import FastAPI, Header, Request, HTTPException
from pydantic import BaseModel

# ---- your engine imports (adjust paths to your project) ----
from config import load_config, AppConfig
from risk_engine import RiskEngine
from order_manager import OrderManager
from utils import setup_logger
from trade_signal import TradeSignal
from kiteconnect import KiteConnect

TV_SECRET = os.getenv("TV_SHARED_SECRET", "TV_SHARED_SECRET")
ENV = os.getenv("ENV", "SIM")  # SIM or LIVE

app = FastAPI()
logger = setup_logger("tv_webhook", "logs/tv_webhook.log")

# lazy singletons
_cfg: Optional[AppConfig] = None
_risk: Optional[RiskEngine] = None
_om: Optional[OrderManager] = None

def _init_once():
    global _cfg, _risk, _om
    if _cfg is None:
        _cfg = load_config("config.yaml")
        kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
        # Access token is required only for LIVE
        if ENV == "LIVE":
            kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN", ""))
        _risk = RiskEngine(_cfg)
        _om = OrderManager(_cfg, _risk, kite)
    return _cfg, _risk, _om

class TVAlert(BaseModel):
    secret: str
    algo_id: Optional[str] = None
    symbol: str
    exchange: Optional[str] = None
    side: str           # BUY or SELL
    qty_type: Optional[str] = "SHARES"  # or "INR" for notional
    qty: float          # shares or notional, depending on qty_type
    price: Optional[float] = None
    time: Optional[str] = None
    note: Optional[str] = None

def map_tv_symbol_to_kite(tv_symbol: str, exchange_hint: Optional[str]) -> str:
    """
    TradingView symbols often look like NSE:RELIANCE or RELIANCE.
    Zerodha expects tradingsymbol like 'RELIANCE' for equity cash.
    Adjust here if you trade F&O (e.g., 'RELIANCE24AUGFUT').
    """
    s = tv_symbol.split(":")[-1].upper().strip()
    # remove odd suffixes if needed
    return s

@app.post("/webhook")
async def webhook(body: TVAlert, x_tradingview_signature: Optional[str] = Header(default=None)):
    # 1) auth
    if body.secret != TV_SECRET:
        logger.warning("Bad secret in webhook")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 2) init engine objects
    cfg, risk, om = _init_once()

    # 3) symbol normalization
    symbol = map_tv_symbol_to_kite(body.symbol, body.exchange)

    # 4) turn alert into TradeSignal
    side = body.side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="Invalid side")

    # qty handling
    quantity = int(round(body.qty)) if body.qty_type == "SHARES" else 0

    ts = TradeSignal(
        symbol=symbol,
        side=side,
        quantity=quantity,
        time_in_force="DAY",
        limit_price=body.price,
        algo_id=body.algo_id or "tv_alert"
    )

    logger.info(f"TV order incoming: {ts}")

    # 5) risk check (optional)
    ok, reason = risk.check(ts) if hasattr(risk, "check") else (True, "no risk engine")
    if not ok:
        logger.warning(f"Risk rejected: {reason}")
        raise HTTPException(status_code=400, detail=f"Risk rejected: {reason}")

    # 6) route order
    try:
        order_id = om.place_order(ts)  # ensure your OrderManager has place_order(TradeSignal)
        logger.info(f"Order placed {order_id} | {ts}")
        return {"status": "ok", "order_id": order_id, "symbol": symbol, "env": ENV}
    except Exception as e:
        logger.exception("Order placement failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run:  uvicorn tv_webhook:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

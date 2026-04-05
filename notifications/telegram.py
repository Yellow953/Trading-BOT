import logging
import requests
from decimal import Decimal
import config

logger = logging.getLogger(__name__)

_API_BASE = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"


def send_message(text: str) -> None:
    """Send a Telegram message. Never raises — a Telegram outage must not crash the bot."""
    try:
        resp = requests.post(
            f"{_API_BASE}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            logger.warning("Telegram send failed: %s %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.warning("Telegram send exception (ignored): %s", exc)


# --- Alert formatters ---

def fmt_entry(
    entry_price: Decimal,
    sl_price: Decimal,
    tp_price: Decimal,
    quantity: Decimal,
) -> str:
    return (
        f"<b>LONG ENTRY</b>\n"
        f"Price:    <code>{entry_price:.2f}</code> USDT\n"
        f"Qty:      <code>{quantity}</code> BTC\n"
        f"SL:       <code>{sl_price:.2f}</code> USDT\n"
        f"TP:       <code>{tp_price:.2f}</code> USDT"
    )


def fmt_exit(exit_price: Decimal, entry_price: Decimal, quantity: Decimal) -> str:
    pnl = (exit_price - entry_price) * quantity
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    sign = "+" if pnl >= 0 else ""
    return (
        f"<b>POSITION CLOSED</b>\n"
        f"Exit price: <code>{exit_price:.2f}</code> USDT\n"
        f"PnL:        <code>{sign}{pnl:.4f}</code> USDT ({sign}{pnl_pct:.2f}%)"
    )


def fmt_daily_halt(loss_pct: Decimal) -> str:
    return (
        f"<b>DAILY LOSS LIMIT HIT</b>\n"
        f"Loss: <code>{float(loss_pct) * 100:.2f}%</code>\n"
        f"Trading paused until next UTC midnight."
    )


def fmt_expired_close(entry_price: Decimal, quantity: Decimal) -> str:
    return (
        f"<b>MAX HOLD TIME REACHED</b>\n"
        f"Closing position after {config.MAX_HOLD_HOURS}h.\n"
        f"Entry: <code>{entry_price:.2f}</code> USDT | Qty: <code>{quantity}</code> BTC"
    )


def fmt_grid_buy_fill(level_price: Decimal, sell_level: Decimal, quantity: Decimal) -> str:
    return (
        f"<b>GRID BUY FILLED</b>\n"
        f"Bought: <code>{quantity}</code> BTC @ <code>{level_price:.2f}</code> USDT\n"
        f"Sell placed @ <code>{sell_level:.2f}</code> USDT"
    )


def fmt_grid_cycle(
    buy_price: Decimal,
    sell_price: Decimal,
    quantity: Decimal,
    profit: Decimal,
    total_pnl: Decimal,
    cycles: int,
) -> str:
    return (
        f"<b>GRID CYCLE COMPLETE</b>\n"
        f"Buy: <code>{buy_price:.2f}</code>  Sell: <code>{sell_price:.2f}</code>  "
        f"Qty: <code>{quantity}</code> BTC\n"
        f"Profit: <code>+{profit:.4f}</code> USDT\n"
        f"Total PnL: <code>{total_pnl:.4f}</code> USDT | Cycles: {cycles}"
    )


def fmt_grid_summary(
    realized: Decimal,
    unrealized: Decimal,
    cycles: int,
    open_buys: int,
    open_sells: int,
) -> str:
    total = realized + unrealized
    sign = "+" if total >= 0 else ""
    return (
        f"<b>GRID DAILY SUMMARY</b>\n"
        f"Realized PnL:   <code>+{realized:.4f}</code> USDT\n"
        f"Unrealized PnL: <code>{unrealized:+.4f}</code> USDT\n"
        f"Total:          <code>{sign}{total:.4f}</code> USDT\n"
        f"Cycles: {cycles} | Open buys: {open_buys} | Open sells: {open_sells}"
    )

import logging
from decimal import Decimal, ROUND_DOWN
from binance.client import Client
from binance.exceptions import BinanceAPIException
import config

logger = logging.getLogger(__name__)

_client = Client(
    config.BINANCE_API_KEY,
    config.BINANCE_API_SECRET,
    testnet=config.BINANCE_TESTNET,
)


def _quantize_price(price: Decimal) -> Decimal:
    return price.quantize(config.PRICE_TICK, rounding=ROUND_DOWN)


def _quantize_qty(qty: Decimal) -> Decimal:
    return qty.quantize(config.QUANTITY_STEP, rounding=ROUND_DOWN)


def get_usdt_balance() -> Decimal:
    """Return the free USDT balance."""
    try:
        bal = _client.get_asset_balance(asset="USDT")
        return Decimal(bal["free"])
    except BinanceAPIException as exc:
        logger.error("Failed to fetch USDT balance: %s", exc)
        raise


def place_market_buy(quantity: Decimal) -> dict:
    """Place a market buy order. Returns the full order response including fills.

    Raises BinanceAPIException on failure.
    """
    qty_str = str(_quantize_qty(quantity))
    logger.info("Placing market BUY: qty=%s %s", qty_str, config.SYMBOL)
    try:
        order = _client.create_order(
            symbol=config.SYMBOL,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=qty_str,
            newOrderRespType=Client.ORDER_RESP_TYPE_FULL,
        )
        logger.info("Market BUY placed: orderId=%s", order.get("orderId"))
        return order
    except BinanceAPIException as exc:
        logger.error("Market BUY failed: %s", exc)
        raise


def place_oco_sell(
    quantity: Decimal,
    tp_price: Decimal,
    sl_trigger: Decimal,
    sl_limit: Decimal,
) -> dict:
    """Place an OCO sell order to simultaneously set SL and TP.

    Binance enforces for SELL OCO: tp_price > sl_trigger > sl_limit.

    Args:
        quantity:   BTC amount to sell
        tp_price:   Limit price for take-profit (above current price)
        sl_trigger: Stop price that triggers the stop-limit order
        sl_limit:   Limit price for the stop order (set slightly below sl_trigger)

    Returns the Binance OCO response dict.
    Raises BinanceAPIException on failure.
    """
    qty_str = str(_quantize_qty(quantity))
    tp_str = str(_quantize_price(tp_price))
    sl_trigger_str = str(_quantize_price(sl_trigger))
    sl_limit_str = str(_quantize_price(sl_limit))

    logger.info(
        "Placing OCO SELL: qty=%s TP=%s SL_trigger=%s SL_limit=%s",
        qty_str, tp_str, sl_trigger_str, sl_limit_str,
    )
    try:
        order = _client.create_oco_order(
            symbol=config.SYMBOL,
            side=Client.SIDE_SELL,
            quantity=qty_str,
            price=tp_str,
            stopPrice=sl_trigger_str,
            stopLimitPrice=sl_limit_str,
            stopLimitTimeInForce=Client.TIME_IN_FORCE_GTC,
        )
        logger.info("OCO SELL placed: orderListId=%s", order.get("orderListId"))
        return order
    except BinanceAPIException as exc:
        logger.error("OCO SELL failed: %s", exc)
        raise


def place_market_sell(quantity: Decimal) -> dict:
    """Place a market sell order (used for forced exits: max hold time, exit signal)."""
    qty_str = str(_quantize_qty(quantity))
    logger.info("Placing market SELL: qty=%s %s", qty_str, config.SYMBOL)
    try:
        order = _client.create_order(
            symbol=config.SYMBOL,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=qty_str,
            newOrderRespType=Client.ORDER_RESP_TYPE_FULL,
        )
        logger.info("Market SELL placed: orderId=%s", order.get("orderId"))
        return order
    except BinanceAPIException as exc:
        logger.error("Market SELL failed: %s", exc)
        raise


def place_limit_buy(price: Decimal, quantity: Decimal) -> dict:
    """Place a GTC limit buy order. Returns the full order response."""
    price_str = str(_quantize_price(price))
    qty_str = str(_quantize_qty(quantity))
    logger.info("Placing limit BUY: qty=%s @ %s %s", qty_str, price_str, config.SYMBOL)
    try:
        order = _client.create_order(
            symbol=config.SYMBOL,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=qty_str,
            price=price_str,
        )
        logger.info("Limit BUY placed: orderId=%s", order.get("orderId"))
        return order
    except BinanceAPIException as exc:
        logger.error("Limit BUY failed: %s", exc)
        raise


def place_limit_sell(price: Decimal, quantity: Decimal) -> dict:
    """Place a GTC limit sell order. Returns the full order response."""
    price_str = str(_quantize_price(price))
    qty_str = str(_quantize_qty(quantity))
    logger.info("Placing limit SELL: qty=%s @ %s %s", qty_str, price_str, config.SYMBOL)
    try:
        order = _client.create_order(
            symbol=config.SYMBOL,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=qty_str,
            price=price_str,
        )
        logger.info("Limit SELL placed: orderId=%s", order.get("orderId"))
        return order
    except BinanceAPIException as exc:
        logger.error("Limit SELL failed: %s", exc)
        raise


def get_open_order_ids() -> set:
    """Return the set of open order IDs for the configured symbol."""
    try:
        open_orders = _client.get_open_orders(symbol=config.SYMBOL)
        return {order["orderId"] for order in open_orders}
    except BinanceAPIException as exc:
        logger.error("Failed to fetch open orders: %s", exc)
        raise


def get_current_price() -> Decimal:
    """Return the latest trade price for the configured symbol."""
    try:
        ticker = _client.get_symbol_ticker(symbol=config.SYMBOL)
        return Decimal(ticker["price"])
    except BinanceAPIException as exc:
        logger.error("Failed to fetch current price: %s", exc)
        raise


def cancel_all_open_orders() -> None:
    """Cancel all open orders for the configured symbol.

    Binance spot has no batch cancel endpoint — orders are cancelled individually.
    """
    try:
        open_orders = _client.get_open_orders(symbol=config.SYMBOL)
        if not open_orders:
            logger.debug("No open orders to cancel")
            return
        for order in open_orders:
            try:
                _client.cancel_order(
                    symbol=config.SYMBOL,
                    orderId=order["orderId"],
                )
                logger.info("Cancelled order %s", order["orderId"])
            except BinanceAPIException as exc:
                # Order may have already filled — log and continue
                logger.warning("Could not cancel order %s: %s", order["orderId"], exc)
    except BinanceAPIException as exc:
        logger.error("Failed to fetch open orders for cancellation: %s", exc)
        raise

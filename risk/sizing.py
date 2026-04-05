from decimal import Decimal, ROUND_DOWN
import config


def calculate_quantity(balance: Decimal, entry_price: Decimal) -> Decimal:
    """Return the BTC quantity to buy based on 10% of available balance.

    Truncates (ROUND_DOWN) to the Binance LOT_SIZE stepSize of 0.00001 BTC.
    Raises ValueError if the resulting notional is below Binance's minimum.
    """
    raw_quantity = (balance * config.POSITION_SIZE_PCT) / entry_price
    quantity = raw_quantity.quantize(config.QUANTITY_STEP, rounding=ROUND_DOWN)

    notional = quantity * entry_price
    if notional < config.MIN_NOTIONAL:
        raise ValueError(
            f"Position notional {notional} USDT is below minimum {config.MIN_NOTIONAL} USDT. "
            f"Increase balance or reduce position size."
        )

    return quantity

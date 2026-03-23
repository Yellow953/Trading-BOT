"""ML Trading Bot CLI."""
import logging
import click
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@click.group()
@click.option("--config", default="config.yaml", help="Path to config file.")
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """ML Trading Bot."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    log_level = ctx.obj["config"]["general"].get("log_level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level))


@cli.command()
@click.option("--market", required=True, type=click.Choice(["crypto", "stocks", "forex"]))
@click.option("--symbol", required=True)
@click.option("--timeframe", default="1d")
@click.option("--start", default=None, help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD")
@click.pass_context
def fetch(ctx: click.Context, market: str, symbol: str, timeframe: str, start: str | None, end: str | None) -> None:
    """Fetch and cache market data."""
    from src.data.providers.crypto import CryptoProvider
    from src.data.providers.stocks import StocksProvider
    from src.data.providers.forex import ForexProvider
    from src.data.cache import DataCache
    from src.data.preprocess import preprocess
    from datetime import datetime, timezone
    import rich.console

    console = rich.console.Console()
    config = ctx.obj["config"]
    cache = DataCache(config["cache"]["db_path"])

    providers = {
        "crypto": CryptoProvider,
        "stocks": StocksProvider,
        "forex": ForexProvider,
    }
    provider = providers[market](config)

    since = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else datetime(2023, 1, 1, tzinfo=timezone.utc)
    until = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)

    df = cache.get_or_fetch(provider, symbol, timeframe, since, until)
    df = preprocess(df)

    console.print(f"[green]Fetched {len(df)} rows for {symbol} ({timeframe})[/green]")
    console.print(f"Date range: {df.index[0]} → {df.index[-1]}")


@cli.command()
@click.option("--market", default=None, type=click.Choice(["crypto", "stocks", "forex"]))
@click.option("--symbol", default=None)
@click.option("--strategy", default=None)
@click.option("--start", default=None)
@click.option("--end", default=None)
@click.pass_context
def backtest(ctx: click.Context, market: str | None, symbol: str | None, strategy: str | None, start: str | None, end: str | None) -> None:
    """Run walk-forward backtest. (Phase 2)"""
    click.echo("Backtest not yet implemented (Phase 2).")


@cli.command()
@click.option("--market", default=None, type=click.Choice(["crypto", "stocks", "forex"]))
@click.pass_context
def train(ctx: click.Context, market: str | None) -> None:
    """Train/retrain models. (Phase 2)"""
    click.echo("Train not yet implemented (Phase 2).")


@cli.command()
@click.pass_context
def compare(ctx: click.Context) -> None:
    """Show strategy competition results. (Phase 3)"""
    click.echo("Compare not yet implemented (Phase 3).")


@cli.command()
@click.pass_context
def paper(ctx: click.Context) -> None:
    """Start paper trading loop. (Phase 4)"""
    click.echo("Paper trading not yet implemented (Phase 4).")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show paper trading portfolio status. (Phase 4)"""
    click.echo("Status not yet implemented (Phase 4).")


@cli.command()
@click.option("--format", "fmt", default="terminal", type=click.Choice(["terminal", "csv"]))
@click.pass_context
def report(ctx: click.Context, fmt: str) -> None:
    """Export results. (Phase 2+)"""
    click.echo("Report not yet implemented (Phase 2+).")


if __name__ == "__main__":
    cli()

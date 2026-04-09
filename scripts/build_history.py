"""Build historical data cache.

Fetches extended OHLCV data and stores in SQLite cache.

Usage:
    python scripts/build_history.py --symbol BTC/USDT:USDT --timeframe 1h --days 365
    python scripts/build_history.py --all --days 180
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from hfi.core.config import load_config
from hfi.data.cache import CandleCache
from hfi.data.historical import fetch_historical_ohlcv, fetch_daily_coingecko
from hfi.exchange.client import ExchangeClient

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


async def main(symbols: list[str], timeframe: str, days: int) -> None:
    config = load_config("config")
    client = ExchangeClient(config.exchange)
    cache = CandleCache()
    await cache.init_db()
    await client.connect()

    for symbol in symbols:
        console.print(f"\n[bold cyan]Fetching {symbol} {timeframe} ({days} days)...[/bold cyan]")

        df = await fetch_historical_ohlcv(
            client, symbol, timeframe, days=days, cache=cache,
        )

        if df.empty:
            console.print(f"[red]No data for {symbol}[/red]")
            continue

        count = await cache.count(symbol, timeframe)
        console.print(
            f"[green]Done:[/green] {len(df)} new bars fetched. "
            f"Cache total: {count} bars "
            f"({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})"
        )

    # Also fetch daily data for market condition labeling
    console.print(f"\n[bold cyan]Fetching daily data from CoinGecko for condition labeling...[/bold cyan]")
    for symbol in symbols:
        daily_df = await fetch_daily_coingecko(symbol, days=min(days, 365))
        if not daily_df.empty:
            await cache.store(symbol, "1d", daily_df)
            console.print(f"  [green]{symbol}[/green]: {len(daily_df)} daily bars cached")

    await client.close()
    console.print("\n[bold green]History build complete![/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build historical data cache")
    parser.add_argument("--symbol", default=None, help="Single symbol to fetch")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--all", action="store_true", help="Fetch all configured pairs")
    args = parser.parse_args()

    config = load_config("config")
    if args.symbol:
        symbols = [args.symbol]
    elif args.all:
        symbols = config.exchange.pairs
    else:
        symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

    asyncio.run(main(symbols, args.timeframe, args.days))

"""Run backtest for all engines with walk-forward validation.

Usage:
    python scripts/run_backtest.py                    # All engines, BTC
    python scripts/run_backtest.py --symbol ETH/USDT:USDT
    python scripts/run_backtest.py --engine TREND_FOLLOWER
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from hfi.backtest.runner import run_backtest
from hfi.backtest.validator import walk_forward_validate
from hfi.core.config import load_config
from hfi.data.fetcher import fetch_ohlcv_df
from hfi.exchange.client import ExchangeClient

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")


async def main(symbol: str, engine: str | None, config_dir: str) -> None:
    config = load_config(config_dir)
    client = ExchangeClient(config.exchange)
    await client.connect()

    engines = [engine] if engine else ["TREND_FOLLOWER", "MEAN_REVERSION", "MOMENTUM_SCALPER"]

    # Fetch data
    console.print(f"\n[bold]Fetching OHLCV data for {symbol}...[/bold]")
    df = await fetch_ohlcv_df(client, symbol, "1h", limit=1000)
    await client.close()

    if df.empty:
        console.print("[red]No data fetched. Check your connection.[/red]")
        return

    console.print(f"Got {len(df)} candles from {df.index[0]} to {df.index[-1]}\n")

    # Results table
    results_table = Table(title="Backtest Results", border_style="cyan")
    results_table.add_column("Engine")
    results_table.add_column("Trades", justify="right")
    results_table.add_column("Return", justify="right")
    results_table.add_column("Sharpe", justify="right")
    results_table.add_column("Sortino", justify="right")
    results_table.add_column("Max DD", justify="right")
    results_table.add_column("Win Rate", justify="right")
    results_table.add_column("PF", justify="right")

    for eng in engines:
        console.print(f"[bold]Running backtest: {eng}[/bold]")
        result = run_backtest(df.copy(), config, eng)

        color = "green" if result.total_return > 0 else "red"
        sharpe_color = "green" if result.sharpe_ratio > 1.0 else "yellow" if result.sharpe_ratio > 0 else "red"

        results_table.add_row(
            eng,
            str(result.total_trades),
            f"[{color}]{result.total_return:+.2%}[/{color}]",
            f"[{sharpe_color}]{result.sharpe_ratio:.2f}[/{sharpe_color}]",
            f"{result.sortino_ratio:.2f}",
            f"[{'red' if result.max_drawdown > 0.15 else 'yellow'}]{result.max_drawdown:.2%}[/]",
            f"{result.win_rate:.1%}",
            f"{result.profit_factor:.2f}",
        )

    console.print(results_table)

    # Walk-forward validation for best engine
    console.print(f"\n[bold]Walk-Forward Validation (TREND_FOLLOWER)...[/bold]")
    wf = walk_forward_validate(df.copy(), config, "TREND_FOLLOWER")
    console.print(f"  OOS Sharpe: {wf.oos_sharpe:.2f}")
    console.print(f"  OOS Return: {wf.oos_total_return:+.2%}")
    console.print(f"  OOS Max DD: {wf.oos_max_drawdown:.2%}")
    console.print(f"  Deflated Sharpe: {wf.deflated_sharpe:.2f}")
    console.print(f"  Overfit: {'[red]YES[/red]' if wf.is_overfit else '[green]NO[/green]'}")
    console.print(f"  Folds: {wf.num_folds} | Total OOS trades: {wf.oos_total_trades}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFI Backtest Runner")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Trading pair")
    parser.add_argument("--engine", default=None, help="Specific engine to test")
    parser.add_argument("--config", default="config", help="Config directory")
    args = parser.parse_args()

    asyncio.run(main(args.symbol, args.engine, args.config))

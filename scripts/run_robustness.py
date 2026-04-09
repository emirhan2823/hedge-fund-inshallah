"""Run robustness test suite.

Usage:
    python scripts/run_robustness.py --symbol BTC/USDT:USDT --engine TREND_FOLLOWER --iterations 20
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from hfi.backtest.robustness import RobustnessValidator
from hfi.core.config import load_config
from hfi.data.cache import CandleCache
from hfi.data.historical import fetch_historical_ohlcv, fetch_daily_coingecko
from hfi.exchange.client import ExchangeClient

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


async def main(symbol: str, engine: str, iterations: int, days: int) -> None:
    config = load_config("config")
    client = ExchangeClient(config.exchange)
    cache = CandleCache()
    await cache.init_db()
    await client.connect()

    # Fetch historical data
    console.print(f"\n[bold]Fetching {days} days of hourly data for {symbol}...[/bold]")
    hourly_df = await fetch_historical_ohlcv(client, symbol, "1h", days=days, cache=cache)

    console.print(f"[bold]Fetching daily data from CoinGecko for condition labeling...[/bold]")
    daily_df = await fetch_daily_coingecko(symbol, days=min(days, 365))

    await client.close()

    if hourly_df.empty or daily_df.empty:
        console.print("[red]Not enough data for robustness test[/red]")
        return

    console.print(f"Hourly data: {len(hourly_df)} bars ({hourly_df.index[0].date()} to {hourly_df.index[-1].date()})")
    console.print(f"Daily data: {len(daily_df)} bars")

    # Run robustness test
    console.print(f"\n[bold cyan]Running robustness test: {engine} on {symbol} ({iterations} iterations)...[/bold cyan]")

    validator = RobustnessValidator()
    result = validator.run_robustness_test(
        hourly_data={symbol: hourly_df},
        daily_data=daily_df,
        config=config,
        engine_name=engine,
        symbol=symbol,
        n_iterations=iterations,
        seed=42,
    )

    # Display results
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print(f"[bold]ROBUSTNESS TEST: {engine} on {symbol}[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]")

    # Verdict
    color = "green" if result.is_robust else "red"
    console.print(f"\n[bold {color}]VERDICT: {result.verdict}[/bold {color}]")

    # Summary table
    table = Table(title="Overall Metrics", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Pass Rate", f"{result.pass_rate:.0%}")
    table.add_row("Avg Return", f"{result.avg_return:+.2%}")
    table.add_row("Std Return", f"{result.std_return:.2%}")
    table.add_row("Avg Sharpe", f"{result.avg_sharpe:.2f}")
    table.add_row("Avg Max DD", f"{result.avg_max_dd:.2%}")
    table.add_row("Avg Win Rate", f"{result.avg_win_rate:.0%}")
    table.add_row("Avg Profit Factor", f"{result.avg_profit_factor:.2f}")
    table.add_row("Total Trades", str(result.total_trades))
    table.add_row("OOS Degradation", f"{result.oos_degradation_pct:.0%}")
    table.add_row("MC DD (95th)", f"${result.mc_dd_95th:.2f}")
    table.add_row("MC DD (Actual)", f"${result.mc_dd_actual:.2f}")
    table.add_row("Overfit?", "[red]YES[/red]" if result.is_overfit else "[green]NO[/green]")
    console.print(table)

    # Condition breakdown
    if result.condition_breakdown:
        cond_table = Table(title="Performance by Market Condition", border_style="yellow")
        cond_table.add_column("Condition")
        cond_table.add_column("Samples", justify="right")
        cond_table.add_column("Avg Return", justify="right")
        cond_table.add_column("Avg Sharpe", justify="right")
        cond_table.add_column("Pass Rate", justify="right")

        for cond, data in result.condition_breakdown.items():
            ret_color = "green" if data["avg_return"] > 0 else "red"
            cond_table.add_row(
                cond,
                str(data["n_samples"]),
                f"[{ret_color}]{data['avg_return']:+.2%}[/{ret_color}]",
                f"{data['avg_sharpe']:.2f}",
                f"{data['pass_rate']:.0%}",
            )
        console.print(cond_table)

    # Save report
    report_dir = Path("data/results/robustness")
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"{engine}_{symbol.replace('/', '').replace(':', '')}_{ts}.json"

    report = {
        "engine": engine,
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": result.verdict,
        "pass_rate": result.pass_rate,
        "avg_return": result.avg_return,
        "std_return": result.std_return,
        "avg_sharpe": result.avg_sharpe,
        "avg_max_dd": result.avg_max_dd,
        "is_robust": result.is_robust,
        "is_overfit": result.is_overfit,
        "oos_degradation_pct": result.oos_degradation_pct,
        "mc_dd_95th": result.mc_dd_95th,
        "mc_dd_actual": result.mc_dd_actual,
        "condition_breakdown": result.condition_breakdown,
        "iterations": result.iterations,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    console.print(f"\n[dim]Report saved: {report_path}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFI Robustness Test")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Trading pair")
    parser.add_argument("--engine", default="TREND_FOLLOWER", help="Engine to test")
    parser.add_argument("--iterations", type=int, default=20, help="Number of random iterations")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data")
    args = parser.parse_args()

    asyncio.run(main(args.symbol, args.engine, args.iterations, args.days))

"""Run engine-specific performance analysis.

Usage:
    python scripts/analyze_engine.py --symbol BTC/USDT:USDT --engine TREND_FOLLOWER
    python scripts/analyze_engine.py --results-dir data/results/TREND_FOLLOWER_BTCUSDT_20260410/
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from rich.console import Console
from rich.table import Table

from hfi.analysis.engine_analyzer import EngineAnalyzer
from hfi.backtest.csv_export import create_run_dir, export_trades_csv, export_analysis_json
from hfi.backtest.runner import run_backtest
from hfi.core.config import load_config
from hfi.data.fetcher import fetch_ohlcv_df
from hfi.exchange.client import ExchangeClient

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


async def main(symbol: str, engine: str, results_dir: str | None) -> None:
    config = load_config("config")
    analyzer = EngineAnalyzer()

    if results_dir:
        # Load from existing CSV
        csv_path = Path(results_dir) / "trades.csv"
        if not csv_path.exists():
            console.print(f"[red]trades.csv not found in {results_dir}[/red]")
            return
        df = pd.read_csv(csv_path)
        trades = df.to_dict("records")
    else:
        # Run fresh backtest
        client = ExchangeClient(config.exchange)
        await client.connect()
        console.print(f"[bold]Fetching data and running backtest: {engine} on {symbol}...[/bold]")
        ohlcv = await fetch_ohlcv_df(client, symbol, "1h", limit=1000)
        await client.close()

        result = run_backtest(ohlcv.copy(), config, engine, symbol=symbol)
        trades = result.trades

        # Export
        if result.total_trades > 0:
            run_dir = create_run_dir(engine, symbol)
            export_trades_csv(result, engine, symbol, run_dir)
            export_analysis_json(result, engine, symbol, run_dir)
            console.print(f"[dim]Results saved to {run_dir}[/dim]")

    if not trades:
        console.print("[red]No trades to analyze[/red]")
        return

    # Run analysis
    report = analyzer.analyze(trades, engine, symbol)

    # Display
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print(f"[bold]ENGINE ANALYSIS: {report.engine} on {report.symbol}[/bold]")
    console.print(f"[bold]Total Trades: {report.total_trades}[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]")

    # Metrics table
    metrics_table = Table(title="Performance Metrics", border_style="cyan")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")

    for key, value in report.metrics.items():
        if isinstance(value, float):
            if "rate" in key or "pct" in key or "wr_" in key:
                metrics_table.add_row(key, f"{value:.1%}")
            else:
                metrics_table.add_row(key, f"{value:.4f}")
        else:
            metrics_table.add_row(key, str(value))
    console.print(metrics_table)

    # Direction stats
    if report.direction_stats:
        dir_table = Table(title="Direction Breakdown", border_style="green")
        dir_table.add_column("Direction")
        dir_table.add_column("Count", justify="right")
        dir_table.add_column("Win Rate", justify="right")
        dir_table.add_column("Avg PnL", justify="right")
        dir_table.add_column("Total PnL", justify="right")

        for d, stats in report.direction_stats.items():
            color = "green" if stats["total_pnl"] > 0 else "red"
            dir_table.add_row(
                d.upper(),
                str(stats["count"]),
                f"{stats['win_rate']:.0%}",
                f"[{color}]${stats['avg_pnl']:+.2f}[/{color}]",
                f"[{color}]${stats['total_pnl']:+.2f}[/{color}]",
            )
        console.print(dir_table)

    # Exit reasons
    if report.exit_reason_stats:
        exit_table = Table(title="Exit Reason Breakdown", border_style="yellow")
        exit_table.add_column("Reason")
        exit_table.add_column("Count", justify="right")
        exit_table.add_column("Wins", justify="right")
        exit_table.add_column("Win Rate", justify="right")
        exit_table.add_column("Total PnL", justify="right")

        for reason, stats in report.exit_reason_stats.items():
            wr = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
            color = "green" if stats["total_pnl"] > 0 else "red"
            exit_table.add_row(
                reason,
                str(stats["count"]),
                str(stats["wins"]),
                f"{wr:.0%}",
                f"[{color}]${stats['total_pnl']:+.2f}[/{color}]",
            )
        console.print(exit_table)

    # Recommendations
    if report.recommendations:
        console.print(f"\n[bold yellow]RECOMMENDATIONS:[/bold yellow]")
        for i, rec in enumerate(report.recommendations, 1):
            console.print(f"  {i}. {rec}")

    if report.parameter_suggestions:
        console.print(f"\n[bold cyan]SUGGESTED PARAMETER CHANGES:[/bold cyan]")
        for param, value in report.parameter_suggestions.items():
            console.print(f"  {param}: {value}")

    # Save analysis JSON
    if not results_dir:
        results_dir = "data/results"
    analysis_path = Path(results_dir) / f"engine_analysis_{engine}.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_data = {
        "engine": report.engine,
        "symbol": report.symbol,
        "total_trades": report.total_trades,
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in report.metrics.items()},
        "recommendations": report.recommendations,
        "parameter_suggestions": report.parameter_suggestions,
        "direction_stats": report.direction_stats,
        "exit_reason_stats": report.exit_reason_stats,
    }
    analysis_path.write_text(json.dumps(analysis_data, indent=2))
    console.print(f"\n[dim]Analysis saved: {analysis_path}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFI Engine Analyzer")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Trading pair")
    parser.add_argument("--engine", default="TREND_FOLLOWER", help="Engine to analyze")
    parser.add_argument("--results-dir", default=None, help="Load trades from existing results dir")
    args = parser.parse_args()

    asyncio.run(main(args.symbol, args.engine, args.results_dir))

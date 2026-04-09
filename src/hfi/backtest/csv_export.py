"""Trade CSV and analysis JSON export.

Every backtest run gets a unique timestamped directory.
NEVER overwrites previous results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hfi.backtest.runner import BacktestResult

logger = logging.getLogger(__name__)


def create_run_dir(
    engine: str,
    symbol: str,
    base_dir: Path = Path("data/results"),
) -> Path:
    """Create unique timestamped output directory."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_symbol = symbol.replace("/", "").replace(":", "")
    dir_name = f"{engine}_{clean_symbol}_{ts}"
    run_dir = base_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def export_trades_csv(
    result: BacktestResult,
    engine: str,
    symbol: str,
    output_dir: Path | None = None,
) -> Path | None:
    """Export trades to CSV. Returns path to CSV file."""
    if not result.trades:
        logger.warning("No trades to export for %s %s", engine, symbol)
        return None

    if output_dir is None:
        output_dir = create_run_dir(engine, symbol)

    df = pd.DataFrame(result.trades)
    csv_path = output_dir / "trades.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Exported %d trades to %s", len(result.trades), csv_path)
    return csv_path


def export_analysis_json(
    result: BacktestResult,
    engine: str,
    symbol: str,
    output_dir: Path | None = None,
) -> Path | None:
    """Export analysis summary to JSON."""
    if output_dir is None:
        output_dir = create_run_dir(engine, symbol)

    analysis = {
        "engine": engine,
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "total_return": round(result.total_return, 6),
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "sortino_ratio": round(result.sortino_ratio, 4),
            "max_drawdown": round(result.max_drawdown, 6),
            "win_rate": round(result.win_rate, 4),
            "profit_factor": round(result.profit_factor, 4),
            "total_trades": result.total_trades,
            "avg_trade_pnl": round(result.avg_trade_pnl, 4),
            "avg_win": round(result.avg_win, 4),
            "avg_loss": round(result.avg_loss, 4),
            "best_trade": round(result.best_trade, 4),
            "worst_trade": round(result.worst_trade, 4),
            "avg_bars_held": round(result.avg_bars_held, 2),
        },
        "exit_reasons": _count_exit_reasons(result.trades),
        "direction_breakdown": _direction_breakdown(result.trades),
    }

    json_path = output_dir / "analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2))
    logger.info("Exported analysis to %s", json_path)
    return json_path


def _count_exit_reasons(trades: list[dict]) -> dict[str, int]:
    """Count trades by exit reason."""
    counts: dict[str, int] = {}
    for t in trades:
        reason = t.get("exit_reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _direction_breakdown(trades: list[dict]) -> dict:
    """Breakdown win rate by direction."""
    longs = [t for t in trades if t.get("direction") == "long"]
    shorts = [t for t in trades if t.get("direction") == "short"]

    def _stats(tlist: list[dict]) -> dict:
        if not tlist:
            return {"count": 0, "win_rate": 0.0, "avg_pnl": 0.0}
        wins = sum(1 for t in tlist if t.get("pnl_usd", t.get("pnl", 0)) > 0)
        avg = sum(t.get("pnl_usd", t.get("pnl", 0)) for t in tlist) / len(tlist)
        return {"count": len(tlist), "win_rate": round(wins / len(tlist), 4), "avg_pnl": round(avg, 4)}

    return {"long": _stats(longs), "short": _stats(shorts)}

"""Rich CLI dashboard — real-time portfolio display."""

from __future__ import annotations

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import BarColumn, Progress, TextColumn

from hfi.core.types import PortfolioState
from hfi.execution.order_manager import Position


console = Console()


def render_dashboard(
    portfolio: PortfolioState,
    positions: list[Position],
    current_prices: dict[str, float],
    milestones: list[float] | None = None,
) -> None:
    """Render full CLI dashboard."""
    console.clear()
    console.print(Panel("[bold cyan]HFI - Hedge Fund Inshallah[/bold cyan]", style="cyan"))

    # Portfolio summary
    _render_portfolio(portfolio)

    # Open positions
    if positions:
        _render_positions(positions, current_prices)

    # Milestone tracker
    if milestones:
        _render_milestones(portfolio.balance_usd, milestones)

    # Stats
    _render_stats(portfolio)


def _render_portfolio(portfolio: PortfolioState) -> None:
    """Render portfolio summary panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="bold")
    table.add_column("Value", justify="right")

    pnl_color = "green" if portfolio.daily_pnl >= 0 else "red"
    dd_color = "green" if portfolio.drawdown_pct < 0.05 else "yellow" if portfolio.drawdown_pct < 0.10 else "red"

    table.add_row("Balance", f"${portfolio.balance_usd:.2f}")
    table.add_row("Equity", f"${portfolio.equity_usd:.2f}")
    table.add_row("Unrealized PnL", f"${portfolio.unrealized_pnl:+.2f}")
    table.add_row("Daily PnL", f"[{pnl_color}]${portfolio.daily_pnl:+.2f} ({portfolio.daily_pnl_pct:+.2%})[/{pnl_color}]")
    table.add_row("Drawdown", f"[{dd_color}]{portfolio.drawdown_pct:.2%}[/{dd_color}]")
    table.add_row("Open Positions", str(portfolio.open_positions))
    table.add_row("Consecutive Losses", str(portfolio.consecutive_losses))

    console.print(Panel(table, title="Portfolio", border_style="blue"))


def _render_positions(positions: list[Position], prices: dict[str, float]) -> None:
    """Render open positions table."""
    table = Table(title="Open Positions", border_style="green")
    table.add_column("ID", style="dim")
    table.add_column("Symbol")
    table.add_column("Side")
    table.add_column("Entry")
    table.add_column("Current")
    table.add_column("PnL $")
    table.add_column("PnL %")
    table.add_column("SL")
    table.add_column("TP")
    table.add_column("Engine", style="dim")

    for pos in positions:
        price = prices.get(pos.symbol, pos.entry_price)
        pnl = pos.unrealized_pnl(price)
        pnl_pct = pos.unrealized_pnl_pct(price)
        color = "green" if pnl >= 0 else "red"

        table.add_row(
            pos.id[-8:],
            pos.symbol.split("/")[0],
            f"[{'green' if pos.side == 'long' else 'red'}]{pos.side.upper()}[/]",
            f"${pos.entry_price:.4f}",
            f"${price:.4f}",
            f"[{color}]${pnl:+.2f}[/{color}]",
            f"[{color}]{pnl_pct:+.2%}[/{color}]",
            f"${pos.stop_loss:.4f}",
            f"${pos.take_profit:.4f}",
            pos.engine,
        )

    console.print(table)


def _render_milestones(balance: float, milestones: list[float]) -> None:
    """Render milestone progress bar."""
    # Find current milestone target
    current_target = milestones[-1]
    for m in milestones:
        if balance < m:
            current_target = m
            break

    progress_pct = min(1.0, balance / current_target)

    milestone_text = " → ".join(
        f"[bold green]${m:.0f}[/]" if balance >= m else f"${m:.0f}"
        for m in milestones
    )

    console.print(Panel(
        f"Progress: ${balance:.2f} / ${current_target:.0f} ({progress_pct:.0%})\n"
        f"Milestones: {milestone_text}",
        title="Snowball Progress",
        border_style="yellow",
    ))


def _render_stats(portfolio: PortfolioState) -> None:
    """Render trading statistics."""
    if portfolio.total_trades == 0:
        console.print(Panel("No trades yet", title="Stats", border_style="dim"))
        return

    win_rate = portfolio.winning_trades / portfolio.total_trades if portfolio.total_trades > 0 else 0

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Trades", str(portfolio.total_trades))
    table.add_row("Win Rate", f"{win_rate:.1%}")
    table.add_row("Max Equity", f"${portfolio.max_equity:.2f}")

    console.print(Panel(table, title="Statistics", border_style="magenta"))

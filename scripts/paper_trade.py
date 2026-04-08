"""Start paper trading bot.

Usage:
    python scripts/paper_trade.py
    python scripts/paper_trade.py --balance 200
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hfi.execution.trader import run_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/paper_trade.log"),
    ],
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFI Paper Trading")
    parser.add_argument("--config", default="config", help="Config directory")
    parser.add_argument("--balance", type=float, default=100.0, help="Starting balance")
    args = parser.parse_args()

    print("=" * 50)
    print("  HFI - Hedge Fund Inshallah")
    print("  Mode: PAPER TRADING")
    print(f"  Starting balance: ${args.balance:.2f}")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    asyncio.run(run_bot(args.config, live=False))

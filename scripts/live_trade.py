"""Start LIVE trading bot.

DANGER: This uses REAL money. Make sure you:
1. Have tested with paper trading first
2. Have set up API keys in .env
3. Understand the risks

Usage:
    python scripts/live_trade.py --live
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
        logging.FileHandler("data/live_trade.log"),
    ],
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFI Live Trading")
    parser.add_argument("--config", default="config", help="Config directory")
    parser.add_argument("--live", action="store_true", help="REQUIRED: Enable live trading")
    args = parser.parse_args()

    if not args.live:
        print("ERROR: --live flag is REQUIRED for live trading.")
        print("If you meant to paper trade, use: python scripts/paper_trade.py")
        sys.exit(1)

    print("=" * 50)
    print("  HFI - Hedge Fund Inshallah")
    print("  Mode: *** LIVE TRADING ***")
    print("  WARNING: REAL MONEY AT RISK")
    print("=" * 50)

    confirm = input("\nType 'YES I UNDERSTAND' to continue: ")
    if confirm != "YES I UNDERSTAND":
        print("Aborted.")
        sys.exit(0)

    asyncio.run(run_bot(args.config, live=True))

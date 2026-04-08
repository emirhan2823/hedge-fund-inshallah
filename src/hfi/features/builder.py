"""Feature builder — compute indicators from OHLCV using pandas-ta.

Produces FeatureVector for each candle.
Skill ref: pandas-ta.md, feature-engineering.md
"""

from __future__ import annotations

import logging
from datetime import timezone

import numpy as np
import pandas as pd
import pandas_ta as ta

from hfi.core.types import FeatureVector

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, symbol: str) -> list[FeatureVector]:
    """Build feature vectors from OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with DatetimeIndex (timestamp) and columns: open, high, low, close, volume
        symbol: Trading pair symbol (e.g. "BTC/USDT:USDT")

    Returns:
        List of FeatureVector for each row with enough data.
    """
    if len(df) < 60:
        logger.warning("Not enough data for features: %d rows (need 60+)", len(df))
        return []

    # Compute all indicators at once
    feat = pd.DataFrame(index=df.index)

    # --- Volatility ---
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    feat["atr_14"] = atr
    feat["atr_14_pct"] = atr / df["close"]

    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None:
        feat["bb_width"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1]  # (upper-lower)/mid
        feat["bb_pct_b"] = (df["close"] - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2])
    else:
        feat["bb_width"] = 0.0
        feat["bb_pct_b"] = 0.5

    # --- Trend ---
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    feat["adx_14"] = adx.iloc[:, 0] if adx is not None else 0.0

    feat["ema_8"] = ta.ema(df["close"], length=8)
    feat["ema_21"] = ta.ema(df["close"], length=21)
    feat["ema_55"] = ta.ema(df["close"], length=55)

    # --- Momentum ---
    feat["rsi_14"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    feat["macd_hist"] = macd.iloc[:, 1] if macd is not None else 0.0  # histogram
    feat["roc_10"] = ta.roc(df["close"], length=10)

    # --- Volume ---
    vol_sma = ta.sma(df["volume"], length=20)
    feat["volume_sma_20"] = vol_sma
    feat["volume_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)

    # --- Derived ---
    sma_20 = ta.sma(df["close"], length=20)
    std_20 = df["close"].rolling(20).std()
    feat["zscore_close_20"] = (df["close"] - sma_20) / std_20.replace(0, np.nan)

    # Hurst exponent (simplified via rescaled range)
    feat["hurst_exponent"] = _rolling_hurst(df["close"], window=50)

    # ATR percentile (rolling 100-bar percentile rank)
    feat["atr_pctl"] = feat["atr_14"].rolling(100, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # Drop rows with NaN (warmup period)
    feat = feat.dropna()

    if feat.empty:
        logger.warning("All features are NaN after warmup for %s", symbol)
        return []

    # Convert to FeatureVector list
    vectors = []
    for ts, row in feat.iterrows():
        try:
            fv = FeatureVector(
                timestamp=ts.to_pydatetime().replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.to_pydatetime(),
                symbol=symbol,
                atr_14=float(row["atr_14"]),
                atr_14_pct=float(row["atr_14_pct"]),
                bb_width=float(row["bb_width"]),
                adx_14=float(row["adx_14"]),
                ema_8=float(row["ema_8"]),
                ema_21=float(row["ema_21"]),
                ema_55=float(row["ema_55"]),
                rsi_14=float(row["rsi_14"]),
                macd_hist=float(row["macd_hist"]),
                roc_10=float(row["roc_10"]),
                bb_pct_b=float(row["bb_pct_b"]),
                volume_ratio=float(row["volume_ratio"]),
                volume_sma_20=float(row["volume_sma_20"]),
                zscore_close_20=float(row["zscore_close_20"]),
                hurst_exponent=float(row["hurst_exponent"]),
                atr_pctl=float(row["atr_pctl"]) if not np.isnan(row["atr_pctl"]) else None,
            )
            vectors.append(fv)
        except Exception as e:
            logger.debug("Skipping row %s: %s", ts, e)
            continue

    logger.info("Built %d feature vectors for %s", len(vectors), symbol)
    return vectors


def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build features as DataFrame (for backtesting, no FeatureVector conversion).

    Returns DataFrame with all indicator columns added to original OHLCV.
    """
    if len(df) < 60:
        return df

    # Volatility
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_14_pct"] = df["atr_14"] / df["close"]

    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None:
        df["bb_upper"] = bb.iloc[:, 0]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_lower"] = bb.iloc[:, 2]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    else:
        df["bb_upper"] = df["bb_mid"] = df["bb_lower"] = df["close"]
        df["bb_width"] = 0.0
        df["bb_pct_b"] = 0.5

    # Trend
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is not None:
        df["adx_14"] = adx_df.iloc[:, 0]
    else:
        df["adx_14"] = 0.0

    df["ema_8"] = ta.ema(df["close"], length=8)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_55"] = ta.ema(df["close"], length=55)

    # Momentum
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd_df is not None:
        df["macd_line"] = macd_df.iloc[:, 0]
        df["macd_hist"] = macd_df.iloc[:, 1]
        df["macd_signal"] = macd_df.iloc[:, 2]
    else:
        df["macd_line"] = df["macd_hist"] = df["macd_signal"] = 0.0

    df["roc_10"] = ta.roc(df["close"], length=10)

    # Volume
    df["volume_sma_20"] = ta.sma(df["volume"], length=20)
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

    # Derived
    sma_20 = ta.sma(df["close"], length=20)
    std_20 = df["close"].rolling(20).std()
    df["zscore_close_20"] = (df["close"] - sma_20) / std_20.replace(0, np.nan)

    df["hurst_exponent"] = _rolling_hurst(df["close"], window=50)

    df["atr_pctl"] = df["atr_14"].rolling(100, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    return df


def _rolling_hurst(series: pd.Series, window: int = 50) -> pd.Series:
    """Simplified rolling Hurst exponent via rescaled range method."""
    def _hurst(x: np.ndarray) -> float:
        if len(x) < 20:
            return 0.5
        returns = np.diff(np.log(x[x > 0]))
        if len(returns) < 10:
            return 0.5

        n = len(returns)
        mean_r = np.mean(returns)
        cumdev = np.cumsum(returns - mean_r)
        r = np.max(cumdev) - np.min(cumdev)
        s = np.std(returns, ddof=1)
        if s == 0 or r == 0:
            return 0.5

        rs = r / s
        h = np.log(rs) / np.log(n)
        return float(np.clip(h, 0.0, 1.0))

    return series.rolling(window, min_periods=20).apply(_hurst, raw=True)

import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import tushare as ts


class DataLoaderError(Exception):
    """Custom exception for data loading and validation errors."""
    pass


def validate_inputs(
    ts_code: str,
    end_date_str: str,
    rolling_window: int,
    er_window: int,
) -> None:
    """
    Validate user inputs.
    Raises DataLoaderError if validation fails.
    """
    if not isinstance(ts_code, str) or not re.match(r"^\d{6}\.(sh|sz|bj)$", ts_code, re.IGNORECASE):
        raise DataLoaderError(
            "Invalid ticker format. Use 'XXXXXX.sh', 'XXXXXX.sz', or 'XXXXXX.bj' "
            "(e.g. '600549.sh')."
        )

    if not isinstance(end_date_str, str):
        raise DataLoaderError("end_date_str must be a string in YYYYMMDD format.")

    try:
        datetime.strptime(end_date_str, "%Y%m%d")
    except ValueError as exc:
        raise DataLoaderError(
            "Invalid date format. Please use 'YYYYMMDD' (e.g. '20260326')."
        ) from exc

    if not isinstance(rolling_window, int) or rolling_window <= 0:
        raise DataLoaderError("rolling_window must be a positive integer.")

    if not isinstance(er_window, int) or er_window <= 0:
        raise DataLoaderError("er_window must be a positive integer.")


def get_tushare_pro(token: str):
    """
    Initialize Tushare Pro client.
    """
    if not token or not isinstance(token, str):
        raise DataLoaderError("Missing or invalid Tushare token.")
    ts.set_token(token)
    return ts.pro_api()


def infer_limit_rule(company_name: str, market_board: str) -> tuple[float, str]:
    """
    Infer daily price limit percentage by board / ST status.
    """
    if market_board in ["创业板", "科创板"]:
        return 0.195, "20%"
    if market_board == "北交所":
        return 0.295, "30%"
    if "ST" in company_name.upper():
        return 0.0495, "5%"
    return 0.095, "10%"


def _fetch_stock_basic(pro, ts_code: str) -> tuple[str, str, float, str]:
    """
    Fetch stock metadata and infer limit rule.
    """
    basic_info = pro.stock_basic(ts_code=ts_code, fields="ts_code,name,market")

    if basic_info is None or basic_info.empty:
        company_name = "Unknown"
        market_board = "Unknown"
        limit_pct, limit_rule = 0.095, "10%"
        return company_name, market_board, limit_pct, limit_rule

    company_name = str(basic_info["name"].iloc[0])
    market_board = str(basic_info["market"].iloc[0])
    limit_pct, limit_rule = infer_limit_rule(company_name, market_board)
    return company_name, market_board, limit_pct, limit_rule


def _safe_history_start_date(end_date_str: str, rolling_window: int, er_window: int) -> str:
    """
    Compute a conservative history start date.
    We deliberately over-fetch calendar days to survive holidays / suspensions / NaNs.
    """
    end_dt = pd.to_datetime(end_date_str)
    lookback_days = max(int(rolling_window * 4), int(er_window * 8), 365)
    start_dt = end_dt - pd.Timedelta(days=lookback_days)
    return start_dt.strftime("%Y%m%d")


def _fetch_price_data(pro, ts_code: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    Fetch both HFQ-adjusted and raw daily bar data, then merge them.
    """
    hfq = ts.pro_bar(
        ts_code=ts_code,
        start_date=start_date_str,
        end_date=end_date_str,
        adj="hfq",
        api=pro,
    )
    raw = ts.pro_bar(
        ts_code=ts_code,
        start_date=start_date_str,
        end_date=end_date_str,
        adj=None,
        api=pro,
    )

    if hfq is None or hfq.empty:
        raise DataLoaderError(f"No HFQ data found for {ts_code} up to {end_date_str}.")
    if raw is None or raw.empty:
        raise DataLoaderError(f"No raw data found for {ts_code} up to {end_date_str}.")

    hfq = hfq.sort_values("trade_date").reset_index(drop=True)
    raw = raw.sort_values("trade_date").reset_index(drop=True)

    keep_hfq = [
        "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"
    ]
    keep_raw = [
        "trade_date", "open", "high", "low", "close", "pre_close"
    ]

    hfq = hfq[keep_hfq].rename(
        columns={
            "open": "open_hfq",
            "high": "high_hfq",
            "low": "low_hfq",
            "close": "close_hfq",
            "pre_close": "pre_close_hfq",
            "vol": "vol",
            "amount": "amount",
        }
    )

    raw = raw[keep_raw].rename(
        columns={
            "open": "open_raw",
            "high": "high_raw",
            "low": "low_raw",
            "close": "close_raw",
            "pre_close": "pre_close_raw",
        }
    )

    df = pd.merge(hfq, raw, on="trade_date", how="inner")
    if df.empty:
        raise DataLoaderError("Merged HFQ/raw dataset is empty after join on trade_date.")

    df["Date"] = pd.to_datetime(df["trade_date"])
    return df


def _engineer_features(
    df: pd.DataFrame,
    ts_code: str,
    company_name: str,
    market_board: str,
    limit_pct: float,
    limit_rule: str,
) -> pd.DataFrame:
    """
    Feature engineering for Kalman model input.
    """
    # Keep only actual trading days with volume
    df = df[df["vol"] > 0].copy()
    if df.empty:
        raise DataLoaderError("No valid trading rows left after filtering zero-volume rows.")

    # Unified naming
    df["Ticker"] = ts_code
    df["Company_Name"] = company_name
    df["Market_Board"] = market_board
    df["Limit_Rule"] = limit_rule
    df["limit_pct"] = limit_pct

    # Core prices
    df["Close"] = df["close_hfq"]         # model-facing adjusted close
    df["Close_raw"] = df["close_raw"]     # display-facing raw close
    df["Volume"] = df["vol"]

    # Log price for KF
    if (df["Close"] <= 0).any():
        raise DataLoaderError("Encountered non-positive HFQ close price, cannot take log.")
    df["log_close"] = np.log(df["Close"])

    # Keep NaNs for now; outer function decides final trimming
    return df


def fetch_kalman_data(
    ts_code: str,
    end_date_str: str,
    rolling_window: int,
    er_window: int,
    tushare_token: str,
    min_history_buffer: int = 30,
) -> pd.DataFrame:
    """
    Fetch and prepare data for the Kalman filter.

    Returns a dataframe with:
    - HFQ adjusted series for modeling
    - raw close for display / target-price mapping
    - metadata and engineered features

    Raises DataLoaderError on failure.
    """
    validate_inputs(ts_code, end_date_str, rolling_window, er_window)

    pro = get_tushare_pro(tushare_token)
    company_name, market_board, limit_pct, limit_rule = _fetch_stock_basic(pro, ts_code)

    start_date_str = _safe_history_start_date(
        end_date_str=end_date_str,
        rolling_window=rolling_window,
        er_window=er_window,
    )

    df = _fetch_price_data(
        pro=pro,
        ts_code=ts_code,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
    )

    df = _engineer_features(
        df=df,
        ts_code=ts_code,
        company_name=company_name,
        market_board=market_board,
        limit_pct=limit_pct,
        limit_rule=limit_rule,
    )

    # Drop only rows that are unusable for the model
    required_cols = ["Date", "Close", "Close_raw", "Volume", "log_close", "limit_pct"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Need enough rows after feature burn-in
    min_required = rolling_window + min_history_buffer
    if len(df) < min_required:
        raise DataLoaderError(
            f"Not enough valid rows after preprocessing. Need at least {rolling_window}, got {len(df)}."
        )

    # Final tail selection for model input
    df = df.tail(rolling_window).reset_index(drop=True)

    # Final safety check
    if len(df) != rolling_window:
        raise DataLoaderError(
            f"Final dataset length mismatch. Expected {rolling_window}, got {len(df)}."
        )

    return df
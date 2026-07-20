import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from utils.data_loader import DataLoaderError, fetch_kalman_data, validate_inputs


def make_bars(rows: int = 8, start: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=rows)
    close = start + np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "ts_code": ["TEST.SH"] * rows,
            "trade_date": dates.strftime("%Y%m%d"),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "pre_close": close - 1.0,
            "vol": np.full(rows, 1_000.0),
            "amount": np.full(rows, 10_000.0),
        }
    )


class DataLoaderIndexSupportTests(unittest.TestCase):
    def test_validate_inputs_accepts_non_stock_index_market_suffix(self):
        validate_inputs("000300.CSI", "20260131", 5, 3)

    @patch("utils.data_loader.ts.pro_bar")
    @patch("utils.data_loader.get_tushare_pro")
    def test_stock_path_remains_unchanged(self, get_pro, pro_bar):
        pro = MagicMock()
        pro.stock_basic.return_value = pd.DataFrame(
            [{"ts_code": "600519.SH", "name": "贵州茅台", "market": "主板"}]
        )
        get_pro.return_value = pro

        hfq = make_bars(start=200.0)
        raw = make_bars(start=100.0)
        pro_bar.side_effect = [hfq, raw]

        result = fetch_kalman_data(
            "600519.sh", "20260131", 5, 3, "token", min_history_buffer=0
        )

        self.assertEqual(pro_bar.call_count, 2)
        pro.index_basic.assert_not_called()
        pro.index_daily.assert_not_called()
        self.assertEqual(result["Ticker"].iloc[-1], "600519.SH")
        self.assertEqual(result["Company_Name"].iloc[-1], "贵州茅台")
        self.assertTrue((result["Close"] != result["Close_raw"]).all())

    @patch("utils.data_loader.ts.pro_bar")
    @patch("utils.data_loader.get_tushare_pro")
    def test_index_daily_is_normalized_to_stock_schema(self, get_pro, pro_bar):
        pro = MagicMock()
        pro.stock_basic.return_value = pd.DataFrame()
        pro.index_basic.return_value = pd.DataFrame(
            [{"ts_code": "000300.CSI", "name": "沪深300", "market": "CSI"}]
        )
        pro.index_daily.return_value = make_bars()
        get_pro.return_value = pro

        result = fetch_kalman_data(
            "000300.csi", "20260131", 5, 3, "token", min_history_buffer=0
        )

        pro_bar.assert_not_called()
        pro.index_basic.assert_called_once_with(
            ts_code="000300.CSI",
            market="",
            fields="ts_code,name,market",
        )
        pro.index_daily.assert_called_once()
        self.assertEqual(result["Ticker"].iloc[-1], "000300.CSI")
        self.assertEqual(result["Company_Name"].iloc[-1], "沪深300")
        self.assertTrue(result["Close"].equals(result["Close_raw"]))
        self.assertTrue(result["open_hfq"].equals(result["open_raw"]))
        self.assertTrue((result["limit_pct"] == 0.095).all())
        self.assertTrue((result["Limit_Rule"] == "10%").all())

    @patch("utils.data_loader.get_tushare_pro")
    def test_unknown_code_raises_clear_error(self, get_pro):
        pro = MagicMock()
        pro.stock_basic.return_value = pd.DataFrame()
        pro.index_basic.return_value = pd.DataFrame()
        get_pro.return_value = pro

        with self.assertRaisesRegex(DataLoaderError, "No stock or index metadata"):
            fetch_kalman_data(
                "UNKNOWN.TEST", "20260131", 5, 3, "token", min_history_buffer=0
            )

    @patch("utils.data_loader.get_tushare_pro")
    def test_empty_index_daily_response_raises_clear_error(self, get_pro):
        pro = MagicMock()
        pro.stock_basic.return_value = pd.DataFrame()
        pro.index_basic.return_value = pd.DataFrame(
            [{"ts_code": "000001.SH", "name": "上证指数", "market": "SSE"}]
        )
        pro.index_daily.return_value = pd.DataFrame()
        get_pro.return_value = pro

        with self.assertRaisesRegex(DataLoaderError, "No index daily data"):
            fetch_kalman_data(
                "000001.SH", "20260131", 5, 3, "token", min_history_buffer=0
            )


if __name__ == "__main__":
    unittest.main()

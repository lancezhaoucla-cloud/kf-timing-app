import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import matplotlib.font_manager as fm
import time

from config import KALMAN_PARAMS, DEFAULT_TICKER, DEFAULT_END_DATE
from utils.data_loader import fetch_kalman_data, DataLoaderError
from utils.kalman_model import optimize_kalman_parameters, run_kalman_filter


# ==========================================
# Configure Matplotlib for Chinese Support
# ==========================================
def setup_chinese_font():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, "assets", "fonts", "NotoSansSC-Regular.ttf")

    st.write("Font path:", font_path)
    st.write("Font exists:", os.path.exists(font_path))

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()

        st.write("Loaded font name:", font_name)

        mpl.rcParams["font.family"] = [font_name]
        mpl.rcParams["font.sans-serif"] = [font_name]
        mpl.rcParams["axes.unicode_minus"] = False

        # 验证 matplotlib 实际找到的字体
        st.write("Matplotlib resolved font:", fm.findfont(font_name, fallback_to_default=False))
    else:
        st.warning("Custom font file not found. Falling back to default font.")
        mpl.rcParams["font.family"] = ["DejaVu Sans"]
        mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False

setup_chinese_font()


# ==========================================
# Streamlit Page Setup
# ==========================================
st.set_page_config(page_title="右侧交易助手", layout="wide")

if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = 0.0

st.title("基于卡尔曼滤波的A股右侧交易助手")

st.markdown("""
> **模型简介**：本系统基于状态空间模型（State-Space Model），利用**卡尔曼滤波器**将A股价格动态分解为“底层趋势”与“短波周期”两个隐状态。
>
> 底层数学引擎采用MLE（最大似然估计）进行超参数的全局寻优，并深度结合A股微观结构，引入了基于市场环境的动态调节观测噪声及过程噪声，以及针对涨跌停板未释放动能的价格修正机制。
>
> 模型旨在滤除市场杂音，动态判别市场状态（单边趋势 vs 趋势周期共振），为您提供具备严谨统计学支撑的右侧交易信号。
""")

st.markdown("---")


# ==========================================
# Helper
# ==========================================
def calculate_perf_stats(daily_ret, trading_days_per_year=250):
    r = daily_ret.dropna()
    if len(r) == 0:
        return None

    n = len(r)
    total_ret = (1 + r).prod() - 1
    years = n / trading_days_per_year
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(trading_days_per_year)
    sharpe = (r.mean() / r.std() * np.sqrt(trading_days_per_year)) if r.std() > 1e-12 else np.nan

    cum = (1 + r).cumprod()
    peak = cum.cummax()
    max_dd = (cum / peak - 1).min()

    return {
        "总回报": f"{total_ret * 100:.2f}%",
        "年化回报": f"{cagr * 100:.2f}%",
        "年化波动率": f"{ann_vol * 100:.2f}%",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{max_dd * 100:.2f}%",
    }


# ==========================================
# 1. Sidebar Inputs & Hyperparameters
# ==========================================
st.sidebar.header("模型设置")

# 用局部 params，避免污染全局配置
params = KALMAN_PARAMS.copy()

with st.sidebar.form(key="model_config_form"):
    ticker = st.text_input(
        "股票代码",
        value=DEFAULT_TICKER,
        help="股票代码 (例: 600549.sh)，不支持指数、ETF、基金等其它证券类型。"
    )

    end_date = st.text_input(
        "截止日期",
        value=DEFAULT_END_DATE,
        help="取离截止日期最近一个交易日日期，请填写 YYYYMMDD 格式。"
    )

    st.markdown("---")
    st.subheader("卡尔曼滤波器设置")

    rolling_window_options = {"短": 60, "中": 120, "长": 250}
    default_rw_label = "长"
    selected_rolling_window = st.selectbox(
        "滚动窗口",
        options=list(rolling_window_options.keys()),
        index=list(rolling_window_options.keys()).index(default_rw_label),
        help="总交易天数，用于卡尔曼滤波器拟合。短：60天，中：120天，长：250天。建议使用长窗口。"
    )
    params["rolling_window"] = rolling_window_options[selected_rolling_window]

    with st.expander("进阶设置", expanded=False):
        trend_threshold_options = {"低": 0.0, "中": 0.001, "高": 0.005}
        selected_trend_threshold = st.selectbox(
            "趋势阈值",
            options=list(trend_threshold_options.keys()),
            index=0,
            help="趋势对数收益斜率阈值，入场交易须大于此阈值。低中高对应数值为0，0.001，0.005。"
        )
        params["trend_threshold"] = trend_threshold_options[selected_trend_threshold]

        cycle_slope_threshold_options = {"低": 0.0, "中": 0.001, "高": 0.005}
        selected_cycle_slope_threshold = st.selectbox(
            "周期斜率阈值",
            options=list(cycle_slope_threshold_options.keys()),
            index=0,
            help="周期对数收益斜率阈值，入场交易须大于此阈值。低中高对应数值为0，0.001，0.005。"
        )
        params["cycle_slope_threshold"] = cycle_slope_threshold_options[selected_cycle_slope_threshold]

        cycle_z_threshold_options = {"低": 2.5, "中": 3.0, "高": 3.5}
        selected_cycle_z_threshold = st.selectbox(
            "周期顶点阈值",
            options=list(cycle_z_threshold_options.keys()),
            index=1,
            help="周期顶点卖出阈值。"
        )
        params["cycle_z_threshold"] = cycle_z_threshold_options[selected_cycle_z_threshold]

        cycle_snr_threshold_options = {"低": 0.2, "中": 0.5, "高": 0.8}
        selected_cycle_snr_threshold = st.selectbox(
            "周期信噪比阈值",
            options=list(cycle_snr_threshold_options.keys()),
            index=1,
            help="与趋势振幅相比较，周期振幅须大于此阈值才被视为有效。低中高对应数值为0.2，0.5，0.8。"
        )
        params["cycle_snr_threshold"] = cycle_snr_threshold_options[selected_cycle_snr_threshold]

        vol_options = {"低": 1.0, "中": 5.0, "高": 10.0}
        selected_vol = st.selectbox(
            "交易量敏感度",
            options=list(vol_options.keys()),
            index=1,
            help="数值越高模型越信任放量交易后的价格。"
        )
        params["vol_scale"] = vol_options[selected_vol]

        er_scale_options = {"低": 1.0, "中": 5.0, "高": 10.0}
        selected_er_scale = st.selectbox(
            "短期趋势敏感度",
            options=list(er_scale_options.keys()),
            index=1,
            help="数值越高模型越信任近一个月价格走势。"
        )
        params["er_scale"] = er_scale_options[selected_er_scale]

        params["max_R_t_threshold"] = st.number_input(
            "交易量敏感度上限",
            value=float(params["max_R_t_threshold"]),
            format="%.2f",
            help="调整模型对交易量冲击的最高容忍度。推荐数值为1，5，10。"
        )
        params["Q_scale_cap"] = st.number_input(
            "趋势信任度上限",
            value=float(params["Q_scale_cap"]),
            format="%.2f",
            help="调整模型对短期趋势信任度的上限。推荐数值为1，3，5。"
        )
        params["slippage"] = st.number_input(
            "交易滑点",
            value=float(params["slippage"]),
            format="%.4f",
            help="回测模拟交易滑点，默认为 0.001，即 0.1%。"
        )


    run_model = st.form_submit_button("运行模型", type="primary")


# ==========================================
# 2. Main Execution Block
# ==========================================
if run_model:
    current_time = time.time()
    time_since_last_run = current_time - st.session_state.last_run_time

    if time_since_last_run < 10.0:
        remaining_time = 10.0 - time_since_last_run
        st.warning(f"请等待 {remaining_time:.1f} 秒后再运行模型，以防止 API 滥用。")
        st.stop()

    st.session_state.last_run_time = current_time

    # --- Data Loading ---
    try:
        with st.spinner(f"正在获取 {ticker} 的数据..."):
            df = fetch_kalman_data(
                ts_code=ticker,
                end_date_str=end_date,
                rolling_window=params["rolling_window"],
                er_window=params["ER_window"],
                tushare_token=st.secrets["TUSHARE_TOKEN"],
            )
    except DataLoaderError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"数据加载失败：{e}")
        st.stop()

    if df.empty:
        st.error("数据为空，无法运行模型。")
        st.stop()

    # --- Modeling ---
    progress_bar = st.progress(0, text="准备运行 Kalman Filter...")

    def progress_callback(value, text=None):
        progress_bar.progress(value, text=text or "处理中...")

    with st.spinner("运行MLE参数优化..."):
        optimized_params = optimize_kalman_parameters(df, params)

    df = run_kalman_filter(
        df,
        optimized_params,
        params,
        progress_callback=progress_callback,
    )

    progress_bar.empty()
    st.success("MLE参数优化完成！")

    if df.empty or len(df) == 0:
        st.error("模型运行后结果为空。")
        st.stop()

    # ==========================================
    # 3. Latest Trading Signal & Target Price
    # ==========================================
    latest_row = df.iloc[-1]
    latest_date_dt = latest_row["Date"]
    latest_date_str = latest_date_dt.strftime("%Y%m%d")
    company_name = latest_row.get("Company_Name", "未知")
    data_ticker = latest_row.get("Ticker", ticker)

    st.subheader(
    f"最新交易信号: {company_name} ({data_ticker}) | 上一交易日：{latest_date_str}"
)

    real_close = latest_row.get("Close_raw", latest_row.get("Close", np.nan))
    if pd.isna(real_close):
        st.error("缺少展示价格字段 Close_raw / Close。")
        st.stop()

    latest_slope = latest_row["kf_slope"]
    is_cycle_alive = bool(latest_row.get("is_cycle_alive", False))
    current_regime = "趋势 + 周期回调" if is_cycle_alive else "纯趋势跟踪"
    action = "买入" if latest_row.get("signal_long", 0) == 1 else "持有现金/卖出"

    if is_cycle_alive:
        expected_log_ret = latest_row["pred_log_close_next"] - latest_row["log_close"]
    else:
        expected_log_ret = latest_slope

    expected_ret = np.exp(expected_log_ret) - 1
    target_price = real_close * (1 + expected_ret)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("交易信号", action)
    col2.metric("目标价格", f"¥{target_price:.2f}")
    col3.metric("预期回报", f"{expected_ret * 100:.2f}%")
    col4.metric("当前市场状态", current_regime)

    st.markdown("---")

    # ==========================================
    # 4. Filter Visualizations
    # ==========================================
    st.subheader("模型可视化")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["Date"], df["log_close"], label="对数收盘价", color="black", linestyle="--", alpha=0.7)
    ax1.plot(df["Date"], df["kf_level"], label="趋势", linewidth=2.5)
    ax1.plot(df["Date"], df["trend_plus_cycle"], label="趋势 + 周期拟合", linewidth=2, alpha=0.7)
    ax1.set_title("趋势 + 周期卡尔曼滤波")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df["Date"], df["kf_cycle"], label="周期")
    ax2.axhline(0, color="black", linestyle="--")
    ax2.set_title(f"周期卡尔曼滤波 (周期天数={optimized_params['opt_cycle_days']:.2f})")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
    plt.close(fig2)

    z_window = params["cycle_z_window"]
    if len(df) <= z_window:
        st.warning("样本不足，无法绘制策略净值图。")
    else:
        df_plot = df.iloc[z_window:].copy()

        if len(df_plot) > 0 and df_plot["cum_stock"].iloc[0] != 0 and df_plot["cum_strategy_after_cost"].iloc[0] != 0:
            rebased_cum_stock = df_plot["cum_stock"] / df_plot["cum_stock"].iloc[0]
            rebased_cum_strat = df_plot["cum_strategy_after_cost"] / df_plot["cum_strategy_after_cost"].iloc[0]
            rebased_excess = (rebased_cum_strat / rebased_cum_stock) - 1

            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(df_plot["Date"], rebased_cum_stock, label="买入并持有", color="C0")
            ax3.plot(df_plot["Date"], rebased_cum_strat, label="策略（非真实交易模拟）", color="C2")
            ax3.set_ylabel("单位净值 (初始为1)")
            ax3.set_title("策略 vs 买入并持有")
            ax3.grid(True)

            ax4 = ax3.twinx()
            ax4.plot(df_plot["Date"], rebased_excess * 100, label="超额收益(%)", linestyle="--", alpha=0.85, color="C1")
            ax4.axhline(0, color="gray", linestyle=":", linewidth=1)
            ax4.set_ylabel("超额收益(%)")

            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax4.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            st.pyplot(fig3)
            plt.close(fig3)

    st.markdown("---")

    # ==========================================
    # 5. Evaluation Metrics
    # ==========================================
    st.subheader("回测表现（非真实交易模拟）")

    z_window = params["cycle_z_window"]
    _ret_cols = ["stock_ret_today", "strategy_ret_today_after_cost"]

    if len(df) > z_window:
        aligned = df.iloc[z_window:].dropna(subset=_ret_cols)

        bh_stats = calculate_perf_stats(aligned["stock_ret_today"])
        strat_stats = calculate_perf_stats(aligned["strategy_ret_today_after_cost"])

        if bh_stats and strat_stats:
            metrics_df = pd.DataFrame({
                "买入并持有": bh_stats,
                "策略（非真实交易模拟）": strat_stats
            })
            st.table(metrics_df)

        valid = df.iloc[z_window:].dropna(subset=["pred_ret_1d", "real_ret_1d"]).copy()
        hit_ratio = ((valid["pred_ret_1d"] > 0) == (valid["real_ret_1d"] > 0)).mean() if len(valid) > 0 else np.nan
        avg_turnover = df["turnover"].mean() if "turnover" in df.columns else np.nan

        col_a, col_b = st.columns(2)
        col_a.metric("胜率", f"{hit_ratio * 100:.2f}%" if pd.notna(hit_ratio) else "N/A")
        col_b.metric("平均日换手率", f"{avg_turnover * 100:.2f}%" if pd.notna(avg_turnover) else "N/A")
    else:
        st.warning("样本不足，无法展示评估指标。")
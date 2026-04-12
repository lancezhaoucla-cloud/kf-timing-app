import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import matplotlib.font_manager as fm
import time

from config import KALMAN_PARAMS, DEFAULT_TICKER, DEFAULT_END_DATE, build_profile_config
from utils.kalman_model import prepare_model_features, optimize_kalman_parameters, run_kalman_filter
from utils.profile_selector import detect_profile_from_df
from utils.data_loader import fetch_kalman_data, DataLoaderError

# ==========================================
# Configure Matplotlib for Chinese Support
# ==========================================
def setup_chinese_font():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, "assets", "fonts", "NotoSansSC-Regular.ttf")

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()


        mpl.rcParams["font.family"] = [font_name]
        mpl.rcParams["font.sans-serif"] = [font_name]
        mpl.rcParams["axes.unicode_minus"] = False

    else:
        st.warning("Custom font file not found. Falling back to default font.")
        mpl.rcParams["font.family"] = ["DejaVu Sans"]
        mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False

setup_chinese_font()


# ==========================================
# Streamlit Page Setup
# ==========================================
st.set_page_config(page_title="趋势交易助手", layout="wide")

if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = 0.0

st.title("基于卡尔曼滤波的A股趋势交易助手")

st.markdown("""
> **模型简介**：本系统基于状态空间模型（State-Space Model），利用**卡尔曼滤波器**将A股价格动态分解为“底层趋势”与“短波周期”两个隐状态。
>
> 底层数学引擎采用MLE（最大似然估计）进行超参数的全局寻优，并深度结合A股微观结构，引入了基于市场环境的动态调节观测噪声及过程噪声，以及针对涨跌停板未释放动能的价格修正机制。
> 
> 模型通过多维特征识别市场结构，自动匹配不同交易风格（如趋势跟随、突破追强、防守稳健等），减少调参负担。
>
> 模型旨在滤除市场杂音，动态判别市场状态（单边趋势 vs 趋势周期共振），为您提供具备严谨统计学支撑的趋势交易信号。
>
> 推荐对多支股票同时对比使用，辅助择时判断，把握交易机会。
""")

st.markdown("---")

# ==========================================
# 长期回测验证（可折叠 + Tabs）
# ==========================================
with st.expander("📊 长期回测案例", expanded=False):

    st.markdown("""
    该部分展示模型在历史数据上的**样本外表现**，
    每次交易使用推荐交易风格，用于验证策略在不同市场环境下的稳定性与风险控制能力。使用收盘价到收盘价收益率，不考虑交易费率及滑点。
    """)

    tabs = st.tabs(["宁德时代（300750.SZ）","贵州茅台（600519.SH）"])

    # ==============================
    # Tab 1：宁德时代
    # ==============================
    with tabs[0]:

        st.subheader("策略 vs 买入并持有（长期表现）")

        st.image("assets/images/300750.png", width="stretch")

        st.markdown("### 📈 样本外统计分析",help="收盘价到收盘价收益率，不考虑交易费率及滑点。",)

        metrics_df = pd.DataFrame({
            "指标": [
                "年化收益率", 
                "年化波动率",
                "夏普比率",
                "最大回撤",
                "胜率（日度）",
                "持仓时间占比"
            ],
            "买入并持有": [
                "39.16%",
                "46.02%",
                "0.95",
                "-62.88%",
                "47.72%",
                "100.00%"
            ],
            "策略": [
                "26.33%",
                "26.65%",
                "1.01",
                "-23.47%",
                "48.30%",
                "29.15%"
            ]
        })

        st.dataframe(metrics_df, width="stretch")

        st.info("""
        💡 **解读：**
        - 策略显著降低了波动率与最大回撤，风险控制能力明显优于买入并持有；
        - 在牺牲部分收益的前提下，实现了更高的风险调整收益（Sharpe Ratio）；
        - 持仓时间仅约30%，体现出策略具备明显的择时特征；
        - 在强趋势牛股（如宁德时代）中，策略可能跑输长期持有，但在震荡或下行环境中更具优势。
        """)
    with tabs[1]:

        st.subheader("策略 vs 买入并持有（长期表现）")

        st.image("assets/images/600519.png", width="stretch")

        st.markdown("### 📈 样本外统计分析",help="收盘价到收盘价收益率，不考虑交易费率及滑点。",)

        metrics_df = pd.DataFrame({
            "指标": [
                "年化收益率",
                "年化波动率",
                "夏普比率",
                "最大回撤",
                "胜率（日度）",
                "持仓时间占比"
            ],
            "买入并持有": [
                "7.74%",
                "27.63%",
                "0.41",
                "-47.48%",
                "47.66%",
                "100.00%"
            ],
            "策略":[
                "10.34%",
                "15.79%",
                "0.70",
                "-16.61%",
                "48.49%",
                "30.25%"
            ]
        })

        st.dataframe(metrics_df, width="stretch")

        st.markdown("### 🧾 最终表现总结")

        st.info("""
        💡 **解读：**
        - 相较于买入并持有，策略在贵州茅台上实现了更高的年化收益，同时显著降低了波动率与最大回撤；
        - 夏普比率由 0.41 提升至 0.70，说明策略不仅提高了收益，也明显改善了风险调整后的表现；
        - 持仓时间约为30%，体现出模型并非长期满仓，而是更倾向于在趋势占优阶段参与市场，在震荡或下行阶段主动规避风险；
        - 这说明对于长期慢牛、但中间伴随较大波动的核心资产，趋势交易模型有机会在控制回撤的同时取得优于买入并持有的结果。
        """)

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

PROFILE_LABELS = {
    "Trend_follower": "趋势跟随型",
    "Breakseeker": "突破追强型",
    "Defender": "防守稳健型",
    "Activist": "主动交易型",
    "All_other": "中性默认型",
    "Custom": "自定义风格",
}

st.sidebar.header("模型设置")

# 用局部 params，避免污染全局配置
params = KALMAN_PARAMS.copy()

# -----------------------------
# 放在 form 外：立即响应的模式选择
# -----------------------------
style_mode = st.sidebar.radio(
    "交易风格模式",
    ["推荐交易风格", "指定交易风格", "自定义交易风格"],
    index=0,
    help="推荐交易风格：模型自动识别；指定交易风格：手动选择预设人格；自定义交易风格：手动调整超参数。"
)

st.sidebar.markdown("---")

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
    st.subheader("基础设置")

    rolling_window_options = {"60日": 60, "120日": 120, "250日": 250}
    default_rw_label = "120日"
    selected_rolling_window = st.selectbox(
        "滚动窗口",
        options=list(rolling_window_options.keys()),
        index=list(rolling_window_options.keys()).index(default_rw_label),
        help="总交易天数，用于卡尔曼滤波器拟合。短：60天，中：120天，长：250天。"
    )
    params["rolling_window"] = rolling_window_options[selected_rolling_window]

    selected_profile_mode = None
    manual_profile = None

    # -----------------------------
    # 模式 1：推荐交易风格
    # -----------------------------
    if style_mode == "推荐交易风格":
        st.info("系统将根据当前市场结构自动识别交易风格。")
        classifier_window_options = {"20日": 20, "60日": 60, "120日": 120}
        default_classifier_window_label = "20日"
        selected_classifier_window = st.selectbox(
            "市场环境判断周期",
            options=list(classifier_window_options.keys()),
            index=list(classifier_window_options.keys()).index(default_classifier_window_label),
            help="提取周期长度内的市场环境特征，用于判断市场环境。"
        )
        params["classifier_window"] = classifier_window_options[selected_classifier_window]

    # -----------------------------
    # 模式 2：指定交易风格
    # -----------------------------
    elif style_mode == "指定交易风格":
        profile_name_map = {
            "Trend_follower": "趋势跟随型",
            "Breakseeker": "突破追强型",
            "Defender": "防守稳健型",
            "Activist": "主动交易型",
            "All_other": "中性默认型",
        }

        manual_profile = st.selectbox(
            "选择交易风格",
            options=list(PROFILE_LABELS.keys())[:-1],  # 不含 Custom
    index=list(PROFILE_LABELS.keys()).index("All_other"),
    format_func=lambda x: PROFILE_LABELS.get(x, x),
)
        st.caption(f"当前手动指定风格：{profile_name_map[manual_profile]}")

    # -----------------------------
    # 模式 3：自定义交易风格
    # -----------------------------
    elif style_mode == "自定义交易风格":
        st.subheader("自定义风格设置")

        jump_penalty_options = {"极低": 0.0,"低": 0.5, "中": 2.0, "高": 4.0}
        default_jump_penalty_label = "中"
        selected_jump_penalty = st.selectbox(
            "大幅波动惩罚",
            options=list(jump_penalty_options.keys()),
            index=list(jump_penalty_options.keys()).index(default_jump_penalty_label),
            help="大幅波动惩罚，数值越高模型越不信任股价大幅波动。"
        )
        params["jump_alpha"] = jump_penalty_options[selected_jump_penalty]

        with st.expander("进阶设置", expanded=True):
            trend_threshold_options = {"极低": -0.005,"低": 0.0, "中": 0.001, "高": 0.005}
            selected_trend_threshold = st.selectbox(
                "趋势阈值",
                options=list(trend_threshold_options.keys()),
                index=1,
                help="趋势对数收益斜率阈值，入场交易须大于此阈值。"
            )
            params["trend_threshold"] = trend_threshold_options[selected_trend_threshold]

            cycle_slope_threshold_options = {"极低": -0.005,"低": 0.0, "中": 0.001, "高": 0.005}
            selected_cycle_slope_threshold = st.selectbox(
                "周期斜率阈值",
                options=list(cycle_slope_threshold_options.keys()),
                index=1,
                help="周期对数收益斜率阈值，入场交易须大于此阈值。"
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

            cycle_snr_threshold_options = {"低": 0.2, "中": 0.5, "高": 0.8, "极高": 1.0,}
            selected_cycle_snr_threshold = st.selectbox(
                "周期强度阈值",
                options=list(cycle_snr_threshold_options.keys()),
                index=1,
                help="与趋势振幅相比较，周期振幅须大于此阈值才被视为有效。"
            )
            params["cycle_snr_threshold"] = cycle_snr_threshold_options[selected_cycle_snr_threshold]

            vol_window_options = {"20日": 20, "60日": 60, "120日": 120}
            default_vol_window_label = "20日"
            selected_vol_window = st.selectbox(
                "交易量回测窗口",
                options=list(vol_window_options.keys()),
                index=list(vol_window_options.keys()).index(default_vol_window_label),
                help="交易量回测窗口，用于计算交易量敏感度。"
            )
            params["vol_window"] = vol_window_options[selected_vol_window]

            vol_options = {"低": 1.0, "中": 5.0, "高": 10.0, "极高": 20.0,}
            selected_vol = st.selectbox(
                "交易量敏感度",
                options=list(vol_options.keys()),
                index=1,
                help="数值越高模型越信任放量交易后的价格（影响观测噪声）。"
            )
            params["vol_scale"] = vol_options[selected_vol]

            er_window_options = {"20日": 20, "60日": 60, "120日": 120}
            default_er_window_label = "20日"
            selected_er_window = st.selectbox(
                "短期趋势窗口",
                options=list(er_window_options.keys()),
                index=list(er_window_options.keys()).index(default_er_window_label),
                help="短期趋势窗口，用于计算短期趋势敏感度。"
            )
            params["ER_window"] = er_window_options[selected_er_window]

            er_scale_options = {"低": 1.0, "中": 5.0, "高": 10.0, "极高": 20.0,}
            selected_er_scale = st.selectbox(
                "短期趋势敏感度",
                options=list(er_scale_options.keys()),
                index=1,
                help="数值越高模型越信任近一个月价格走势（影响过程噪声）。"
            )
            params["er_scale"] = er_scale_options[selected_er_scale]

            max_R_t_threshold_options = {"低": 3.0, "中": 5.0, "高": 10.0}
            selected_max_R_t_threshold = st.selectbox(
                "交易量敏感度上限",
                options=list(max_R_t_threshold_options.keys()),
                index=1,
                help="调整模型对交易量冲击的最高容忍度（影响观测噪声）。"
            )
            params["max_R_t_threshold"] = max_R_t_threshold_options[selected_max_R_t_threshold]

            Q_scale_cap_options = {"低": 3.0, "中": 5.0, "高": 10.0}
            selected_Q_scale_cap = st.selectbox(
                "趋势信任度上限",
                options=list(Q_scale_cap_options.keys()),
                index=1,
                help="调整模型对短期趋势信任度的上限（影响过程噪声）。"
            )
            params["Q_scale_cap"] = Q_scale_cap_options[selected_Q_scale_cap]

            params["slippage"] = st.number_input(
                "交易滑点",
                value=float(params["slippage"]),
                format="%.4f",
                help="回测模拟交易滑点，默认为 0.000。"
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
        st.warning(f"请等待 {remaining_time:.1f} 秒后再运行模型。")
        st.stop()

    st.session_state.last_run_time = current_time

    # ------------------------------------------
    # Step 1. Data Loading
    # ------------------------------------------
    try:
        with st.spinner(f"正在获取 {ticker} 的数据..."):
            # 这里先用基础 params 拉数据，不在 fetch 阶段绑定具体 profile
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

    # ------------------------------------------
    # Step 2. Decide active profile / config
    # ------------------------------------------
    base_conf = params.copy()

    if style_mode == "推荐交易风格":
        try:
            selected_profile, profile_feat = detect_profile_from_df(df, base_conf)
        except Exception as e:
            st.error(f"自动识别交易风格失败：{e}")
            st.stop()

        active_conf = build_profile_config(base_conf, selected_profile)
        active_conf["profile_name"] = selected_profile

    elif style_mode == "指定交易风格":
        selected_profile = manual_profile
        profile_feat = {}
        active_conf = build_profile_config(base_conf, selected_profile)
        active_conf["profile_name"] = selected_profile

    else:  # 自定义交易风格
        selected_profile = "Custom"
        profile_feat = {}
        active_conf = base_conf.copy()
        active_conf["profile_name"] = selected_profile

    # ------------------------------------------
    # Step 3. Recompute profile-dependent model features
    # ------------------------------------------
    try:
        df_model = prepare_model_features(df, active_conf)
    except Exception as e:
        st.error(f"模型特征准备失败：{e}")
        st.stop()

    if df_model.empty:
        st.error("模型特征准备后数据为空。")
        st.stop()

    # ------------------------------------------
    # Step 4. Modeling
    # ------------------------------------------
    progress_bar = st.progress(0, text="准备运行 Kalman Filter...")

    def progress_callback(value, text=None):
        progress_bar.progress(value, text=text or "处理中...")

    try:
        with st.spinner("运行MLE参数优化..."):
            optimized_params = optimize_kalman_parameters(df_model, active_conf)

        df_result = run_kalman_filter(
            df_model,
            optimized_params,
            active_conf,
            progress_callback=progress_callback,
        )
    except Exception as e:
        progress_bar.empty()
        st.error(f"模型运行失败：{e}")
        st.stop()

    progress_bar.empty()
    st.success("MLE参数优化完成！")

    if df_result.empty or len(df_result) == 0:
        st.error("模型运行后结果为空。")
        st.stop()

    # ------------------------------------------
    # Step 5. Display current profile
    # ------------------------------------------
    st.subheader("当前交易风格")

    selected_profile_cn = PROFILE_LABELS.get(selected_profile, selected_profile)

    if style_mode == "推荐交易风格":
        st.success(f"模型推荐风格：{selected_profile_cn}")
    elif style_mode == "指定交易风格":
        st.info(f"用户指定风格：{selected_profile_cn}")
    else:
        st.warning("当前为自定义交易风格")

    if profile_feat:
        with st.expander("诊断信息", expanded=False):
            profile_debug_df = pd.DataFrame(
            [{"指标": k, "值": v} for k, v in profile_feat.items()]
            )
            st.dataframe(profile_debug_df, width="stretch")

    # 后续所有展示，请统一使用 df_result
    df = df_result

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

    # -----------------------------
    # Price fields
    # Close      : HFQ price used by model
    # Close_raw  : raw unadjusted price shown to user
    # -----------------------------
    hfq_close = latest_row.get("Close", np.nan)
    real_close = latest_row.get("Close_raw", np.nan)

    if pd.isna(hfq_close) or pd.isna(real_close):
        st.error("缺少展示价格字段 Close / Close_raw。")
        st.stop()

    if hfq_close <= 0 or real_close <= 0:
        st.error("Close / Close_raw 必须为正数，无法计算目标价格。")
        st.stop()

    # -----------------------------
    # Core latest states
    # -----------------------------
    latest_slope = float(latest_row.get("kf_slope", 0.0))
    is_cycle_alive = bool(latest_row.get("is_cycle_alive", False))
    current_regime = "趋势 + 周期回调" if is_cycle_alive else "纯趋势跟踪"
    raw_signal_long = int(latest_row.get("signal_long", 0))
    raw_action = "买入" if raw_signal_long == 1 else "持有现金/卖出"

    pred_log_close_next = latest_row.get("pred_log_close_next", np.nan)
    current_log_close = latest_row.get("log_close", np.nan)

    if pd.isna(pred_log_close_next) or pd.isna(current_log_close):
        st.error("缺少 pred_log_close_next / log_close，无法计算目标价格。")
        st.stop()

    pred_log_close_next = float(pred_log_close_next)
    current_log_close = float(current_log_close)

    # -----------------------------
    # Signal explanation return
    # regime-aware explanation logic
    # -----------------------------
    if is_cycle_alive:
        expected_log_ret_signal = pred_log_close_next - current_log_close
    else:
        expected_log_ret_signal = latest_slope

    expected_ret_signal = float(np.exp(expected_log_ret_signal) - 1.0)

    # -----------------------------
    # Target price calculation
    # 1) convert predicted next log HFQ price -> HFQ price
    # 2) convert HFQ price -> raw price using today's adjustment factor
    # -----------------------------
    pred_close_hfq_next = float(np.exp(pred_log_close_next))

    adj_factor_today = float(hfq_close / real_close)
    adj_factor_today = max(adj_factor_today, 1e-12)

    target_price = float(pred_close_hfq_next / adj_factor_today)

    # Display return should be based on displayed raw price
    expected_ret_display = float(target_price / real_close - 1.0)
        # -----------------------------
    # Final display action
    # Avoid showing "buy" when displayed expected return is non-positive
    # -----------------------------
    min_display_upside = float(active_conf.get("min_display_upside", 0.0))

    if raw_signal_long == 1 and expected_ret_display > min_display_upside:
        action = "买入"
    elif raw_signal_long == 1 and expected_ret_display <= min_display_upside:
        action = "持有现金/卖出"
    else:
        action = "持有现金/卖出"

    # ---------------------------------------------------------
    # 空仓/卖出原因诊断逻辑 (Reason Diagnostics)
    # 使用 active_conf，而不是 KALMAN_PARAMS
    # ---------------------------------------------------------
    reason_msg = ""
    if action == "持有现金/卖出":
        reasons = []

        log_trend_thresh = np.log1p(float(active_conf["trend_threshold"]))
        log_cycle_slope_thresh = np.log1p(float(active_conf["cycle_slope_threshold"]))
        z_thresh = float(active_conf["cycle_z_threshold"])

        curr_trend = float(latest_row.get("trend_slope", 0.0))
        curr_cycle_slope = float(latest_row.get("cycle_slope", 0.0))
        curr_cycle_z = float(latest_row.get("cycle_z", 0.0))

        burn_in_days = max(
            20,
            int(active_conf["cycle_z_window"]),
            int(active_conf["ER_window"])
        )
        current_index = len(df) - 1
        is_in_burn_in = current_index < burn_in_days

        if is_in_burn_in:
            reasons.append("处于模型初始化预热期 (Burn-in)")

        # 展示层口径优先：若目标价不高于最新原始收盘价，则对用户而言没有可执行的上行空间
        no_display_upside = expected_ret_display <= 0

        if is_cycle_alive:
            if curr_trend <= log_trend_thresh:
                reasons.append("主趋势斜率偏弱或正在向下")
            if curr_cycle_slope <= log_cycle_slope_thresh:
                reasons.append("短周期动能拐头向下（处于下跌波段）")
            if curr_cycle_z >= z_thresh:
                reasons.append(f"短周期处于超买极值区域（Z-score: {curr_cycle_z:.2f}，防范回调）")

            # 先看展示口径，再看模型解释口径
            if no_display_upside:
                reasons.append("按原始价格口径计算，次日目标价不高于最新收盘价，上行空间不足")
            elif expected_log_ret_signal <= 0:
                reasons.append("卡尔曼滤波预测次日上行空间不足")

        else:
            if curr_trend <= log_trend_thresh:
                reasons.append("市场无明显周期规律，且主趋势偏弱（未达右侧单边趋势跟随标准）")

            if no_display_upside:
                reasons.append("按原始价格口径计算，次日目标价不高于最新收盘价，上行空间有限")
            elif expected_log_ret_signal <= 0:
                reasons.append("趋势延续性不足，模型预测次日上行空间有限")

        if reasons:
            reason_msg = "；".join(reasons) + "。"
        else:
            reason_msg = "当前未满足入场条件，但未检测到单一主导原因，建议结合图形与趋势状态综合判断。"

    # -----------------------------
    # Metrics display
    # -----------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("次日交易建议", action, help = "基于实际收盘价和模型预测价格")
    col2.metric("模型原始信号", raw_action, help = "基于模型过滤后的收盘价和模型预测价格")
    col3.metric("次日预期价格", f"¥{target_price:.2f}")
    col4.metric("次日预期回报", f"{expected_ret_display * 100:.2f}%")
    col5.metric("当前市场状态", current_regime)

    if raw_action == "买入" and action == "持有现金/卖出":
        st.warning("⚠️ 模型原始信号为买入，但按原始价格口径计算次日上行空间不足，信号被过滤。")
    if action == "持有现金/卖出":
        st.info(f"💡 **未触发买入原因分析**：{reason_msg}")
    elif action == "买入":
        if is_cycle_alive:
            st.success("✅ **满足买入条件**：主趋势向上，短周期动能改善，处于周期回调后再启动阶段。")
        else:
            st.success("✅ **满足买入条件**：周期特征弱化，但主趋势保持向上，符合右侧趋势跟随条件。")

    # Optional debug info
    with st.expander("查看目标价计算细节", expanded=False):
        debug_price_df = pd.DataFrame([
            {"项目": "最新原始收盘价 Close_raw", "值": real_close},
            {"项目": "最新后复权收盘价 Close", "值": hfq_close},
            {"项目": "当日复权因子 Close / Close_raw", "值": adj_factor_today},
            {"项目": "预测次日对数后复权价格 pred_log_close_next", "值": pred_log_close_next},
            {"项目": "预测次日后复权价格 exp(pred_log_close_next)", "值": pred_close_hfq_next},
            {"项目": "换算后的次日原始目标价", "值": target_price},
            {"项目": "用于信号解释的预期对数收益", "值": expected_log_ret_signal},
            {"项目": "用于展示的次日预期收益(原始价格口径)", "值": expected_ret_display},
        ])
        st.dataframe(debug_price_df, width="stretch")

    st.markdown("---")

    # ==========================================
    # 4. Filter Visualizations
    # ==========================================
    with st.expander("模型可视化", expanded=False):

        z_window = int(active_conf["cycle_z_window"])

        # -----------------------------
        # Chart 1: KF fit in log-price space
        # -----------------------------
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(df["Date"], df["log_close"], label="对数收盘价(后复权)", color="black", linestyle="--", alpha=0.7)
        ax1.plot(df["Date"], df["kf_level"], label="趋势", linewidth=2.5)
        ax1.plot(df["Date"], df["trend_plus_cycle"], label="趋势 + 周期拟合", linewidth=2, alpha=0.7)
        ax1.set_title("趋势 + 周期卡尔曼滤波")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)
        plt.close(fig1)

        # -----------------------------
        # Chart 2: Cycle component
        # -----------------------------
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(df["Date"], df["kf_cycle"], label="周期")
        ax2.axhline(0, color="black", linestyle="--")
        ax2.set_title(f"周期卡尔曼滤波 (周期天数={optimized_params['opt_cycle_days']:.2f})")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
        plt.close(fig2)

        # -----------------------------
        # Chart 3: Strategy NAV vs Buy & Hold
        # Use post-burn-in aligned sample
        # -----------------------------
        if len(df) <= z_window:
            st.warning("样本不足，无法绘制策略净值图。")
        else:
            ret_cols = ["stock_ret_today", "strategy_ret_today_after_cost"]
            aligned = df.iloc[z_window:].dropna(subset=ret_cols).copy()

            if len(aligned) == 0:
                st.warning("有效样本不足，无法绘制策略净值图。")
            else:
                aligned["bh_nav"] = (1.0 + aligned["stock_ret_today"]).cumprod()
                aligned["strat_nav"] = (1.0 + aligned["strategy_ret_today_after_cost"]).cumprod()
                aligned["excess_nav"] = aligned["strat_nav"] / aligned["bh_nav"] - 1.0

                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.plot(aligned["Date"], aligned["bh_nav"], label="买入并持有", color="C0")
                ax3.plot(aligned["Date"], aligned["strat_nav"], label="策略（非真实交易模拟）", color="C2")
                ax3.set_ylabel("单位净值 (初始为1)")
                ax3.set_title("策略 vs 买入并持有")
        ret_cols = ["stock_ret_today", "strategy_ret_today_after_cost"]
        aligned = df.iloc[z_window:].dropna(subset=ret_cols).copy()

        if len(aligned) == 0:
            st.warning("有效样本不足，无法绘制策略净值图。")
        else:
            # 用与评估指标完全一致的收益序列重新生成净值曲线
            aligned["bh_nav"] = (1.0 + aligned["stock_ret_today"]).cumprod()
            aligned["strat_nav"] = (1.0 + aligned["strategy_ret_today_after_cost"]).cumprod()
            aligned["excess_nav"] = aligned["strat_nav"] / aligned["bh_nav"] - 1.0

            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(aligned["Date"], aligned["bh_nav"], label="买入并持有", color="C0")
            ax3.plot(aligned["Date"], aligned["strat_nav"], label="策略（非真实交易模拟）", color="C2")
            ax3.set_ylabel("单位净值 (初始为1)")
            ax3.set_title("策略 vs 买入并持有")
            ax3.grid(True)

            ax4 = ax3.twinx()
            ax4.plot(
                aligned["Date"],
                aligned["excess_nav"] * 100,
                label="超额收益(%)",
                linestyle="--",
                alpha=0.85,
                color="C1"
            )
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
    st.subheader("模型原始信号回测表现（非真实交易模拟）")

    z_window = int(active_conf["cycle_z_window"])
    ret_cols = ["stock_ret_today", "strategy_ret_today_after_cost"]

    if len(df) <= z_window:
        st.warning("样本不足，无法展示评估指标。")
    else:
        aligned = df.iloc[z_window:].dropna(subset=ret_cols).copy()

        if len(aligned) == 0:
            st.warning("有效样本不足，无法展示评估指标。")
        else:
            bh_stats = calculate_perf_stats(aligned["stock_ret_today"])
            strat_stats = calculate_perf_stats(aligned["strategy_ret_today_after_cost"])

            if bh_stats and strat_stats:
                metrics_df = pd.DataFrame({
                    "买入并持有": bh_stats,
                    "策略（非真实交易模拟）": strat_stats
                })
                st.table(metrics_df)

            valid = aligned.dropna(subset=["pred_ret_1d", "real_ret_1d"]).copy()
            hit_ratio = (
                ((valid["pred_ret_1d"] > 0) == (valid["real_ret_1d"] > 0)).mean()
                if len(valid) > 0 else np.nan
            )

            avg_turnover = (
                aligned["turnover"].mean()
                if "turnover" in aligned.columns and len(aligned) > 0
                else np.nan
            )

            col_a, col_b = st.columns(2)
            col_a.metric("胜率", f"{hit_ratio * 100:.2f}%" if pd.notna(hit_ratio) else "N/A",help="模型预测次日涨跌方向与实际涨跌方向一致的比例。")
            col_b.metric("平均日调仓幅度", f"{avg_turnover * 100:.2f}%" if pd.notna(avg_turnover) else "N/A",help="策略每日仓位变动的平均幅度（0→1 为满仓买入，1→0 为清仓卖出）。")
# 📈 KF Timing App – A-Share Kalman Filter Trading Assistant

A Streamlit-based application for A-share market timing using a state-space model (Kalman Filter).  
The system decomposes price dynamics into **trend** and **cycle components**, routes decisions through **market-adaptive trading profiles**, and generates **right-side trading signals** based on statistically grounded state estimation.

---

## 🚀 Features

### Core Model
- 📊 **Kalman Filter State-Space Model**
  - Trend (level + slope)
  - Stochastic cycle component (sin/cos representation)

- 🔍 **MLE-based Parameter Optimization**
  - Automatic calibration of process noise (Q) and observation noise (R)
  - L-BFGS-B optimization

- 🧠 **Market-Adaptive Noise Modeling**
  - ER-driven process noise scaling (trend vs. noise detection)
  - Volume-driven observation noise scaling

- 🇨🇳 **A-Share Microstructure Awareness**
  - Limit-up / limit-down adjustment (shadow price mechanism)
  - Board-specific trading limits (Main, ChiNext, STAR, BSE)

- ⚙️ **Cycle Tradeability Gate**
  - Detects whether short-term cycle is "tradable"
  - Switches between trend-only and trend + cycle regimes

### Trading Style Profiles
- 🎯 **Five Preset Profiles**
  - **Trend Follower** – smooth, persistent trends
  - **Breakseeker** – breakout / ignition events (return + volume shock)
  - **Defender** – low tradability, defensive posture
  - **Activist** – transitional / high-curvature markets
  - **All Other** – neutral default

- 🔄 **Three Selection Modes**
  - **Recommended** – auto-detect profile from market structure (efficiency, coherence, curvature, breakout ignition)
  - **Manual** – pick a preset profile directly
  - **Custom** – tune thresholds, noise scaling, jump penalty, slippage, and more

### Signal & Analytics
- 📉 **Backtest Simulation**
  - Signal-based strategy returns with transaction cost & turnover modeling
  - Performance metrics (CAGR, Sharpe, Max Drawdown, hit ratio, avg turnover)

- 💡 **Actionable Signal Output**
  - Next-day trade recommendation (Buy / Hold Cash / Sell)
  - Model raw signal vs. display signal (filtered by upside on raw price)
  - Target price and expected return (HFQ → raw price conversion)
  - Regime classification (trend-only vs. trend + cycle)
  - No-buy reason diagnostics (trend weakness, cycle overbought, burn-in, etc.)

- 📊 **Long-Term Case Studies**
  - Pre-computed out-of-sample backtests for CATL (300750.SZ) and Kweichow Moutai (600519.SH)

### UI & Deployment
- 🌐 **Streamlit Dashboard**
  - Chinese-language interface with embedded Noto Sans SC font
  - Automatic Tushare stock/index detection from the entered code
  - Sidebar form for instrument code, date, rolling window (60 / 120 / 250 days), and style settings
  - Collapsible model visualizations (trend/cycle decomposition, NAV vs. buy & hold, excess return)
  - 10-second cooldown between model runs (API rate protection)

---

## 🧠 Model Intuition

The system separates the problem into three layers:

### 1. Estimation Layer (Kalman Filter)
- Extracts latent structure from noisy price series: trend and cycle
- Produces filtered states and next-day forecasts

### 2. Profile Layer (Market Routing)
- Classifies the current market environment into a trading profile
- Each profile overrides key hyperparameters (thresholds, noise trust, cycle sensitivity)
- Uses path efficiency, linear coherence, curvature, and breakout ignition signals

### 3. Decision Layer (Trading Logic)
- Determines whether cycle information is **tradable**
- Generates trading signals based on trend slope, cycle dynamics, and signal confidence
- Filters display signals when expected upside on raw price is insufficient

> The model is **not purely predictive**, but a **state estimation + regime routing + decision support system**.

---

## 📊 Example Outputs

- Latest trading signal (Buy / Hold Cash / Sell) with reason diagnostics
- Model raw signal vs. filtered display signal
- Target price and expected return (raw price basis)
- Active trading profile and diagnostic features
- Market regime classification (trend-only vs. trend + cycle)
- Strategy vs. buy & hold performance with excess return chart
- Cycle and trend decomposition plots
- Direction hit ratio and average daily turnover

---

## 🛠️ Installation (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser. Enter a Tushare stock code (e.g. `600519.SH`)
or index code (e.g. `000001.SH` or `000300.CSI`) and click **运行模型**.
The app checks `stock_basic` and then `index_basic` to select the appropriate
daily data endpoint automatically. ETFs and funds are not supported.

---

## 🔐 Secrets Configuration

Create a local file:

`.streamlit/secrets.toml`

Add:

```toml
TUSHARE_TOKEN = "your_token_here"
```

For Streamlit Cloud deployment, configure secrets in the web UI instead.

A valid [Tushare Pro](https://tushare.pro/) token is required for live A-share
data. Index requests use `index_basic` and `index_daily`; your token must have
permission to access those endpoints.

---

## 📦 Project Structure

```text
KF_Timing_App/
├── app.py                      # Streamlit UI and orchestration
├── config.py                   # Default params, profile configs, build_profile_config()
├── requirements.txt
├── utils/
│   ├── data_loader.py          # Tushare stock/index fetch, validation, limit rules
│   ├── kalman_model.py         # Feature prep, MLE optimization, Kalman filter
│   └── profile_selector.py     # Market structure classifier & profile routing
├── assets/
│   ├── fonts/
│   │   └── NotoSansSC-Regular.ttf
│   └── images/
│       ├── 300750.png          # CATL long-term backtest chart
│       └── 600519.png          # Moutai long-term backtest chart
└── .streamlit/
    └── secrets.toml            # Local secrets (not committed)
```

---

## ⚠️ Disclaimer

This project is for research and educational purposes only.  
It is not financial advice and should not be used for real trading without further validation.  
Backtest results shown in the app (including long-term case studies) do not include transaction fees or slippage unless configured in custom mode.

---

## 👤 Author

Developed by Lance Zhao  
UCLA Math/Econ | Quant Research | A-share Systematic Strategies

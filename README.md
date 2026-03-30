# 📈 KF Timing App – A-Share Kalman Filter Trading Assistant

A Streamlit-based application for A-share market timing using a state-space model (Kalman Filter).  
The system decomposes price dynamics into **trend** and **cycle components**, and generates **right-side trading signals** based on statistically grounded state estimation.

---

## 🚀 Features

- 📊 **Kalman Filter State-Space Model**
  - Trend (level + slope)
  - Stochastic cycle component (sin/cos representation)

- 🔍 **MLE-based Parameter Optimization**
  - Automatic calibration of process noise (Q) and observation noise (R)
  - L-BFGS-B optimization

- 🧠 **Market-Adaptive Noise Modeling**
  - ER-driven process noise scaling (trend vs noise detection)
  - Volume-driven observation noise scaling

- 🇨🇳 **A-Share Microstructure Awareness**
  - Limit-up / limit-down adjustment (shadow price mechanism)
  - Board-specific trading limits (Main, ChiNext, STAR, BSE)

- ⚙️ **Cycle Tradeability Gate**
  - Detects whether short-term cycle is “tradable”
  - Switches between:
    - Trend-only regime
    - Trend + cycle regime

- 📉 **Backtest Simulation**
  - Signal-based strategy returns
  - Transaction cost & turnover modeling
  - Performance metrics (CAGR, Sharpe, Max Drawdown)

- 🌐 **Streamlit Deployment**
  - Interactive UI
  - Real-time signal display
  - Visualization dashboard

---

## 🧠 Model Intuition

The system separates the problem into two layers:

### 1. Estimation Layer (Kalman Filter)
- Extracts latent structure from noisy price series:
  - Trend
  - Cycle
- Produces filtered states and forecasts

### 2. Decision Layer (Trading Logic)
- Determines whether cycle information is **tradable**
- Generates trading signals based on:
  - Trend slope
  - Cycle dynamics
  - Signal confidence

> The model is **not purely predictive**, but a **state estimation + decision support system**.

---

## 📊 Example Outputs

- Latest trading signal (Buy / Neutral)
- Target price and expected return
- Market regime classification
- Strategy vs Buy & Hold performance
- Cycle and trend decomposition plots

---

## 🛠️ Installation (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```
## 🔐 Secrets Configuration

Create a local file:

.streamlit/secrets.toml

Add:

TUSHARE_TOKEN = "your_token_here"

For Streamlit Cloud deployment, configure secrets in the web UI instead.

## 📦 Project Structure
```text
KF_Timing_App/
├── app.py
├── config.py
├── requirements.txt
├── utils/
│   ├── data_loader.py
│   └── kalman_model.py
├── assets/
│   └── fonts/
│       └── NotoSansSC-Regular.ttf
└── .streamlit/
    └── secrets.toml
```
## ⚠️ Disclaimer

This project is for research and educational purposes only.
It is not financial advice and should not be used for real trading without further validation.


## 👤 Author

Developed by Lance Zhao
UCLA Math/Econ | Quant Research | A-share Systematic Strategies

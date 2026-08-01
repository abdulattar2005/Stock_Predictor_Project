"""
Stock Price Predictor
======================
Predicts next-day closing stock prices using historical price/volume data.

Pipeline:
  1. Load historical data (yfinance if available, else CSV, else synthetic demo data)
  2. Engineer features (lag prices, moving averages, volatility, RSI, volume trend)
  3. Time-ordered train/test split (NEVER shuffle time series data)
  4. Train a Linear Regression baseline model
  5. Train a Random Forest and Gradient Boosting model (more advanced, non-linear)
  6. Evaluate with MAE, RMSE, R^2, and directional accuracy
  7. Plot actual vs. predicted prices and save results

Usage:
    python stock_predictor.py --ticker AAPL --start 2019-01-01 --end 2024-01-01
    python stock_predictor.py --csv my_stock_data.csv
    python stock_predictor.py --demo          # runs with synthetic data, no internet needed
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
 
# Try an interactive GUI backend so chart windows pop up (works on Windows/Mac/most
# desktop Linux). Falls back to Agg (save-to-file only, no popup) in headless
# environments like servers or CI, where no display is available.
try:
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    plt.figure()  # test that the backend can actually create a window
    plt.close()
except Exception:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
warnings.filterwarnings("ignore")
 
 
# ----------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------
def load_data(ticker=None, start=None, end=None, csv_path=None, demo=False):
    """
    Load historical OHLCV data from one of three sources:
      - a local CSV file (columns: Date, Open, High, Low, Close, Volume)
      - Yahoo Finance via yfinance (requires internet)
      - synthetic demo data (for offline testing / demonstration)
    Returns a DataFrame indexed by Date, sorted ascending.
    """
    if demo:
        print("[INFO] Generating synthetic demo data (no internet required)...")
        return _generate_synthetic_data()
 
    if csv_path:
        print(f"[INFO] Loading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        return df
 
    try:
        import yfinance as yf
        print(f"[INFO] Downloading {ticker} data from Yahoo Finance ({start} to {end})...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("No data returned — check ticker/date range or internet access.")
        # yfinance sometimes returns MultiIndex columns; flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"[WARN] Could not fetch live data ({e}).")
        if "not subscriptable" in str(e):
            print("[HINT] This usually means your yfinance version needs Python 3.9+.")
            print("[HINT] Fix: check `python --version`, then either upgrade Python to 3.9+")
            print("[HINT]      and run `pip install --upgrade yfinance`, OR keep your current")
            print("[HINT]      Python and run `pip install \"yfinance==0.2.28\"` instead.")
        print("[INFO] Falling back to synthetic demo data instead.")
        return _generate_synthetic_data()
 
 
def _generate_synthetic_data(n_days=1500, seed=42):
    """Creates a realistic-looking synthetic stock price series (random walk + drift + seasonality)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
 
    drift = 0.0003
    volatility = 0.018
    returns = rng.normal(drift, volatility, n_days)
    seasonal = 0.002 * np.sin(np.linspace(0, 40 * np.pi, n_days))
    returns += seasonal
 
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n_days)))
    open_ = low + (high - low) * rng.random(n_days)
    volume = rng.integers(1_000_000, 8_000_000, n_days)
 
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df
 
 
# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------
def engineer_features(df):
    """
    Builds predictive features from raw OHLCV data.
    Target: next day's closing price.
    """
    data = df.copy()
 
    # Lagged closing prices (previous 1, 2, 3, 5, 10 days)
    for lag in [1, 2, 3, 5, 10]:
        data[f"close_lag_{lag}"] = data["Close"].shift(lag)
 
    # Moving averages
    for window in [5, 10, 20]:
        data[f"ma_{window}"] = data["Close"].rolling(window).mean()
 
    # Volatility (rolling std of daily returns)
    data["daily_return"] = data["Close"].pct_change()
    data["volatility_10"] = data["daily_return"].rolling(10).std()
 
    # RSI (Relative Strength Index, 14-day)
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    data["rsi_14"] = 100 - (100 / (1 + rs))
 
    # Volume trend
    data["volume_ma_10"] = data["Volume"].rolling(10).mean()
    data["volume_change"] = data["Volume"].pct_change()
 
    # High-low spread (intraday volatility proxy)
    data["hl_spread"] = (data["High"] - data["Low"]) / data["Close"]
 
    # Target: next day's close
    data["target"] = data["Close"].shift(-1)
 
    data = data.dropna()
    return data
 
 
# ----------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (time-ordered, no shuffling!)
# ----------------------------------------------------------------------
def time_split(data, feature_cols, target_col="target", test_size=0.2):
    split_idx = int(len(data) * (1 - test_size))
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]
 
    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]
    return X_train, X_test, y_train, y_test, test.index
 
 
# ----------------------------------------------------------------------
# 4. MODELING
# ----------------------------------------------------------------------
def train_models(X_train, y_train, X_test, scaler):
    """Trains Linear Regression (baseline) and two ensemble models (advanced)."""
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
 
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }
 
    predictions = {}
    for name, model in models.items():
        if name == "Linear Regression":
            model.fit(X_train_s, y_train)
            predictions[name] = model.predict(X_test_s)
        else:
            # Tree-based models don't need scaling, but it doesn't hurt
            model.fit(X_train, y_train)
            predictions[name] = model.predict(X_test)
        models[name] = model
 
    return models, predictions
 
 
# ----------------------------------------------------------------------
# 5. EVALUATION
# ----------------------------------------------------------------------
def evaluate(y_test, predictions, y_train_last):
    """Prints MAE, RMSE, R^2, and directional accuracy for each model."""
    results = []
    actual_direction = np.sign(np.diff(np.concatenate([[y_train_last], y_test.values])))
 
    for name, y_pred in predictions.items():
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
 
        pred_direction = np.sign(np.diff(np.concatenate([[y_train_last], y_pred])))
        directional_acc = np.mean(pred_direction == actual_direction) * 100
 
        results.append(
            {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
             "Directional Accuracy (%)": directional_acc}
        )
 
    results_df = pd.DataFrame(results).set_index("Model")
    print("\n" + "=" * 60)
    print("MODEL EVALUATION (on held-out test period)")
    print("=" * 60)
    print(results_df.round(4).to_string())
    print("=" * 60)
    return results_df
 
 
# ----------------------------------------------------------------------
# 6. PLOTTING
# ----------------------------------------------------------------------
def plot_results(test_dates, y_test, predictions, out_path="prediction_results.png"):
    plt.figure(figsize=(13, 6))
    plt.plot(test_dates, y_test.values, label="Actual Price", color="black", linewidth=2)
 
    colors = {"Linear Regression": "#1f77b4", "Random Forest": "#ff7f0e", "Gradient Boosting": "#2ca02c"}
    for name, y_pred in predictions.items():
        plt.plot(test_dates, y_pred, label=f"{name} (Predicted)",
                  color=colors.get(name), linestyle="--", alpha=0.85)
 
    plt.title("Stock Price Prediction: Actual vs. Predicted (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=150)
    print(f"\n[INFO] Chart saved to: {out_path}")
 
 
# ----------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (bonus — shows what drives the advanced model)
# ----------------------------------------------------------------------
def plot_feature_importance(model, feature_cols, out_path="feature_importance.png"):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="#2ca02c")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Feature importance chart saved to: {out_path}")
 
 
# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stock Price Predictor")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--csv", type=str, default=None, help="Path to local CSV file instead of live download")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data (no internet needed)")
    args = parser.parse_args()
 
    df = load_data(ticker=args.ticker, start=args.start, end=args.end,
                    csv_path=args.csv, demo=args.demo)
 
    print(f"[INFO] Loaded {len(df)} rows of data.")
    data = engineer_features(df)
    print(f"[INFO] {len(data)} rows remain after feature engineering (initial rows dropped for rolling windows).")
 
    feature_cols = [c for c in data.columns if c not in ["target"]]
    # Keep only the engineered numeric features (drop raw OHLCV to avoid leakage/redundancy is optional;
    # here we keep Close itself since "today's close" predicting "tomorrow's close" is standard and valid)
    feature_cols = [c for c in feature_cols if c not in ["Open", "High", "Low"]]
 
    X_train, X_test, y_train, y_test, test_dates = time_split(data, feature_cols)
    print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
 
    scaler = StandardScaler()
    models, predictions = train_models(X_train, y_train, X_test, scaler)
 
    y_train_last = y_train.iloc[-1]
    evaluate(y_test, predictions, y_train_last)
 
    plot_results(test_dates, y_test, predictions)
    plot_feature_importance(models["Random Forest"], feature_cols)
 
    # Predict the very next trading day's close using the most recent row
    latest_features = data[feature_cols].iloc[[-1]]
    latest_scaled = scaler.transform(latest_features)
    print("\n[NEXT-DAY PREDICTIONS]")
    print(f"  Linear Regression : {models['Linear Regression'].predict(latest_scaled)[0]:.2f}")
    print(f"  Random Forest     : {models['Random Forest'].predict(latest_features)[0]:.2f}")
    print(f"  Gradient Boosting : {models['Gradient Boosting'].predict(latest_features)[0]:.2f}")
 
    print("\n[INFO] Opening chart windows... close them to end the program.")
    plt.show()
 
 
if __name__ == "__main__":
    main()
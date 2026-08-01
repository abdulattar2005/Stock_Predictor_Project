Stock Price Predictor

A machine learning project that predicts next-day stock closing prices from historical price/volume data, using both a Linear Regression baseline and more advanced ensemble models (Random Forest, Gradient Boosting).# Stock_Predictor_Project.

How to run it

pip install yfinance scikit-learn pandas numpy matplotlib

 Real data (needs internet):
python stock_predictor.py --ticker AAPL --start 2018-01-01 --end 2024-01-01

 Your own CSV (columns: Date, Open, High, Low, Close, Volume):
python stock_predictor.py --csv my_data.csv

 No internet? Run with built-in synthetic demo data:
python stock_predictor.py --demo

It prints evaluation metrics to the console and saves two charts: prediction_results.png (actual vs. predicted) and feature_importance.png.

Project pipeline (this is the story to tell in your writeup/presentation)
1. Data collection

Historical daily OHLCV (Open/High/Low/Close/Volume) data is pulled via yfinance. Stock prices are a time series, so this is different from typical ML datasets — order matters, and you can never train on the future to predict the past.

2. Feature engineering

Raw prices alone are a poor input (a $150 stock and a $1500 stock aren't comparable), so the model instead learns from derived signals:

Lagged closes (price 1, 2, 3, 5, 10 days ago) — recent momentum
Moving averages (5/10/20-day) — trend direction
Volatility (rolling std of returns) — how turbulent the stock has been
RSI (Relative Strength Index) — a classic technical indicator for overbought/oversold conditions
Volume trend — whether trading activity is rising or falling
High-low spread — intraday volatility proxy

The target is next day's closing price (Close.shift(-1)).

3. Train/test split — the most important detail

Unlike normal ML problems, you must never shuffle time series data before splitting. This project uses a chronological split: the model trains on the earlier ~80% of days and is tested on the most recent ~20%, exactly as it would be used in practice (predicting tomorrow from what you know today).

4. Models
Linear Regression — the required baseline. Assumes a linear relationship between features and next-day price. Fast, interpretable, surprisingly strong on price data because yesterday's close is usually the best predictor of tomorrow's close.
Random Forest and Gradient Boosting — the "more advanced techniques." These capture non-linear interactions (e.g., "high volatility + falling volume" behaves differently than either signal alone) that linear regression cannot.
5. Evaluation metrics
MAE (Mean Absolute Error) — average dollar error
RMSE (Root Mean Squared Error) — penalizes large misses more heavily
R² — how much of the price variance the model explains
Directional Accuracy — % of days the model correctly predicted up vs. down movement. This one matters a lot in finance: a model can have a great R² just by "predicting" that tomorrow ≈ today, while still being useless for actually trading on direction.
6. Visualization

prediction_results.png overlays actual vs. predicted prices on the held-out test period so you can visually judge how well each model tracks real movement (and where it lags behind sudden moves — a common, expected weakness of these models).

Honest limitations to mention in your report
Stock prices are close to a random walk; a naive "tomorrow = today" model is a very tough baseline to beat, and R² can look deceptively high without adding real predictive value.
This model uses only price/volume history — no news, earnings, macro data, or sentiment.
Past performance patterns are not guaranteed to hold in the future.
This is an educational project, not financial advice or a trading system.
Extending the project further (great for extra credit)
LSTM/GRU (deep learning): reshape lagged features into sequences and use tensorflow.keras.layers.LSTM for a true sequence model. Needs more data and tuning but is the natural "next level" beyond ensembles.
Multiple tickers: loop the pipeline over a list of tickers and compare accuracy across sectors (e.g., tech vs. utilities — utilities are usually more predictable).
Hyperparameter tuning: use GridSearchCV (careful: use TimeSeriesSplit, not regular k-fold, to avoid leaking future data into training folds).
Add sentiment features: pull news headlines and add a sentiment score as a feature.
Backtest a simple trading strategy: convert predictions into buy/sell signals and simulate returns vs. buy-and-hold, to show whether directional accuracy translates into practical value.

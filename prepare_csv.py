"""
prepare_csv.py
==============
Converts the multi-asset CSV (Date + Price/Vol. columns for many assets in one file)
into the single-stock Date,Open,High,Low,Close,Volume format that stock_predictor.py expects.

Usage:
    python prepare_csv.py --input my_stock_data.csv --asset Apple --output apple_clean.csv

Available --asset names in this file:
    Natural_Gas, Crude_oil, Copper, Bitcoin, Platinum, Ethereum, S&P_500 (no volume),
    Nasdaq_100, Apple, Tesla, Microsoft, Silver, Google, Nvidia, Berkshire,
    Netflix, Amazon, Meta, Gold
"""

import argparse
import pandas as pd


def clean_numeric(series):
    """Removes thousand-separator commas and converts to float."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .replace("nan", pd.NA)
        .astype(float)
    )


def main():
    parser = argparse.ArgumentParser(description="Reshape multi-asset CSV into single-stock format")
    parser.add_argument("--input", required=True, help="Path to the raw multi-asset CSV")
    parser.add_argument("--asset", required=True, help="Asset name prefix, e.g. Apple, Tesla, Bitcoin")
    parser.add_argument("--output", required=True, help="Path to write the cleaned single-stock CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    price_col = f"{args.asset}_Price"
    vol_col = f"{args.asset}_Vol."

    if price_col not in df.columns:
        available = sorted(set(c.replace("_Price", "").replace("_Vol.", "") for c in df.columns if c != "Date" and c != "Unnamed: 0"))
        raise SystemExit(f"[ERROR] '{args.asset}' not found. Available assets: {available}")

    # Parse dates (file mixes formats like "31-01-2024" and "2/2/2024" in the
    # same column, so format='mixed' parses each row individually instead of
    # guessing one format for the whole column, which silently drops rows).
    dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")

    close = clean_numeric(df[price_col])

    if vol_col in df.columns:
        volume = clean_numeric(df[vol_col])
    else:
        print(f"[WARN] No volume column found for {args.asset}; filling Volume with 0.")
        volume = pd.Series(0, index=df.index)

    out = pd.DataFrame({
        "Date": dates,
        # No separate Open/High/Low is available in the source file, so we use
        # Close for all three. This is a standard simplification when only a
        # daily closing price is provided (it just means the "hl_spread" feature
        # in stock_predictor.py will be flat/uninformative for this data).
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": volume,
    })

    out = out.dropna(subset=["Date", "Close"])
    out = out.sort_values("Date").reset_index(drop=True)
    out["Volume"] = out["Volume"].ffill().fillna(0)

    out.to_csv(args.output, index=False)
    print(f"[INFO] Wrote {len(out)} rows to {args.output}")
    print(out.head())
    print(out.tail())


if __name__ == "__main__":
    main()

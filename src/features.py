import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from utils import get_logger




def load_prices_monthly(db_path: str, csv_path: str, logger) -> pd.DataFrame:
    """Load monthly prices from a SQLite database or a CSV file.

    Args:
        db_path (str): Path to the SQLite database file.
        csv_path (str): Path to the CSV file.
        logger (Logger): Logger instance for logging.

    Returns:
        pd.DataFrame: DataFrame containing the monthly prices.
    """
    try:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql("SELECT * FROM prices_monthly", conn, parse_dates=["date"]).set_index("date").sort_index()
        logger.info(f"Loaded monthly prices from SQLite Database at: {db_path} (table 'prices_monthly')")
        return df
    
    except Exception as e:
        logger.warning(f"SQLite load failed ({e}); trying CSV fallback: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
        logger.info(f"Loaded monthly prices from CSV at: {csv_path}")
        return df





def rsi(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    """Compute the Relative Strength Index (RSI) for a given price series.

    Args:
        prices (pd.DataFrame): DataFrame of prices with DatetimeIndex.
        period (int): Period for RSI calculation.

    Returns:
        pd.DataFrame: DataFrame containing the RSI values.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi





def macd(prices: pd.DataFrame, fast=12, slow=26, signal=9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the Moving Average Convergence Divergence (MACD) for a given price series.

    Args:
        prices (pd.DataFrame): DataFrame of prices with DatetimeIndex.
        fast (int, optional): Fast EMA period. Defaults to 12.
        slow (int, optional): Slow EMA period. Defaults to 26.
        signal (int, optional): Signal line period. Defaults to 9.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: MACD line and signal line.
    """
    macd_line = prices.ewm(span=fast, adjust=False).mean() - prices.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line





def compute_features(prices_monthly: pd.DataFrame, ret_windows: tuple[int],
                      vol_windows: tuple[int], rsi_period: int,
                      macd_cfg: tuple[int]) -> pd.DataFrame:
    """Compute technical features from monthly prices.

    Args:
        prices_monthly (pd.DataFrame): Monthly prices DataFrame.
        ret_windows (tuple[int]): Return windows for momentum features.
        vol_windows (tuple[int]): Volatility windows.
        rsi_period (int): Period for RSI calculation.
        macd_cfg (tuple[int]): Configuration for MACD (fast, slow, signal).

    Returns:
        pd.DataFrame: DataFrame containing the computed features.
    """
    features = {}

    # Returns / momentum features
    for w in ret_windows:
        features[f"r{w}m"] = prices_monthly.pct_change(w)
        
    # Volatility
    for w in vol_windows:
        features[f"vol{w}"] = prices_monthly.pct_change().rolling(w).std()

    # RSI
    features[f"RSI{rsi_period}"] = rsi(prices_monthly, rsi_period)
    
    # MACD
    fast, slow, sig = macd_cfg
    macd_line, sig_line = macd(prices_monthly, fast, slow, sig)
    features["MACD"] = macd_line
    features["MACDsig"] = sig_line

    # Concatenate on columns and flatten names to TICKER_metric
    out = []
    for name, df in features.items():
        df = df.add_suffix(f"_{name}")
        out.append(df)
    features = pd.concat(out, axis=1).sort_index()
    features.index.name = "date"
    return features




def save_outputs(features: pd.DataFrame, out_csv: str, out_db: str, logger) -> None:
    """Save the computed features to the specified outputs.

    Args:
        features (pd.DataFrame): DataFrame containing the computed features.
        out_csv (str | None): Path to save the features as a CSV file.
        out_db (str | None): Path to save the features to a SQLite database.
        logger (_type_): Logger instance for logging.
    """
    # Save to SQLite Database
    if out_db:
        import sqlite3
        out_db = Path(out_db)
        
        try:
            with sqlite3.connect(out_db) as conn:
                features.to_sql("features_monthly", conn, if_exists="replace", index=True, index_label="date")
            logger.info(f"Saved features to SQLite DB at: {out_db} (table 'features_monthly')")
        except Exception as e:
            logger.error(f"Failed to save features to SQLite DB at: {out_db}. Error: {e}. Skipping.")
    
    # Save CSV     
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_csv)
        logger.info(f"Saved monthly prices CSV to: {out_csv}")
        
        
        
        
        


def main(db_path: str, csv_path: str, out_csv: str, out_db: str,
         ret_windows: tuple[int], vol_windows: tuple[int], 
         rsi_period: int, macd_cfg: tuple[int]) -> None:
    """Main function to compute and save technical features.

    Args:
        db_path (str): Path to the SQLite database.
        csv_path (str): Path to the CSV fallback for monthly prices.
        out_csv (str): Path to save the features as a CSV file.
        out_db (str): Path to save the features to a SQLite database.
        ret_windows (tuple[int]): Return windows for momentum features.
        vol_windows (tuple[int]): Volatility windows.
        rsi_period (int): Period for RSI calculation.
        macd_cfg (tuple[int]): Configuration for MACD (fast, slow, signal).
    """
    # Get logger
    logger = get_logger("features")

    # Load monthly prices and compute features
    prices_monthly = load_prices_monthly(db_path, csv_path, logger)
    features = compute_features(
        prices_monthly,
        ret_windows=ret_windows,
        vol_windows=vol_windows,
        rsi_period=rsi_period,
        macd_cfg=macd_cfg,
    )

    # Save outputs
    save_outputs(features, out_csv, out_db, logger)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate technical features on monthly prices.")
    parser.add_argument("--db", type=str, default="data/data.db", help="SQLite with 'prices_monthly'")
    parser.add_argument("--csv", type=str, default="data/processed/prices_monthly.csv",
                        help="CSV fallback for monthly prices")
    parser.add_argument("--out_csv", type=str, default="data/processed/features_monthly.csv",
                        help="Where to save features CSV")
    parser.add_argument("--out_db", type=str, default="data/data.db",
                        help="SQLite DB to save 'features_monthly' (empty to skip)")
    parser.add_argument("--ret_windows", type=int, nargs="*", default=[1,3,6,12], help="Return windows for momentum features")
    parser.add_argument("--vol_windows", type=int, nargs="*", default=[3,6], help="Volatility windows")
    parser.add_argument("--rsi", type=int, default=14, help="Period for RSI calculation")
    parser.add_argument("--macd", type=int, nargs=3, default=[12,26,9], help="Configuration for MACD (fast, slow, signal)")
    args = parser.parse_args()
    
    main(args.db, args.csv, args.out_csv, args.out_db, args.ret_windows, args.vol_windows, args.rsi, args.macd)
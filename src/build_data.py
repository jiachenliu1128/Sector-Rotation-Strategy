# src/build_dataset.py
from __future__ import annotations
import argparse, sqlite3
from pathlib import Path
import pandas as pd
from utils import get_logger



def read_table(db: str, table: str, csv_fallback: str | None = None) -> pd.DataFrame:
    """Read a table from a SQLite database or fallback to a CSV file.

    Args:
        db (str): Path to SQLite DB file.
        table (str): Name of the table to read.
        csv_fallback (str | None, optional): Path to CSV file to fallback on. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    # Try to read a table from SQLite
    with sqlite3.connect(db) as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["date"]).set_index("date").sort_index()
            return df
        except Exception:
            if not csv_fallback:
                raise RuntimeError(f"Failed to read table '{table}' from DB '{db}' and no CSV fallback provided.")
    # fallback if we cannot read from DB
    df = pd.read_csv(csv_fallback, parse_dates=["date"]).set_index("date").sort_index()
    return df







def get_targets(prices_monthly: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """
    Target: next-month excess return vs benchmark.
    y_{i,t+1} = r_{i,t+1} - r_{bench,t+1}
    
    Args:
        prices_monthly (pd.DataFrame): Monthly prices with tickers as columns.
        benchmark (str): Benchmark ticker for excess return calculation.
        
    Returns:
        pd.DataFrame: DataFrame of targets with columns y_{ticker}.
    """
    # Monthly returns of all tickers
    r1 = prices_monthly.pct_change(1)  
    if benchmark not in r1.columns:
        raise KeyError(f"Benchmark '{benchmark}' not in monthly prices columns.")
    
    # Excess return next month
    y = r1.sub(r1[benchmark], axis=0) 
    y = y.shift(-1) 
    
    # Drop target for benchmark itself 
    if benchmark in y.columns:
        y = y.drop(columns=[benchmark])
        
    # Rename columns to indicate target
    y.columns = [f"y_{c}" for c in y.columns]

    return y





def align_and_join(features_monthly: pd.DataFrame, macro_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Broadcast macro alongside per-ticker features and join. 
    
    Args:
        features_monthly (pd.DataFrame): Monthly features with tickers as columns.
        macro_monthly (pd.DataFrame): Monthly macro data with same index as features_monthly.
        
    Returns:
        pd.DataFrame: Combined DataFrame with features and macro data.
    """
    # Ensure flat macro columns
    macro_monthly = macro_monthly.copy()
    macro_monthly.columns = [str(c) for c in macro_monthly.columns]
    # Join (inner keeps only overlapping months)
    X = features_monthly.join(macro_monthly, how="inner")
    return X





def save_outputs(X: pd.DataFrame, y: pd.DataFrame, out_csv_dir: str, out_db: str, logger) -> None:
    """Save the model inputs and targets to CSV and SQLite.

    Args:
        X (pd.DataFrame): features DataFrame.
        y (pd.DataFrame): targets DataFrame.
        out_csv_dir (str): Directory to save output CSV files.
        out_db (str): SQLite DB to save output tables (if empty, skip).
        logger (_type_): Logger instance for logging.
    """
    # Save to SQLite DB if path provided
    try:
        out_db = Path(out_db)
        with sqlite3.connect(out_db) as conn:
            X.to_sql("X_features", conn, if_exists="replace", index=True, index_label="date")
            y.to_sql("y_targets",  conn, if_exists="replace", index=True, index_label="date")
        logger.info(f"Wrote tables 'X_features' and 'y_targets' to SQLite DB: {out_db}")
    except Exception as e:
        logger.warning(f"Failed to write to SQLite DB at: {out_db}. Error: {e}. Skipping.")
        
    # Save to CSV files
    out_dir = Path(out_csv_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_path = out_dir / "X_features.csv"
    y_path = out_dir / "y_targets.csv"
    X.to_csv(X_path)
    y.to_csv(y_path)
    logger.info(f"Saved features at: {X_path}")
    logger.info(f"Saved targets at: {y_path}")






def main(db: str, prices_csv: str, features_csv: str, macro_csv: str, benchmark: str, out_csv_dir: str, out_db: str) -> None:
    """Build the dataset by loading tables, creating targets, and saving outputs.
    
    Args:
        db (str): Path to SQLite DB with input tables.
        prices_csv (str): Fallback CSV for prices_monthly if DB read fails.
        features_csv (str): Fallback CSV for features_monthly if DB read fails.
        macro_csv (str): Fallback CSV for macro_monthly if DB read fails.
        benchmark (str): Benchmark ticker for excess return target.
        out_dir (str): Directory to save output CSV files.
        out_db (str): SQLite DB to save output tables (if empty, skip).
    """
    # Setup logging
    logger = get_logger("build_dataset")

    # Load data
    prices_monthly = read_table(db, "prices_monthly", csv_fallback=prices_csv)
    features_monthly = read_table(db, "features_monthly", csv_fallback=features_csv)
    macro_monthly = read_table(db, "macro_monthly", csv_fallback=macro_csv)

    # Calculate targets based on benchmark excess returns
    y = get_targets(prices_monthly, benchmark)

    # Align feature rows to target index 
    X = align_and_join(features_monthly, macro_monthly)
    
    # Keep dates where we have both X_t and y_{t+1}
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    # Basic NA handling: drop rows with any NA in X or y (keep last row for live inference if desired)
    na_rows = X.index[X.isna().any(axis=1) | y.isna().any(axis=1)]
    if len(na_rows) > 0:
        logger.warning(f"Dropping {len(na_rows)} rows with NaNs (in features or targets).")
        X = X.drop(index=na_rows)
        y = y.drop(index=na_rows)

    save_outputs(X, y, out_csv_dir, out_db, logger)







if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Build modeling dataset: join features + macro and create targets.")
    args.add_argument("--db", default="data/data.db", help="SQLite DB with prices_monthly, features_monthly, macro_monthly")
    args.add_argument("--prices_csv", default="data/processed/prices_monthly.csv")
    args.add_argument("--features_csv", default="data/processed/features_monthly.csv")
    args.add_argument("--macro_csv", default="data/processed/macro_monthly.csv")
    args.add_argument("--benchmark", default="SPY")
    args.add_argument("--out_csv_dir", default="data/processed")
    args.add_argument("--out_db", default="data/data.db")
    args = args.parse_args()
    main(args.db, args.prices_csv, args.features_csv, args.macro_csv, args.benchmark, args.out_csv_dir, args.out_db)
# src/data_prep.py
import argparse
from pathlib import Path
import pandas as pd
from utils import get_logger




def load_prices_from_sqlite(db_path: str, table: str) -> pd.DataFrame:
    """Load daily ETF prices from SQLite. Expects a 'date' column.
    
    Args:
        db_path (str): Path to SQLite DB file.
        table (str): Table name to load.
        
    Returns:
        pd.DataFrame: DataFrame with 'date' as index.
    """
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df




def load_prices_fallback_csv(csv_path: str) -> pd.DataFrame:
    """Load daily ETF prices from CSV. Expects a 'date' column.
    
    Args:
        csv_path (str): Path to CSV file.
        
    Returns:
        pd.DataFrame: DataFrame with 'date' as index. 
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df



def price_basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex, sorted, drop exact-duplicate rows, ensure float cols.
    
    Args:
        df (pd.DataFrame): Input DataFrame with DatetimeIndex.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    out = df.copy()
    
    # Drop duplicate dates, keep last
    if not out.index.is_monotonic_increasing:
        out = out[~out.index.duplicated(keep="last")].sort_index()
        
    # Coerce columns to numeric where possible
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        
    # Drop columns that are entirely NaN
    out = out.dropna(axis=1, how="all")
    out.index.name = "date"
    return out





def to_prices_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Keep daily prices as is (no resampling).

    Args:
        df (pd.DataFrame): Daily prices DataFrame with DatetimeIndex.

    Returns:
        pd.DataFrame: Daily prices DataFrame with date as index.
    """
    daily = df.copy()
    daily.index.name = "date"
    return daily




def validate_prices_daily(df: pd.DataFrame, logger) -> None:
    """Basic sanity checks; raise on fatal issues, log warnings otherwise.
    
    Args:
        df (pd.DataFrame): Daily prices DataFrame.
        logger (Logger): Logger for status messages.

        Raises:
            ValueError: If any validation checks fail.
    """
    if df.empty:
        raise ValueError("Daily prices are empty.")
    if df.index.has_duplicates:
        raise ValueError("Daily prices index has duplicate dates.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Daily prices index is not sorted.")
    
    # Warn on missing values
    na_percentage = df.isna().mean().sort_values(ascending=False)
    has_na = na_percentage[na_percentage > 0]
    if not has_na.empty:
        logger.warning("Daily prices contain NaNs. Top offenders:\n%s", has_na.head(10))
        
        
        

def save_prices_daily(df: pd.DataFrame, out_csv: str, out_db: str, logger) -> None:
    """Save cleaned daily prices to CSV and optionally to SQLite.

    Args:
        df (pd.DataFrame): Daily prices DataFrame.
        out_csv (str): Path to save CSV file.
        out_sqlite (str): Path to SQLite DB file to save table (if empty, skip).
        logger (Logger): Logger for status messages.
    """
    # Save to SQLite Database
    if out_db:
        import sqlite3
        out_db = Path(out_db)
        
        try:
            with sqlite3.connect(out_db) as conn:
                df.to_sql("prices_daily", conn, if_exists="replace", index=True, index_label="date")
            logger.info(f"Saved daily prices table to SQLite DB at: {out_db} (table 'prices_daily')")
        except Exception as e:
            logger.warning(f"Failed to save daily prices table to SQLite DB at: {out_db}. Error: {e}. Skipping.")

    # Save CSV
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv)
        logger.info(f"Saved daily prices CSV to: {out_csv}")





def main(db_path: str, csv_path: str, out_csv: str, out_db: str) -> None:
    """Main function to load, clean ETF data and save to CSV/SQLite (daily frequency).

    Args:
        db_path (str): Path to SQLite DB with raw ETF data.
        csv_path (str): Path to CSV file with raw ETF data if SQLite not available.
        out_csv (str): Path to save cleaned daily prices CSV.
        out_db (str): Path to SQLite DB to save cleaned daily prices table.
    """
    logger = get_logger("etf_data_prep")
    
    # Check db path, fallback to CSV if not exists
    prices_daily = None
    try:
        prices_daily = load_prices_from_sqlite(db_path, table="etf_data")
        logger.info(f"Loaded daily prices from SQLite Database at: {db_path} (table 'etf_data')")
    except Exception as e:
        logger.warning(f"SQLite load failed ({e}); trying CSV fallback: {csv_path}")
        prices_daily = load_prices_fallback_csv(csv_path)
        logger.info(f"Loaded daily prices from CSV at: {csv_path}")

    # Clean and validate (no resampling to monthly)
    prices_daily = price_basic_clean(prices_daily)
    prices_daily = to_prices_daily(prices_daily)
    validate_prices_daily(prices_daily, logger)

    # Save results
    save_prices_daily(prices_daily, out_csv, out_db, logger)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data preparation and cleaning for ETF data, load daily ETF prices, clean, resample to monthly (EOM)."
    )
    parser.add_argument("--db", type=str, default="data/data.db", help="Path to SQLite DB with table 'etf_data'.")
    parser.add_argument("--csv", type=str, default="data/raw/etf_data.csv",
                        help="CSV file to use if SQLite not available.")
    parser.add_argument("--out_csv", type=str, default="data/processed/prices_daily.csv",
                        help="Where to save daily prices CSV.")
    parser.add_argument("--out_db", type=str, default="data/data.db",
                        help="SQLite DB file to store 'prices_daily' table (set empty to skip).")
    args = parser.parse_args()
    
    main(args.db, args.csv, args.out_csv, args.out_db)
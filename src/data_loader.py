import argparse
import argparse
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import yaml
from dotenv import load_dotenv
from utils import get_logger
import sqlite3




def save_to_sqlite(db_path: Path, table_name: str, df: pd.DataFrame, logger) -> None:
    """Save a DataFrame to a SQLite database, replacing the table.

    Args:
        db_path (Path): Path to the SQLite database file.
        table_name (str): Name of the table to write.
        df (pd.DataFrame): DataFrame to write. Index will be saved under 'date'.
        logger (Logger): Logger for status messages.
    """
    # Skip if DataFrame is empty
    if df is None or df.empty:
        logger.warning(f"Skip saving table '{table_name}': empty DataFrame")
        return
    
    # Ensure parent dir exists and save to SQLite
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=True, index_label="date")
    logger.info(f"Saved table '{table_name}' to SQLite database at '{db_path}'")





def fetch_etf_data(tickers: List[str], start: str, end: str, logger) -> pd.DataFrame:
    """Fetch historical ETF data from Yahoo Finance.

    Args:
        tickers (List[str]): List of ETF tickers to fetch.
        start (str): Start date for the data retrieval.
        end (str): End date for the data retrieval.
        logger (_type_): Logger for status messages.

    Raises:
        RuntimeError: If data retrieval fails.

    Returns:
        pd.DataFrame: DataFrame containing the fetched ETF data.
    """
    import yfinance as yf
    
    logger.info(f"Downloading ETF data for {len(tickers)} tickers from {start} to {end}...")
    data = {}
    
    # Download data for each ticker
    for t in tickers:
        logger.info(f"Downloading data for {t}...")
        df = yf.download(t, start=start, end=end, progress=True)
        if df.empty:
            logger.warning(f"No data downloaded for {t}")
            raise RuntimeError(f"No data downloaded for {t}. Check ticker and network.")
        data[t] = df["Adj Close"].rename(t)
    if not data:
        raise RuntimeError("No ETF data downloaded. Check tickers and network.")
    
    # Concatenate all dataframes into a single dataframe
    etf_data = pd.concat(data.values(), axis=1, keys=data.keys())
    etf_data.index.name = "date"
    return etf_data




def fetch_fred_data(series: List[str], logger) -> Dict[str, pd.DataFrame]:
    """Fetch historical data from the FRED API.

    Args:
        series (List[str]): List of FRED series IDs to fetch.
        logger (_type_): Logger for status messages.

    Raises:
        RuntimeError: If data retrieval fails.
        RuntimeError: If no data is found for a series.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames indexed by series ID.
    """
    from fredapi import Fred
    
    # Load FRED API key from environment variable
    load_dotenv() 
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FRED_API_KEY not found in environment variables.")
    
    # Initialize FRED client
    fred = Fred(api_key=api_key)
    data = {}
    logger.info(f"Downloading FRED data for {len(series)} series...")
    
    # Download each series
    for s in series:
        logger.info(f"Downloading data for {s}...")
        s_data = fred.get_series(s)
        if s_data is None or len(s_data) == 0:
            logger.warning(f"Empty series for {s}")
            raise RuntimeError(f"No data downloaded for FRED series {s}. Check series ID and network.")
        df = pd.DataFrame({"value": s_data})
        df.index.name = "date"
        data[s] = df
    return data





def main(config_path: str, data_dir: str):
    """Main function to load and process data.

    Args:
        config_path (str): Path to the configuration file.
        data_dir (str): Path to the data directory.
    """
    # Initialize logger
    logger = get_logger("data_loader")

    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    start = config["date_range"]["start"]
    end = config["date_range"]["end"]
    etf_tickers = config["etf_tickers"]
    etf_benchmark = config.get("etf_benchmark", "SPY")
    fred_series = config["fred_series"]
    
    # Sanity check for config
    if not etf_tickers:
        logger.error("ETF tickers list is empty in the configuration.")
        return
    if not fred_series:
        logger.error("FRED series list is empty in the configuration.")
        return
    if etf_benchmark in etf_tickers:
        logger.warning(f"Benchmark ETF '{etf_benchmark}' is also in the ETF tickers list. Removing it from tickers to avoid duplication.")
        etf_tickers.remove(etf_benchmark)
    
    # Define database directory
    data_dir = Path(data_dir)
    raw_data_dir = data_dir / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "data.db"
    
    # Fetch and store ETF data into csv and the database
    all_etf_tickers = etf_tickers + [etf_benchmark]
    etf_data = fetch_etf_data(all_etf_tickers, start, end, logger)
    
    out_path = raw_data_dir / "etf_data.csv"
    etf_data.to_csv(out_path)
    logger.info(f"Saved ETF data csv at: {out_path}")
    
    save_to_sqlite(db_path, "etf_data", etf_data, logger)
    
    # Save calendar index from etf data to csv and the database (for later time alignment)
    cal = pd.DataFrame(index=etf_data.index.copy())
    cal.index.name = "date"
    out_path = raw_data_dir / "calendar.csv"
    cal.to_csv(out_path)
    logger.info(f"Saved trading calendar data csv at: {out_path}")

    save_to_sqlite(db_path, "calendar", cal, logger)

    # Fetch and store FRED data into csv and the database
    fred_data = fetch_fred_data(fred_series, logger)
    
    for s, df in fred_data.items():
        out_path = raw_data_dir / f"fred_{s}.csv"
        df.to_csv(out_path)
        logger.info(f"Saved FRED series {s} data csv at: {out_path}")
        save_to_sqlite(db_path, s, df, logger)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Loader for ETFs (Exchange-Traded Funds) and FRED (Federal Reserve Economic Data)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the data directory")
    args = parser.parse_args()
    main(args.config, args.data_dir)
    

import argparse
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import yaml
from dotenv import load_dotenv
from utils import get_logger




def save_to_sqlite(db_path: Path, table_name: str, index_label: str, df: pd.DataFrame, logger) -> None:
    """Save a DataFrame to a SQLite database, replacing the table.

    Args:
        db_path (Path): Path to the SQLite database file.
        table_name (str): Name of the table to write.
        index_label (str): Name of the index column.
        df (pd.DataFrame): DataFrame to write. Index will be saved under 'date'.
        logger (Logger): Logger for status messages.
    """
    import sqlite3
    
    # Skip if DataFrame is empty
    if df is None or df.empty:
        logger.warning(f"Skip saving table '{table_name}': empty DataFrame")
        return
    
    # Ensure parent dir exists and save to SQLite
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=True, index_label=index_label)
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
        # Prefer 'Adj Close' if available, otherwise use 'Close'
        if "Adj Close" in df.columns:
            series = df["Adj Close"].copy()
            series.name = t
        elif "Close" in df.columns:
            series = df["Close"].copy()
            series.name = t
        else:
            raise KeyError(f"No 'Adj Close' or 'Close' column found for {t}")
        data[t] = series
    if not data:
        raise RuntimeError("No ETF data downloaded. Check tickers and network.")
    
    # Concatenate dict of Series; columns will be the tickers
    etf_data = pd.concat(data.values(), axis=1)
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




def fetch_etf_fundamentals(tickers: List[str], logger) -> pd.DataFrame:
    """Fetch basic ETF-level fundamentals (best-effort) for sector ETFs.

    Args:
        tickers (List[str]): List of ETF tickers.
        latest_prices (pd.Series): Latest adjusted close prices for the ETFs.
        logger (Logger): Logger for status messages.

    Returns:
        pd.DataFrame: DataFrame containing the fetched ETF fundamentals.
    """
    import yfinance as yf

    # Fetch fundamentals for each ticker
    rows = []
    logger.info(f"Fetching ETF fundamentals for {len(tickers)} tickers...")
    for t in tickers:
        tk = yf.Ticker(t)
        info = tk.info or {}
        if not info:
            raise RuntimeError(f"No fundamental data fetched for ticker {t}. Check ticker and network.")
        rows.append({
            "ticker": t,
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "dividend_yield": info.get("yield") 
        })
    return pd.DataFrame(rows).set_index("ticker")





def main(config_path: str, data_dir: str):
    """Main function to load and process data.

    Args:
        config_path (str): Path to the configuration file.
        data_dir (str): Path to the data directory.
    """
    # Initialize logger
    logger = get_logger("data_loader")
    logger.info("Starting data loading process...")

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
    logger.info(f"Configuration loaded. ")
    
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

    save_to_sqlite(db_path, "etf_data", "date", etf_data, logger)

    # Save calendar index from etf data to csv and the database (for later time alignment)
    cal = pd.DataFrame({"trading_day": 1}, index=etf_data.index.copy())
    cal.index.name = "date"
    
    out_path = raw_data_dir / "calendar.csv"
    cal.to_csv(out_path)
    logger.info(f"Saved trading calendar data csv at: {out_path}")

    save_to_sqlite(db_path, "calendar", "date", cal, logger)

    # Fetch and store basic ETF-level fundamentals
    latest_prices = etf_data.iloc[-1]
    fundamentals = fetch_etf_fundamentals(all_etf_tickers, logger)
    
    out_path = raw_data_dir / "etf_fundamentals.csv"
    fundamentals.to_csv(out_path)
    logger.info(f"Saved ETF fundamentals csv at: {out_path}")

    save_to_sqlite(db_path, "etf_fundamentals", "ticker", fundamentals, logger)

    # Fetch and store FRED data into csv and the database
    fred_data = fetch_fred_data(fred_series, logger)
    
    for s, df in fred_data.items():
        out_path = raw_data_dir / f"fred_{s}.csv"
        df.to_csv(out_path)
        logger.info(f"Saved FRED series {s} data csv at: {out_path}")
        save_to_sqlite(db_path, f"fred_{s}", "date", df, logger)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Loader for ETFs (Exchange-Traded Funds) and FRED (Federal Reserve Economic Data)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the data directory")
    args = parser.parse_args()
    main(args.config, args.data_dir)
    

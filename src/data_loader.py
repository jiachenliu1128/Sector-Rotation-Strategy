import argparse
import argparse
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import yaml
from dotenv import load_dotenv
import sqlite3





def main(config_path: str, data_dir: str):
    # Load environment variables
    load_dotenv()

    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    start = config["date_range"]["start"]
    end = config["date_range"]["end"]
    tickers = config["etf_tickers"]
    benchmark = config.get("etf_benchmark", "SPY")
    fred_series = config.get("fred", {}).get("series", {})
    
    # Define database directory
    data_dir = Path(data_dir)
    raw_data_dir = data_dir / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "data.db"
    






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Loader for ETFs (Exchange-Traded Funds) and FRED (Federal Reserve Economic Data)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the data directory")
    args = parser.parse_args()
    main(args.config, args.data_dir)
    

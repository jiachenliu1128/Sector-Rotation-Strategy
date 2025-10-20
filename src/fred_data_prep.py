# src/fred_data_prep.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from utils import get_logger





def load_fred_from_sqlite(db_path: str) -> dict[str, pd.DataFrame]:
    """Load all tables whose name starts with 'fred_' from a SQLite DB.
    
    Args:
        db_path (str): Path to SQLite DB file.
    Returns a dict: {table_name: DataFrame(index=date, columns=['value'])}
    """
    import sqlite3
    
    # Get list of fred_* tables
    out = {}
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'fred_%' ORDER BY name;"
    
    # Load each table
    with sqlite3.connect(db_path) as conn:
        tables = [r[0] for r in conn.execute(q).fetchall()]
        for t in tables:
            # Load table into DataFrame
            df = pd.read_sql(f"SELECT * FROM {t}", conn, parse_dates=["date"])
            if "date" not in df or "value" not in df:
                continue
            df = df.set_index("date").sort_index()
            # Coerce to numeric
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            out[t] = df
    return out





def load_fred_from_csv_dir(csv_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Fallback: read fred_*.csv from a directory.
    
    Args:
        csv_dir (str | Path): Directory containing fred_*.csv files.

    Returns:
        dict: {file_stem: DataFrame(index=date, columns=['value'])}
    """
    out = {}
    csv_dir = Path(csv_dir)
    
    for fp in sorted(csv_dir.glob("fred_*.csv")):
        try:
            # Load CSV
            df = pd.read_csv(fp, parse_dates=["date"]) 
            if "date" not in df or "value" not in df:
                continue
            df = df.set_index("date").sort_index()
            # Coerce to numeric
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            out[fp.stem] = df
        except Exception:
            continue
        
    return out





def to_macro_monthly(series_dict: dict[str, pd.DataFrame], logger) -> pd.DataFrame:
    """Resample each FRED series to month-end and merge into a single DF.

    Args:
        series_dict (dict): {name: DataFrame(index=date, columns=['value'])}
        logger: Logger instance for logging warnings/errors.
        
    Returns:
        pd.DataFrame: Merged monthly DataFrame with 'date' as index.
    """
    monthly_frames = []
    
    for name, df in series_dict.items():
        if df.empty:
            continue
        s = df["value"].copy()
        # Ensure datetime index and monotonic
        s = s.sort_index()
        # Month-end sample then FFill for gaps 
        s_m = s.resample("ME").last().ffill()
        s_m.name = name  
        monthly_frames.append(s_m.to_frame())
        
    if not monthly_frames:
        raise ValueError("No FRED series found to resample.")
    
    # Combine all DataFrames
    macro_monthly = pd.concat(monthly_frames, axis=1).sort_index()
    macro_monthly.index.name = "date"
    # Drop all-NaN columns (if any)
    macro_monthly = macro_monthly.dropna(axis=1, how="all")
    # Warn about missing values
    na_cols = macro_monthly.columns[macro_monthly.isna().any()]
    if len(na_cols) > 0:
        logger.warning("Some monthly macro series contain NaNs after resampling: %s", list(na_cols)[:10])
    return macro_monthly






def add_macro_transforms(macro_monthly: pd.DataFrame, logger) -> pd.DataFrame:
    """Add common transformed macro features if source columns exist.

    - CPI YoY from CPIAUCSL, YoY stands for 12-month % change
    - GDP YoY from GDP
    - 10y-FF spread from DGS10 and FEDFUNDS
    - FEDFUNDS monthly change (Δ1m)
    
    Args:
        macro_monthly (pd.DataFrame): Monthly macro DataFrame.
        logger: Logger instance for logging warnings/errors.
        
    Returns:
        pd.DataFrame: DataFrame with added transformed columns.
    """
    out = macro_monthly.copy()

    # CPI YoY
    if "fred_CPIAUCSL" in out.columns:
        out["CPI_yoy"] = out["fred_CPIAUCSL"].pct_change(12, fill_method=None)
    else:
        breakpoint()

    # GDP YoY (usually quarterly, but after ffill we can compute YoY on monthly frame)
    if "fred_GDP" in out.columns:
        out["GDP_yoy"] = out["fred_GDP"].pct_change(12, fill_method=None)
    else:
        breakpoint()

    # 10y - Fed Funds spread
    if "fred_DGS10" in out.columns and "fred_FEDFUNDS" in out.columns:
        out["TermSpread"] = out["fred_DGS10"] - out["fred_FEDFUNDS"]
        out["FEDFUNDS_chg1m"] = out["fred_FEDFUNDS"].diff(1)
    else:
        breakpoint()

    return out




def add_regime_tags(macro_monthly: pd.DataFrame, gdp_col="GDP_yoy", cpi_col="CPI_yoy",
                    inflation_thresh=0.03) -> pd.DataFrame:
    """
    Add simple macro regime classification based on GDP and CPI YoY.

    Args:
        macro_df: DataFrame with GDP_yoy and CPI_yoy columns.
        inflation_thresh: Threshold (e.g. 0.03 = 3%) for high vs low inflation.

    Returns:
        DataFrame with new columns: growth_regime, inflation_regime, regime_tag
    """
    out = macro_monthly.copy()

    # Define sub-regimes
    out["growth_regime"] = np.where(out[gdp_col] > 0, "Expansion", "Recession")
    out["inflation_regime"] = np.where(out[cpi_col] > inflation_thresh, "HighInfl", "LowInfl")

    # Combine
    out["regime_tag"] = out["growth_regime"] + "_" + out["inflation_regime"]

    # One-hot encode for modeling
    regime_dummies = pd.get_dummies(out["regime_tag"], prefix="regime")
    out = pd.concat([out, regime_dummies], axis=1)

    # Remove intermediate columns
    out = out.drop(columns=["growth_regime", "inflation_regime", "regime_tag"])

    return out






def save_macro(df: pd.DataFrame, out_csv: str, out_db: str, logger) -> None:
    """Save the macro monthly DataFrame to CSV and/or SQLite.
    
    Args:
        out_df (pd.DataFrame): Monthly macro DataFrame.
        out_csv (str): Path to save cleaned monthly macro features CSV.
        out_db (str): Path to SQLite DB to save 'macro_monthly' table.
    """
    if out_db:
        import sqlite3
        p = Path(out_db)
        
        with sqlite3.connect(p) as conn:
            df.to_sql("macro_monthly", conn, if_exists="replace", index=True, index_label="date")
        logger.info(f"Saved macro monthly to SQLite DB at: {p}")
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=True)
        logger.info(f"Saved macro monthly CSV to: {out_csv}")
   







def main(db_path: str, csv_path: str, out_csv: str, out_db: str) -> None:
    """Main function to load, clean, resample FRED data and save to CSV/SQLite.

    Args:
        db_path (str): path to SQLite DB with fred_* tables.
        csv_path (str): path to CSV of fred_*.csv files if SQLite not available.
        out_csv (str): path to save cleaned monthly macro features CSV.
        out_db (str): path to SQLite DB to write 'macro_monthly'. Use empty to skip.
    """
    # Load Logger
    logger = get_logger("fred_data_prep")

    # Load FRED series
    try:
        fred = load_fred_from_sqlite(db_path)
        logger.info(f"Loaded {len(fred)} FRED tables from SQLite Database at: {db_path}")
    except Exception as e:
        logger.warning(f"SQLite load failed ({e}); trying CSV dir: {csv_path}")
        fred = load_fred_from_csv_dir(csv_path)
        logger.info(f"Loaded {len(fred)} FRED CSV files from: {csv_path}")

    try:
        # Resample to monthly, merge and transform
        macro_monthly = to_macro_monthly(fred, logger)
        macro_monthly = add_macro_transforms(macro_monthly, logger)

        # Add regime tags
        macro_monthly = add_regime_tags(macro_monthly)
        logger.info(f"Prepared macro monthly DataFrame with shape: {macro_monthly.shape}")
        
    except Exception as e:
        logger.error(f"Error processing FRED data: {e}")
        return

    # 4) Save
    out_db = out_db if out_db.strip() else None
    save_macro(macro_monthly, out_csv, out_db, logger)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 2: Prepare monthly macro features from FRED series.")
    parser.add_argument("--db", type=str, default="data/data.db", help="SQLite DB containing fred_* tables.")
    parser.add_argument("--csv_dir", type=str, default="data/raw", help="Fallback directory with fred_*.csv files.")
    parser.add_argument("--out_csv", type=str, default="data/processed/macro_monthly.csv", help="Output CSV path.")
    parser.add_argument("--out_db", type=str, default="data/data.db", help="SQLite DB to write 'macro_monthly'. Use empty to skip.")
    args = parser.parse_args()
    
    main(args.db, args.csv_dir, args.out_csv, args.out_db)

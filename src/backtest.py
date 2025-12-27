from __future__ import annotations
import argparse
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple
from utils import get_logger

import numpy as np
import pandas as pd





def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Check if a table exists in the SQLite database.

    Args:
        conn (sqlite3.Connection): Connection to the SQLite database.
        table (str): Name of the table to check.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,)
    )
    return cur.fetchone() is not None






def load_predictions(
    db: str,
    task: str,
    model: str,
    csv_dir: str,
    table_override: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load predictions_{task}_{model} from SQLite (preferred) or CSV fallback.
    Expected schemas:

    Regression:    [date, ticker, y_true, y_pred, fold]
    Classification:[date, ticker, y_true_cls, y_prob, y_pred_cls, fold]
    
    Works with classic models (ridge/logit/rf/xgb) and sequence models (lstm/transformer)
    
    Args:
        db: path to SQLite database
        task: 'regression' or 'classification'
        model: model name (e.g., 'ridge', 'rf', 'xgb')
        csv_dir: directory containing CSV fallback files
        table_override: if provided, use this table name instead of default
        
    Returns:
        DataFrame with predictions
    """
    # Determine table name and CSV path
    table = table_override or f"predictions_{task}_{model}"
    csv_path = Path(csv_dir) / f"predictions_{task}_{model}.csv"

    # Try DB
    try:
        with sqlite3.connect(db) as conn:
            if table_exists(conn, table):
                df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["date"])
                return df
            logger.warning(f"Table '{table}' does not exist in DB. Falling back to CSV.")
    except Exception as e:
        logger.warning(f"Failed to read table '{table}' from DB: {e}")

    # Fallback CSV
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"])
    else:
        logger.error(f"Could not find predictions in DB table '{table}' or CSV '{csv_path}'.")
        raise FileNotFoundError(
            f"Could not find predictions in DB table '{table}' or CSV '{csv_path}'."
        )





def load_returns_long(
    db: str,
    table: str,
    csv_path: Optional[str] = None,
    date_col: str = "date",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load realized returns in WIDE form and convert to LONG:
      - Index/column names may be like y_XLF, y_XLK, ... (we'll strip 'y_' to get tickers)
      - Values should be daily returns (absolute or excess), not prices.

    Output columns: [date, ticker, ret]
    
    Args:
        db: path to SQLite database
        table: table name containing returns in WIDE form
        csv_path: optional CSV fallback path
        date_col: name of the date column in the WIDE table 
        
    Returns:
        DataFrame in LONG form with columns [date, ticker, ret]
    """
    # Try DB
    df = None
    try:
        with sqlite3.connect(db) as conn:
            if table_exists(conn, table):
                df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=[date_col])
            logger.info(f"Loaded returns table '{table}' from DB. Falling back to CSV if needed.")
    except Exception as e:
        logger.warning(f"Failed to read table '{table}' from DB: {e}")
        df = None

    # Fallback CSV
    if df is None:
        if not csv_path:
            raise FileNotFoundError(
                f"Missing returns source: table '{table}' not found and no csv_path provided."
            )
        p = Path(csv_path)
        if not p.exists():
            logger.error(f"Returns CSV not found at {p}")
            raise FileNotFoundError(f"Returns CSV not found at {p}")
        df = pd.read_csv(p, parse_dates=[date_col])
        logger.info(f"Loaded returns from CSV at {p}")

    # Make sure date sorted and set index
    df = df.sort_values(date_col).set_index(date_col)

    # Normalize column names -> tickers
    new_cols = []
    for c in df.columns:
        if c.startswith("y_"):
            new_cols.append(c[2:])
        else:
            new_cols.append(c)
    df.columns = new_cols

    # To long
    long_df = df.reset_index().melt(id_vars=[date_col], var_name="ticker", value_name="ret")
    long_df = long_df.rename(columns={date_col: "date"}).dropna(subset=["ret"])
    return long_df




def topk_weights_equal(scores: pd.Series, k: int) -> pd.Series:
    """Equal-weight the top-k by score (descending).
    
    Args:
        scores: Series indexed by ticker with scores
        k: number of top items to select
        
    Returns:
        Series of weights indexed by ticker
    """
    top = scores.sort_values(ascending=False).head(k)
    if top.empty:
        return pd.Series(dtype=float)
    w = pd.Series(0.0, index=scores.index)
    w.loc[top.index] = 1.0 / len(top)
    return w




def topk_long_short_weights_equal(scores: pd.Series, k: int) -> Tuple[pd.Series, pd.Series]:
    """Equal-weight top-k longs and bottom-k shorts.
    
    Args:
        scores: Series indexed by ticker with scores
        k: number of top/bottom items to select
        
    Returns:
        Tuple of (long_weights Series, short_weights Series) indexed by ticker
    """
    longs = scores.sort_values(ascending=False).head(k)
    shorts = scores.sort_values(ascending=True).head(k)
    w_long = pd.Series(0.0, index=scores.index)
    w_short = pd.Series(0.0, index=scores.index)
    if len(longs) > 0:
        w_long.loc[longs.index] = 1.0 / len(longs)
    if len(shorts) > 0:
        w_short.loc[shorts.index] = -1.0 / len(shorts)  
    return w_long, w_short




def equity_curve(returns: pd.Series) -> pd.Series:
    """Compute the equity curve from a series of returns.

    Args:
        returns (pd.Series): Series of returns indexed by time.

    Returns:
        pd.Series: Cumulative returns (equity curve) indexed by time.
    """
    return (1.0 + returns.fillna(0.0)).cumprod()




def max_drawdown(curve: pd.Series) -> float:
    """Compute the maximum drawdown of an equity curve.

    Args:
        curve (pd.Series): Series representing the equity curve indexed by time.

    Returns:
        float: Maximum drawdown as a percentage.
    """
    peak = curve.cummax()
    dd = (curve / peak) - 1.0
    return float(dd.min())





def annualized_stats(rets: pd.Series) -> Tuple[float, float, float]:
    """Return (CAGR, vol_ann, Sharpe) assuming daily returns and rf=0.
    
    Args:
        rets: Series of daily returns
    
    Returns:
        Tuple of (CAGR, vol_ann, Sharpe)
    """
    # Clean returns
    rets = rets.dropna()
    if rets.empty:
        return np.nan, np.nan, np.nan
    
    # Stats
    n = len(rets)
    cagr = (1.0 + rets).prod() ** (12.0 / n) - 1.0
    vol_ann = rets.std(ddof=1) * np.sqrt(12.0)
    sharpe = (rets.mean() * 12.0) / vol_ann if vol_ann > 0 else np.nan
    return float(cagr), float(vol_ann), float(sharpe)





def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    """Turnover = 0.5 * L1 change in weights.
    
    Args:
        prev_w: previous weights Series indexed by ticker
        new_w: new weights Series indexed by ticker
        
    Returns:
        turnover float
    """
    idx = prev_w.index.union(new_w.index)
    prev = prev_w.reindex(idx, fill_value=0.0)
    new = new_w.reindex(idx, fill_value=0.0)
    return 0.5 * float((prev - new).abs().sum())






def backtest(
    preds: pd.DataFrame,
    rets_long: Optional[pd.DataFrame],
    task: str,
    top_k: int = 3,
    long_short: bool = False,
    use_ytrue_if_no_returns: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build a daily sector-rotation portfolio based on predictions.

    Parameters
    ----------
    preds : predictions DataFrame (see load_predictions)
    rets_long : realized returns in long form [date, ticker, ret] (absolute or excess)
    task : 'regression' or 'classification'
    top_k : number of sectors to long (and short, if long_short=True)
    long_short : if True, build a market-neutral long-short portfolio
    use_ytrue_if_no_returns : if True and rets_long is None, use preds['y_true'] as realized ret (regression only)

    Returns
    -------
    port : DataFrame [date, port_ret, bench_ret(optional), long_count, short_count, turnover]
    weights : DataFrame of weights by (date, ticker)
    metrics : DataFrame with summary statistics
    """
    # Scores
    if task == "regression":
        if "y_pred" not in preds.columns:
            raise ValueError("Regression predictions require column 'y_pred'.")
        preds["score"] = preds["y_pred"]
    else:
        # Prefer probability if available; otherwise allow class label fallback.
        if "y_prob" in preds.columns:
            preds["score"] = preds["y_prob"]
        elif "y_pred_cls" in preds.columns:
            if logger:
                logger.warning("No 'y_prob' found; using 'y_pred_cls' as score for ranking.")
            preds["score"] = preds["y_pred_cls"].astype(float)
        elif "score" in preds.columns:
            # If upstream already provided a 'score' column, use it.
            preds["score"] = preds["score"].astype(float)
        else:
            raise ValueError("Classification predictions require 'y_prob' (preferred) or 'y_pred_cls'/'score'.")

    # Realized returns join
    if rets_long is None:
        if task != "regression" or (not use_ytrue_if_no_returns):
            raise ValueError(
                "Returns not provided. For classification (or if use_ytrue_if_no_returns=False) "
                "you must supply realized returns."
            )
        # Use y_true as realized (e.g., excess returns target)
        if "y_true" not in preds.columns:
            raise ValueError("Missing 'y_true' to be used as realized returns.")
        realized = preds[["date", "ticker", "y_true"]].rename(columns={"y_true": "ret"})
    else:
        realized = rets_long.copy()
        
    # Check duplicates of (date,ticker) before merge (There is overlap during walk-forward)
    if preds.duplicated(['date','ticker']).any():
        raise ValueError("Predictions have duplicate (date,ticker). Walk-forward splits have overlapping dates.")
    if realized.duplicated(['date','ticker']).any():
        raise ValueError("Realized returns have duplicate (date,ticker). Walk-forward splits have overlapping dates.")

    # Merge scores with realized returns
    df = preds[["date", "ticker", "score"]].merge(
        realized, on=["date", "ticker"], how="inner"
    ).sort_values(["date", "ticker"])

    # Iterate by day
    port_rows = []
    weights_rows = []
    prev_w = pd.Series(dtype=float)

    # Daily loop
    for dt, g in df.groupby("date"):
        # Extract scores and returns
        scores = g.set_index("ticker")["score"]
        rets = g.set_index("ticker")["ret"]

        if not long_short:
            # Long-only case
            # weights
            w = topk_weights_equal(scores, top_k) 
            # portfolio return: sum of (weights * returns)
            port_ret = float((w * rets.reindex(w.index)).sum())
            # turnover
            t_over = turnover(prev_w, w)
            
            # update prev_w and counts
            prev_w = w
            long_count = int((w > 0).sum())
            short_count = 0
            
            # store portfolio row
            port_rows.append(
                {"date": dt, "port_ret": port_ret, "turnover": t_over,
                 "long_count": long_count, "short_count": short_count}
            )
            
            # store weights
            wr = pd.DataFrame({"date": dt, "ticker": w.index, "weight": w.values})
            weights_rows.append(wr)

        else:
            # Long-short case
            # weights
            w_long, w_short = topk_long_short_weights_equal(scores, top_k)
            w_combined = (w_long + w_short).sort_index()
            # portfolio return: sum of (weights * returns)
            port_ret = float((w_combined * rets.reindex(w_combined.index)).sum())
            # turnover
            t_over = turnover(prev_w, w_combined)
            
            # update prev_w and counts
            prev_w = w_combined
            long_count = int((w_long > 0).sum())
            short_count = int((w_short < 0).sum())
            
            # store portfolio row
            port_rows.append(
                {"date": dt, "port_ret": port_ret, "turnover": t_over,
                 "long_count": long_count, "short_count": short_count}
            )
            
            # store weights
            wr = pd.DataFrame({"date": dt, "ticker": w_combined.index, "weight": w_combined.values})
            weights_rows.append(wr)

    # Compile results
    port = pd.DataFrame(port_rows).sort_values("date").reset_index(drop=True)
    weights = pd.concat(weights_rows, ignore_index=True) if weights_rows else pd.DataFrame()

    # Metrics
    curve = equity_curve(port["port_ret"])
    mdd = max_drawdown(curve)
    cagr, vol_ann, sharpe = annualized_stats(port["port_ret"])
    hit = float((port["port_ret"] > 0).mean())  # if excess returns -> hit vs 0; else interpret as >0 days

    metrics = pd.DataFrame(
        [{
            "n_days": len(port),
            "CAGR": cagr,
            "Vol_ann": vol_ann,
            "Sharpe": sharpe,
            "MaxDD": mdd,
            "Hit_ratio": hit,
            "Turnover_avg": float(port["turnover"].mean() if not port.empty else np.nan),
            "top_k": top_k,
            "long_short": long_short,
        }]
    )
    
    logger.info(f"Backtest completed: {len(port)} days, CAGR={cagr:.2%}, Sharpe={sharpe:.2f}, MaxDD={mdd:.2%}")
    return port, weights, metrics







def main(db: str, csv_dir: str, task: str, model: str, preds_table: str,
         returns_table: str, returns_csv: str,
         top_k: int, long_short: bool, 
         out_dir: str, save_db: bool):
    """Main function to run backtest function.

    Args:
        db (str): Path to SQLite database
        csv_dir (str): Path to csv directory
        task (str): Task type ('regression' or 'classification')
        model (str): Model type ('linreg', 'ridge', 'logit', 'rf', 'xgb')
        preds_table (str): predictions table name override
        returns_table (str): realized returns table name
        returns_csv (str): _description_
        top_k (int): _description_
        long_short (bool): _description_
        no_use_ytrue_fallback (bool): _description_
        out_dir (str): _description_
        save_db (bool): _description_
    """
    # Logging
    logger = get_logger("backtest")
    logger.info("Starting backtest...")

    # Load predictions
    preds = load_predictions(db, task, model, csv_dir, table_override=preds_table, logger=logger)

    # Load realized returns (if present)
    rets_long = None
    try:
        rets_long = load_returns_long(db, returns_table, returns_csv, logger=logger)
    except Exception:
        # OK: we may fall back to y_true if regression
        rets_long = None

    # Backtest
    port, weights, metrics = backtest(
        preds=preds,
        rets_long=rets_long,
        task=task,
        top_k=top_k,
        long_short=long_short,
        logger=logger,
    )

    # Save outputs
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{task}_{model}{'_ls' if long_short else ''}_top{top_k}"
    port_path = out_dir / f"portfolio_{suffix}.csv"
    w_path = out_dir / f"weights_{suffix}.csv"
    m_path = out_dir / f"metrics_{suffix}.csv"

    port.to_csv(port_path, index=False)
    weights.to_csv(w_path, index=False)
    metrics.to_csv(m_path, index=False)
    print(f"Saved portfolio returns -> {port_path}")
    print(f"Saved weights           -> {w_path}")
    print(f"Saved metrics           -> {m_path}")
    logger.info("Backtest outputs saved.")

    if save_db:
        with sqlite3.connect(db) as conn:
            port.to_sql(f"portfolio_{suffix}", conn, if_exists="replace", index=False)
            weights.to_sql(f"weights_{suffix}", conn, if_exists="replace", index=False)
            metrics.to_sql(f"portfolio_metrics_{suffix}", conn, if_exists="replace", index=False)
        print(f"Also wrote to SQLite tables: portfolio_{suffix}, weights_{suffix}, portfolio_metrics_{suffix}")
        logger.info("Backtest outputs saved to SQLite database.")







if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sector rotation backtest from model predictions")
    # Inputs data paths
    p.add_argument("--db", default="data/data.db")
    p.add_argument("--csv_dir", default="data/predictions")
    
    # Task: regression or classification
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    
    # Models: regression -> ridge|rf|xgb ; classification -> logit|rf|xgb
    p.add_argument("--model", choices=["ridge", "logit", "rf", "xgb", "lstm", "transformer"], default="ridge")
    
    # Predictions source that override default table name
    p.add_argument("--preds_table", default=None, help="Override predictions table name")
    
    # Realized returns source (WIDE table -> long). Defaults to y_targets (often excess vs SPY).
    p.add_argument("--returns_table", default="y_targets")
    p.add_argument("--returns_csv", default=None, help="Fallback CSV if returns table missing")

    # Strategy
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--long_short", action="store_true")

    # Outputs
    p.add_argument("--out_dir", default="data/backtest")
    p.add_argument("--save_db", action="store_true")

    args = p.parse_args()

    main(args.db, args.csv_dir, args.task, args.model, args.preds_table,
         args.returns_table, args.returns_csv,
         args.top_k, args.long_short,
         args.out_dir, args.save_db)
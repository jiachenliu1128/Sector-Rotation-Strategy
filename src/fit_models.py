import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from xgboost import XGBRegressor, XGBClassifier

from src.walkforward import WFCV, WFConfig



def read_table(db: str, table: str, csv_fallback: str | None = None) -> pd.DataFrame:
    """Load a table from SQLite, fallback to CSV if provided.
    
    Args:
        db (str): Path to SQLite DB file.
        table (str): Name of the table to read.
        csv_fallback (str | None): Path to CSV file to fallback on.
        
    Returns:
        pd.DataFrame: DataFrame with date index.
    """
    # First try to read from SQLite
    try:
        with sqlite3.connect(db) as conn:
            df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["date"]).set_index("date").sort_index()
            return df
    
    except Exception:
        # Fallback to CSV if DB read fails
        if not csv_fallback:
            raise
        df = pd.read_csv(csv_fallback, parse_dates=["date"]).set_index("date").sort_index()
        return df
    
    
    
    

@dataclass
class RunConfig:
    """
    Configuration for model training and evaluation.
    """
    task: str = "regression"         
    model: str = "ridge"          
    ridge_alpha: float = 10.0
    rf_estimators: int = 300
    rf_depth: Optional[int] = None
    logit_C: float = 1.0
    train_size: int = 60
    test_size: int = 1
    step: int = 1
    expanding: bool = True
    embargo: int = 0
    direction_threshold: float = 0.0   





def get_model(task: str, kind: str, ridge_alpha: float = 10.0,
              rf_estimators: int = 300, rf_depth: Optional[int] = None,
              logit_C: float = 1.0, random_state: int = 42):
    """Get a model for the specified task and kind.

    Args:
        task (str): The task type ("regression" or "classification").
        kind (str): The model kind (e.g., "linreg", "rf", "xgb").
        ridge_alpha (float, optional): The alpha parameter for Ridge regression. Defaults to 10.0.
        rf_estimators (int, optional): The number of trees in the random forest. Defaults to 300.
        rf_depth (Optional[int], optional): The maximum depth of the trees. Defaults to None.
        logit_C (float, optional): The regularization strength for logistic regression. Defaults to 1.0.
        random_state (int, optional): The random seed. Defaults to 42.

    Raises:
        ImportError: If a required library is not installed.
        ValueError: If an unknown model kind is specified.
        ImportError: If a required library is not installed.
        ValueError: If an unknown model kind is specified.
        ValueError: If the task is not "regression" or "classification".

    Returns:
        An instance of the specified model.
    """
    task = task.lower()
    kind = kind.lower()

    
    if task == "regression":
        if kind == "linreg":
            return LinearRegression()
        if kind == "ridge":
            return Ridge(alpha=ridge_alpha, random_state=random_state)
        if kind == "rf":
            return RandomForestRegressor(n_estimators=rf_estimators, max_depth=rf_depth,
                                         random_state=random_state, n_jobs=-1)
        if kind == "xgb":
            return XGBRegressor(objective="reg:squarederror", n_estimators=rf_estimators,
                                 max_depth=rf_depth if rf_depth is not None else 6,
                                 random_state=random_state, n_jobs=-1, tree_method="hist")
        raise ValueError(f"Unknown regression model kind: {kind}, please choose from linreg, ridge, rf, xgb.")

    elif task == "classification":
        if kind == "logit":
            return LogisticRegression(max_iter=200, C=logit_C, n_jobs=-1)
        if kind == "rf":
            return RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_depth,
                                         random_state=random_state, n_jobs=-1)
        if kind == "xgb":
            return XGBClassifier(objective="binary:logistic", n_estimators=rf_estimators,
                                 max_depth=rf_depth if rf_depth is not None else 6,
                                 random_state=random_state, n_jobs=-1, tree_method="hist", eval_metric="logloss")
        raise ValueError(f"Unknown classification model kind: {kind}, please choose from logit, rf, xgb.")

    else:
        raise ValueError("task must be 'regression' or 'classification'")
    
    
    
    
    
    


def extract_feature_importance(model, X_train: pd.DataFrame, task: str, model_kind: str) -> pd.Series:
    """
    Return a pandas Series of feature importances indexed by X_train.columns.

    For linear models: absolute value of coefficients (magnitude only).
    For RandomForest: model.feature_importances_.
    For XGBoost: booster.get_score(importance_type='gain'), mapped to column names.
    If importance is unavailable, returns an empty Series.
    
    Args:
        model: Trained model instance.
        X_train (pd.DataFrame): Training features DataFrame.
        task (str): Task type ("regression" or "classification").
        model_kind (str): Model kind (e.g., "linreg", "rf", "xgb").
        
    Returns:
        pd.Series: Feature importances indexed by feature names.
    """
    kind = model_kind.lower()
    cols = list(X_train.columns)

    # Linear / Ridge / Logistic (scikit-learn)
    if hasattr(model, "coef_") and model.coef_ is not None:
        coef = np.ravel(model.coef_)
        if len(coef) == len(cols):
            return pd.Series(np.abs(coef), index=cols, name="coef_abs")

    # Random Forest
    if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
        imp = np.array(model.feature_importances_)
        if len(imp) == len(cols):
            return pd.Series(imp, index=cols, name="rf_importance")

    # XGBoost (sklearn wrapper)
    if kind == "xgb" and hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            raw = booster.get_score(importance_type="gain")  
            feat_names = booster.feature_names
            if feat_names is None:
                # Map f{i} -> pandas columns
                mapped = {}
                for k, v in raw.items():
                    if k.startswith("f") and k[1:].isdigit():
                        i = int(k[1:])
                        if 0 <= i < len(cols):
                            mapped[cols[i]] = v
                return pd.Series(mapped, name="xgb_gain")
            else:
                # keep only columns present in X
                ser = pd.Series(raw, dtype=float, name="xgb_gain")
                ser = ser.reindex(cols, fill_value=0.0)
                return ser
        except Exception:
            pass

    # Fallback: no importances
    return pd.Series(dtype=float)






def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        Dict[str, float]: Computed regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Hit ratio: sign correctness (ignore zeros)
    signs_true = np.sign(y_true)
    signs_pred = np.sign(y_pred)
    mask = signs_true != 0
    hit = float(np.mean(signs_true[mask] == signs_pred[mask])) if mask.any() else np.nan
    
    # Information Coefficient: correlation
    ic = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    
    return {"mae": mae, "rmse": rmse, "r2": r2, "hit": hit, "ic": ic}






def classification_metrics(y_true_cls: np.ndarray, y_prob: Optional[np.ndarray], y_pred_cls: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true_cls, y_pred_cls)
    prec = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    rec = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    # AUROC only if both classes present and probs available
    try:
        if y_prob is not None and len(np.unique(y_true_cls)) == 2:
            auc = roc_auc_score(y_true_cls, y_prob)
        else:
            auc = np.nan
    except Exception:
        auc = np.nan
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}








def run_walkforward(X: pd.DataFrame, y: pd.DataFrame, cfg: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run walk-forward Cross Validation and model training.
    
    Args:
        X (pd.DataFrame): Features DataFrame with date index.
        y (pd.DataFrame): Targets DataFrame with date index.
        cfg (RunConfig): Configuration for the run.

    Returns:
        predictions_long : DataFrame
            Regression: [date, ticker, y_true, y_pred, fold]
            Classification: [date, ticker, y_true_cls, y_prob, y_pred_cls, fold]
        metrics_long     : DataFrame with per-fold, per-ticker metrics
    """
    # Align indices
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    # Remove "y_" prefix to get tickers
    tickers = [c.replace("y_", "") for c in y.columns]

    # Walk-forward CV splitter
    splitter = WFCV(WFConfig(
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        step=cfg.step,
        expanding=cfg.expanding,
        embargo=cfg.embargo,
        verbose=True,  
    ))

    # Store predictions and metrics
    preds_records: List[Dict] = []
    metrics_records: List[Dict] = []
    feature_importance_records: List[Dict] = []

    # For each fold and each ticker, train model and predict
    total_folds = len(list(splitter.split(X)))
    print(f"Starting walk-forward CV with {total_folds} folds and {len(tickers)} tickers...")
    
    for fold_no, (tr_idx, te_idx) in enumerate(splitter.split(X), start=1):
        print(f"\nFold {fold_no}/{total_folds} - Training period: {X.index[tr_idx[0]].strftime('%Y-%m')} to {X.index[tr_idx[-1]].strftime('%Y-%m')}, "
              f"Test period: {X.index[te_idx[0]].strftime('%Y-%m')} to {X.index[te_idx[-1]].strftime('%Y-%m')}")
        
        # Get values and dates for this fold by index
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        dates_te = X_te.index

        # Scale features on train-only statistics
        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_te_s = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)

        # For each ticker/column in y
        for ticker_idx, (col, ticker) in enumerate(zip(y.columns, tickers), start=1):
            print(f"  Processing ticker {ticker_idx}/{len(tickers)}: {ticker}", end=" ... ")
            y_tr_raw = y.iloc[tr_idx][col].values
            y_te_raw = y.iloc[te_idx][col].values

            # Build model
            model = get_model(cfg.task, cfg.model, cfg.ridge_alpha, cfg.rf_estimators, cfg.rf_depth, cfg.logit_C)

            ####################################################################
            # Regression
            ####################################################################
            if cfg.task == "regression":
                # Drop NaNs in train, fit the model and predict
                mtr = np.isfinite(y_tr_raw)
                if mtr.sum() < 10:
                    print("insufficient data")
                    continue
                model.fit(X_tr_s[mtr], y_tr_raw[mtr])
                y_pred = model.predict(X_te_s)
                print("done")

                # Store predictions
                for d, yt, yp in zip(dates_te.values, y_te_raw, y_pred):
                    preds_records.append({
                        "date": pd.Timestamp(d),
                        "ticker": ticker,
                        "y_true": float(yt) if np.isfinite(yt) else np.nan,
                        "y_pred": float(yp),
                        "fold": fold_no,
                    })
                    
                # Store metrics (only where y_te_raw is finite)
                mask = np.isfinite(y_te_raw)
                if mask.any():
                    m = regression_metrics(y_te_raw[mask], y_pred[mask])
                    m.update({"fold": fold_no, "ticker": ticker})
                    metrics_records.append(m)
                    
                # Store feature importances
                fi = extract_feature_importance(model, X_tr_s[mtr], cfg.task, cfg.model)
                if fi is not None and len(fi) > 0:
                    for feat, val in fi.items():
                        feature_importance_records.append({
                            "fold": fold_no,
                            "ticker": ticker,
                            "feature": str(feat),
                            "importance": float(val),
                        })

            ####################################################################
            # Classification
            ####################################################################
            else:  
                # Create binary labels using threshold
                y_tr_cls = (y_tr_raw > cfg.direction_threshold).astype(int)
                y_te_cls = (y_te_raw > cfg.direction_threshold).astype(int)

                # Drop NaNs in train and fit the model
                mtr = np.isfinite(y_tr_raw)
                if mtr.sum() < 10:
                    print("insufficient data")
                    continue
                model.fit(X_tr_s[mtr], y_tr_cls[mtr])
                print("done")

                # Get predicted probabilities and classes
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_te_s)[:, 1]
                else:
                    # If no predict_proba, try decision_function and convert to probs via sigmoid
                    if hasattr(model, "decision_function"):
                        z = model.decision_function(X_te_s)
                        y_prob = 1.0 / (1.0 + np.exp(-z))
                    else:
                        breakpoint()
                y_pred_cls = model.predict(X_te_s)

                # Store predictions
                for i, d in enumerate(dates_te.values):
                    preds_records.append({
                        "date": pd.Timestamp(d),
                        "ticker": ticker,
                        "y_true_cls": int(y_te_cls[i]) if np.isfinite(y_te_raw[i]) else np.nan,
                        "y_prob": float(y_prob[i]) if y_prob is not None else np.nan,
                        "y_pred_cls": int(y_pred_cls[i]),
                        "fold": fold_no,
                    })

                # Store metrics (only where y_te_raw is finite)
                mte = np.isfinite(y_te_raw)
                if mte.any():
                    yt = y_te_cls[mte]
                    yp = y_pred_cls[mte]
                    pr = y_prob[mte] if y_prob is not None else None
                    m = classification_metrics(yt, pr, yp)
                    m.update({"fold": fold_no, "ticker": ticker})
                    metrics_records.append(m)
                    
                # Store feature importances
                fi = extract_feature_importance(model, X_tr_s[mtr], cfg.task, cfg.model)
                if fi is not None and len(fi) > 0:
                    for feat, val in fi.items():
                        feature_importance_records.append({
                            "fold": fold_no,
                            "ticker": ticker,
                            "feature": str(feat),
                            "importance": float(val),
                        })

    # Convert to DataFrames and return
    predictions_long = pd.DataFrame(preds_records).sort_values(["date", "ticker"]) if preds_records else pd.DataFrame()
    metrics_long = pd.DataFrame(metrics_records).sort_values(["fold", "ticker"]) if metrics_records else pd.DataFrame()
    
    print(f"\nWalk-forward CV completed!")
    print(f"Generated {len(predictions_long)} predictions across {total_folds} folds and {len(tickers)} tickers")
    
    return predictions_long, metrics_long






def main(db: str, X_table: str, y_table: str, X_csv: str, y_csv: str,
         task: str, model: str,
         ridge_alpha: float, rf_estimators: int, rf_depth: Optional[int], logit_C: float,
         train_size: int, test_size: int, step: int, expanding: bool, embargo: int,
         direction_threshold: float,
         out_dir: str, save_db: bool) -> None:
    """Main entry point for running walk-forward CV experiments.
    
    Args:
        db (str): Path to SQLite DB file.
        X_table (str): Name of the features table in the DB.
        y_table (str): Name of the targets table in the DB.
        X_csv (str): Path to CSV file to fallback on if DB read fails.
        y_csv (str): Path to CSV file to fallback on if DB read fails.
        task (str): 'regression' or 'classification'.
        model (str): Model type, see `get_model`.
        ridge_alpha (float): Ridge regression alpha.
        rf_estimators (int): Number of trees for Random Forest.
        rf_depth (int | None): Max depth for Random Forest.
        logit_C (float): Inverse regularization strength for Logistic Regression.
        train_size (int): Training window size in months.
        test_size (int): Test window size in months.
        step (int): Step size in months for rolling window.
        expanding (bool): Whether to use expanding window (True) or rolling (False).
        embargo (int): Embargo period in months to prevent leakage.
        direction_threshold (float): Threshold for classification labels.
        out_dir (str): Directory to save output CSV files.
        save_db (bool): Whether to also save predictions/metrics back to the SQLite DB.
    """
    # Load data
    X = read_table(db, X_table, csv_fallback=X_csv)
    y = read_table(db, y_table, csv_fallback=y_csv)

    # Create model config
    cfg = RunConfig(
        task=task,
        model=model,
        ridge_alpha=ridge_alpha,
        rf_estimators=rf_estimators,
        rf_depth=rf_depth,
        logit_C=logit_C,
        train_size=train_size,
        test_size=test_size,
        step=step,
        expanding=expanding,
        embargo=embargo,
        direction_threshold=direction_threshold,
    )

    # Run walk-forward Cross Validation and fit models
    preds, metrics = run_walkforward(X, y, cfg)

    # Save outputs
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{task}_{model}"
    preds_path = out_dir / f"predictions_{suffix}.csv"
    metrics_path = out_dir / f"metrics_{suffix}.csv"
    preds.to_csv(preds_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved predictions -> {preds_path}")
    print(f"Saved metrics     -> {metrics_path}")

    if save_db:
        with sqlite3.connect(db) as conn:
            preds.to_sql(f"predictions_{suffix}", conn, if_exists="replace", index=False)
            metrics.to_sql(f"metrics_{suffix}", conn, if_exists="replace", index=False)
        print(f"Also wrote to SQLite tables: predictions_{suffix}, metrics_{suffix}")





if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Baseline models with walk-forward CV (regression & classification)")
    args.add_argument("--db", default="data/data.db")
    args.add_argument("--X_table", default="X_features")
    args.add_argument("--y_table", default="y_targets")
    args.add_argument("--X_csv", default="data/processed/X_features.csv")
    args.add_argument("--y_csv", default="data/processed/y_targets.csv")

    # Task: regression or classification
    args.add_argument("--task", choices=["regression", "classification"], default="regression")

    # Models: regression -> linreg|ridge|rf|xgb ; classification -> logit|rf|xgb
    args.add_argument("--model", choices=["linreg", "ridge", "logit", "rf", "xgb"], default="ridge")

    # Hyperparams
    args.add_argument("--ridge_alpha", type=float, default=10.0)
    args.add_argument("--rf_estimators", type=int, default=300)
    args.add_argument("--rf_depth", type=int, default=None)
    args.add_argument("--logit_C", type=float, default=1.0)

    # Walk-forward
    args.add_argument("--train_size", type=int, default=60)
    args.add_argument("--test_size", type=int, default=1)
    args.add_argument("--step", type=int, default=1)
    args.add_argument("--expanding", action="store_true")
    args.add_argument("--rolling", dest="expanding", action="store_false")
    args.set_defaults(expanding=True)
    args.add_argument("--embargo", type=int, default=0)

    # Classification-specific
    args.add_argument("--direction_threshold", type=float, default=0.0, help="Label threshold: y > thr => class 1")

    # Output
    args.add_argument("--out_dir", default="data/processed")
    args.add_argument("--save_db", action="store_true", help="Also write predictions/metrics to SQLite")
    args = args.parse_args()

    main(args.db, args.X_table, args.y_table, args.X_csv, args.y_csv, 
         args.task, args.model,
         args.ridge_alpha, args.rf_estimators, args.rf_depth, args.logit_C,
         args.train_size, args.test_size, args.step, args.expanding, args.embargo, 
         args.direction_threshold, 
         args.out_dir, args.save_db)
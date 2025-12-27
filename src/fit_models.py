import argparse
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import cupy as cp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from xgboost import XGBRegressor, XGBClassifier

from walkforward import WFCV, WFConfig
from utils import get_logger










################################################################################
## Data loading
################################################################################

def read_table(db: str, table: str, csv_fallback: str | None = None) -> pd.DataFrame:
    """Load a table from SQLite, fallback to CSV if provided.
    
    Args:
        db (str): Path to SQLite DB file.
        table (str): Name of the table to read.
        csv_fallback (str | None): Path to CSV file to fallback on.
        
    Returns:
        pd.DataFrame: DataFrame with date index.
    """
    logger = get_logger("fit_models")
    
    # First try to read from SQLite
    try:
        with sqlite3.connect(db) as conn:
            df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["date"]).set_index("date").sort_index()
            logger.info(f"Loaded table '{table}' from SQLite: shape={df.shape}")
            return df
    
    except Exception as e:
        # Fallback to CSV if DB read fails
        logger.warning(f"Failed to load from SQLite table '{table}': {e}. Trying CSV fallback...")
        if not csv_fallback:
            logger.error(f"No CSV fallback provided. Cannot load data.")
            raise
        df = pd.read_csv(csv_fallback, parse_dates=["date"]).set_index("date").sort_index()
        logger.info(f"Loaded data from CSV '{csv_fallback}': shape={df.shape}")
        return df
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
################################################################################
## Configuration
################################################################################

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
    # Performance controls
    use_gpu: bool = True              # Use GPU for XGBoost if available
    skip_feature_importance: bool = False  # Skip computing feature importances
    n_jobs_model: int = -1             # n_jobs to pass into models
    # Deep learning hyperparameters
    seq_len: int = 360
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 30















################################################################################
# Model factory
################################################################################

def get_model(task: str, kind: str, ridge_alpha: float = 10.0,
              rf_estimators: int = 300, rf_depth: Optional[int] = None,
              logit_C: float = 1.0, random_state: int = 42,
              use_gpu: bool = False, n_jobs: int = -1):
    """Get a model for the specified task and kind.

    Args:
        task (str): The task type ("regression" or "classification").
        kind (str): The model kind (e.g., "ridge", "rf", "xgb").
        ridge_alpha (float, optional): The alpha parameter for Ridge regression. Defaults to 10.0.
        rf_estimators (int, optional): The number of trees in the random forest. Defaults to 300.
        rf_depth (Optional[int], optional): The maximum depth of the trees. Defaults to None.
        logit_C (float, optional): The regularization strength for logistic regression. Defaults to 1.0.
        random_state (int, optional): The random seed. Defaults to 42.

    Raises:
        ImportError: If a required library is not installed.
        ValueError: If an unknown model kind is specified.
        ValueError: If the task is not "regression" or "classification".

    Returns:
        An instance of the specified model.
    """
    logger = get_logger("fit_models")
    task = task.lower()
    kind = kind.lower()

    if task == "regression":
        if kind == "linreg":
            return LinearRegression()
        if kind == "ridge":
            return Ridge(alpha=ridge_alpha, random_state=random_state)
        if kind == "rf":
            return RandomForestRegressor(n_estimators=rf_estimators, max_depth=rf_depth,
                                         random_state=random_state, n_jobs=n_jobs)
        if kind == "xgb":
            return XGBRegressor(
                objective="reg:squarederror",
                n_estimators=rf_estimators,
                max_depth=rf_depth,
                random_state=random_state,
                n_jobs=n_jobs,
                tree_method="hist",
                device="cuda" if use_gpu else "cpu",
            )
        logger.error(f"Unknown regression model kind: {kind}")
        raise ValueError(f"Unknown regression model kind: {kind}, please choose from linreg, ridge, rf, xgb.")

    elif task == "classification":
        if kind == "logit":
            return LogisticRegression(max_iter=200, C=logit_C, n_jobs=n_jobs)
        if kind == "rf":
            return RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_depth,
                                         random_state=random_state, n_jobs=n_jobs)
        if kind == "xgb":
            tree_method = "gpu_hist" if use_gpu else "hist"
            return XGBClassifier(
                objective="binary:logistic",
                n_estimators=rf_estimators,
                max_depth=rf_depth,
                random_state=random_state,
                n_jobs=n_jobs,
                tree_method="hist",
                device="cuda" if use_gpu else "cpu",
                eval_metric="logloss",
            )
        logger.error(f"Unknown classification model kind: {kind}")
        raise ValueError(f"Unknown classification model kind: {kind}, please choose from logit, rf, xgb.")

    else:
        logger.error(f"Invalid task: {task}")
        raise ValueError("task must be 'regression' or 'classification'")



def build_sequences(X_hist: pd.DataFrame, X_future: pd.DataFrame, y_future: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Construct sliding window sequences.

    Args:
        X_hist: Historical features used only for context (e.g., last seq_len-1 rows of train).
        X_future: Features aligned to the targets you want to predict (train or test slice).
        y_future: Targets aligned to X_future.
        seq_len: Window length.

    Returns:
        X_seq: shape (n_seq, seq_len, n_features)
        y_seq: shape (n_seq,)
        idx: list of timestamps corresponding to the last element of each window (matches X_future index)
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    frames = []
    if X_hist is not None and len(X_hist) > 0:
        frames.append(X_hist)
    frames.append(X_future)
    X_combined = pd.concat(frames)
    X_arr = X_combined.to_numpy()
    y_arr = np.asarray(y_future)

    X_seq = []
    y_seq = []
    idx_out: List[pd.Timestamp] = []
    start = len(X_combined) - len(X_future)

    for i in range(start, len(X_combined)):
        s = i - seq_len + 1
        if s < 0:
            continue
        window = X_arr[s:i+1]
        if window.shape[0] != seq_len:
            continue
        y_pos = i - start
        if 0 <= y_pos < len(y_arr):
            X_seq.append(window)
            y_seq.append(y_arr[y_pos])
            idx_out.append(X_combined.index[i])

    if not X_seq:
        return np.empty((0, seq_len, X_arr.shape[1])), np.empty((0,)), []

    X_seq_arr = np.stack(X_seq)
    y_seq_arr = np.asarray(y_seq)
    return X_seq_arr, y_seq_arr, idx_out





class LSTMHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1, out_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq, hidden)
        h_last = out[:, -1, :]
        return self.head(h_last)





class TransformerHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1, out_dim: int = 1):
        super().__init__()
        d_model = max(hidden_dim, 8)
        nhead = max(1, min(4, d_model // 8))
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model * 4,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        x = self.input_proj(x) # (batch, seq, d_model)
        h = self.encoder(x) # (batch, seq, d_model)
        h_last = h[:, -1, :]
        return self.head(h_last)






class TorchSeqEstimator:
    """Minimal sklearn-like wrapper around torch sequence models."""

    def __init__(self, model_type: str, task: str, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, lr: float, batch_size: int, epochs: int, device: str = "cpu"):
        self.model_type = model_type
        self.task = task
        self.device = device
        out_dim = 1
        if model_type == "lstm":
            self.net = LSTMHead(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, out_dim=out_dim)
        elif model_type == "transformer":
            self.net = TransformerHead(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, out_dim=out_dim)
        else:
            raise ValueError(f"Unknown model_type {model_type}")

        self.net.to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray):
        if len(X_seq) == 0:
            raise ValueError("No sequences to train on")
        x = torch.from_numpy(X_seq).float()
        y = torch.from_numpy(y_seq).float().view(-1, 1)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad()
                out = self.net(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
        self.net.eval()
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        if len(X_seq) == 0:
            return np.array([])
        x = torch.from_numpy(X_seq).float().to(self.device)
        with torch.no_grad():
            out = self.net(x).squeeze(-1)
            if self.task == "classification":
                out = torch.sigmoid(out)
        return out.cpu().numpy()
    
    
    












################################################################################
## Evaluation metrics and feature importance
################################################################################
    
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
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    
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























################################################################################
## Walk-forward training and evaluation
################################################################################

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
    logger = get_logger("fit_models")
    
    # Align indices
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    logger.info(f"Aligned data: {len(X)} observations from {X.index[0].date()} to {X.index[-1].date()}")

    # Remove "y_" prefix to get tickers
    tickers = [c.replace("y_", "") for c in y.columns]
    logger.info(f"Processing {len(tickers)} tickers: {', '.join(tickers)}")

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
        print(f"\nFold {fold_no}/{total_folds} - Training period: {X.index[tr_idx[0]].strftime('%Y-%m-%d')} to {X.index[tr_idx[-1]].strftime('%Y-%m-%d')}, "
              f"Test period: {X.index[te_idx[0]].strftime('%Y-%m-%d')} to {X.index[te_idx[-1]].strftime('%Y-%m-%d')}")
        
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

            # Sequence models (PyTorch) vs classical sklearn/xgboost
            use_seq = cfg.model in {"lstm", "transformer"}
            device = "cuda" if (cfg.use_gpu and torch.cuda.is_available()) else "cpu"
            if use_seq:
                model = TorchSeqEstimator(
                    model_type=cfg.model,
                    task=cfg.task,
                    input_dim=X_tr_s.shape[1],
                    hidden_dim=cfg.hidden_dim,
                    num_layers=cfg.num_layers,
                    dropout=cfg.dropout,
                    lr=cfg.lr,
                    batch_size=cfg.batch_size,
                    epochs=cfg.epochs,
                    device=device,
                )
            else:
                model = get_model(cfg.task, cfg.model, cfg.ridge_alpha, cfg.rf_estimators, cfg.rf_depth, cfg.logit_C,
                                  use_gpu=cfg.use_gpu, n_jobs=cfg.n_jobs_model)

            ####################################################################
            # Regression
            ####################################################################
            if cfg.task == "regression":
                if use_seq:
                    # Build train sequences
                    X_seq_tr, y_seq_tr, _ = build_sequences(None, X_tr_s, y_tr_raw, cfg.seq_len)
                    mask_tr = np.isfinite(y_seq_tr)
                    X_seq_tr, y_seq_tr = X_seq_tr[mask_tr], y_seq_tr[mask_tr]
                    if len(y_seq_tr) < 10:
                        print("insufficient data")
                        continue
                    
                    # Build test sequences
                    X_seq_te, y_seq_te, idx_seq = build_sequences(X_tr_s.tail(cfg.seq_len - 1), X_te_s, y_te_raw, cfg.seq_len)
                    
                    # Fit and predict
                    model.fit(X_seq_tr, y_seq_tr)
                    y_pred_seq = model.predict(X_seq_te) if len(X_seq_te) > 0 else np.array([])
                    print("done")

                    # Store predictions
                    for d, yt, yp in zip(idx_seq, y_seq_te, y_pred_seq):
                        preds_records.append({
                            "date": pd.Timestamp(d),
                            "ticker": ticker,
                            "y_true": float(yt) if np.isfinite(yt) else np.nan,
                            "y_pred": float(yp),
                            "fold": fold_no,
                        })

                    # Compute metrics
                    mask = np.isfinite(y_seq_te)
                    if mask.any() and len(y_pred_seq) == len(y_seq_te):
                        m = regression_metrics(y_seq_te[mask], y_pred_seq[mask])
                        m.update({"fold": fold_no, "ticker": ticker})
                        metrics_records.append(m)
                else:
                    # Remove NaNs from training data
                    mtr = np.isfinite(y_tr_raw)
                    if mtr.sum() < 10:
                        print("insufficient data")
                        continue
                    
                    # Fit model with fallback for GPU issues in XGBoost
                    try:
                        if cfg.model == "xgb" and cfg.use_gpu:
                            model.fit(cp.asarray(X_tr_s[mtr].values), cp.asarray(y_tr_raw[mtr]))
                            y_pred = cp.asnumpy(model.predict(cp.asarray(X_te_s.values)))
                        else:
                            model.fit(X_tr_s[mtr], y_tr_raw[mtr])
                    except Exception as e:
                        # Fallback: if GPU requested and fails (e.g., no CUDA), retry on CPU
                        logger = get_logger("fit_models")
                        logger.warning(f"Model fit failed on fold {fold_no}, ticker {ticker}: {e}. Retrying with CPU (no GPU)...")
                        if cfg.model == "xgb" and cfg.use_gpu:
                            model = get_model(cfg.task, cfg.model, cfg.ridge_alpha, cfg.rf_estimators, cfg.rf_depth, cfg.logit_C,
                                              use_gpu=False, n_jobs=cfg.n_jobs_model)
                            model.fit(X_tr_s[mtr], y_tr_raw[mtr])
                        else:
                            breakpoint()
                    
                    # Predict
                    if cfg.model == "xgb" and cfg.use_gpu:
                        y_pred = cp.asnumpy(model.predict(cp.asarray(X_te_s.values)))
                    else:        
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

                    # Compute metrics
                    mask = np.isfinite(y_te_raw)
                    if mask.any():
                        m = regression_metrics(y_te_raw[mask], y_pred[mask])
                        m.update({"fold": fold_no, "ticker": ticker})
                        metrics_records.append(m)

                    # Feature importance
                    if not cfg.skip_feature_importance:
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

                if use_seq:
                    # Build train sequences
                    X_seq_tr, y_seq_tr, _ = build_sequences(None, X_tr_s, y_tr_cls, cfg.seq_len)
                    mask_tr = np.isfinite(y_seq_tr)
                    X_seq_tr, y_seq_tr = X_seq_tr[mask_tr], y_seq_tr[mask_tr]
                    if len(y_seq_tr) < 10:
                        print("insufficient data")
                        continue
                    
                    # Build test sequences
                    X_seq_te, y_seq_te, idx_seq = build_sequences(X_tr_s.tail(cfg.seq_len - 1), X_te_s, y_te_cls, cfg.seq_len)
                    
                    # Fit and predict
                    model.fit(X_seq_tr, y_seq_tr)
                    y_prob_seq = model.predict(X_seq_te) if len(X_seq_te) > 0 else np.array([])
                    y_pred_seq_cls = (y_prob_seq > 0.5).astype(int)
                    print("done")

                    # Store predictions
                    for d, yt, yp_prob, yp_cls in zip(idx_seq, y_seq_te, y_prob_seq, y_pred_seq_cls):
                        preds_records.append({
                            "date": pd.Timestamp(d),
                            "ticker": ticker,
                            "y_true_cls": int(yt) if np.isfinite(yt) else np.nan,
                            "y_prob": float(yp_prob),
                            "y_pred_cls": int(yp_cls),
                            "fold": fold_no,
                        })

                    # Compute metrics
                    mte = np.isfinite(y_seq_te)
                    if mte.any() and len(y_prob_seq) == len(y_seq_te):
                        yt = y_seq_te[mte]
                        yp = y_pred_seq_cls[mte]
                        pr = y_prob_seq[mte]
                        m = classification_metrics(yt, pr, yp)
                        m.update({"fold": fold_no, "ticker": ticker})
                        metrics_records.append(m)
                else:
                    # Remove NaNs from training data
                    mtr = np.isfinite(y_tr_raw)
                    if mtr.sum() < 10:
                        print("insufficient data")
                        continue
                    
                    # Fit model with fallback for GPU issues in XGBoost
                    try:
                        if cfg.model == "xgb" and cfg.use_gpu:
                            model.fit(cp.asarray(X_tr_s[mtr].values), cp.asarray(y_tr_cls[mtr]))
                        else:
                            model.fit(X_tr_s[mtr], y_tr_cls[mtr])
                    except Exception as e:
                        logger = get_logger("fit_models")
                        logger.warning(f"Model fit failed on fold {fold_no}, ticker {ticker}: {e}. Retrying with CPU (no GPU)...")
                        if cfg.model == "xgb" and cfg.use_gpu:
                            model = get_model(cfg.task, cfg.model, cfg.ridge_alpha, cfg.rf_estimators, cfg.rf_depth, cfg.logit_C,
                                              use_gpu=False, n_jobs=cfg.n_jobs_model)
                            model.fit(X_tr_s[mtr], y_tr_cls[mtr])
                        else:
                            breakpoint()
                    
                    # Predict probabilities and classes
                    if cfg.model == "xgb" and cfg.use_gpu:
                        y_prob = cp.asnumpy(model.predict_proba(cp.asarray(X_te_s.values))[:, 1])
                        y_pred_cls = cp.asnumpy(model.predict(cp.asarray(X_te_s.values)))
                    else:
                        if hasattr(model, "predict_proba"):
                            y_prob = model.predict_proba(X_te_s)[:, 1]
                        else:
                            if hasattr(model, "decision_function"):
                                z = model.decision_function(X_te_s)
                                y_prob = 1.0 / (1.0 + np.exp(-z))
                            else:
                                breakpoint()
                        y_pred_cls = model.predict(X_te_s)
                    print("done")


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
                    
                    # Compute metrics
                    mte = np.isfinite(y_te_raw)
                    if mte.any():
                        yt = y_te_cls[mte]
                        yp = y_pred_cls[mte]
                        pr = y_prob[mte] if y_prob is not None else None
                        m = classification_metrics(yt, pr, yp)
                        m.update({"fold": fold_no, "ticker": ticker})
                        metrics_records.append(m)
                    
                    # Feature importance
                    if not cfg.skip_feature_importance:
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
    feature_importances_long = pd.DataFrame(feature_importance_records).sort_values(["fold", "ticker", "feature"]) if feature_importance_records else pd.DataFrame()
    
    print(f"\nWalk-forward CV completed!")
    print(f"Generated {len(predictions_long)} predictions across {total_folds} folds and {len(tickers)} tickers")

    return predictions_long, metrics_long, feature_importances_long
























################################################################################
# Main entry point
################################################################################

def main(db: str, X_table: str, y_table: str, X_csv: str, y_csv: str,
         task: str, model: str,
         ridge_alpha: float, rf_estimators: int, rf_depth: Optional[int], logit_C: float,
         train_size: int, test_size: int, step: int, expanding: bool, embargo: int,
         direction_threshold: float, use_gpu: bool, skip_feature_importance: bool, n_jobs_model: int,
         seq_len: int, hidden_dim: int, num_layers: int, dropout: float, lr: float, batch_size: int, epochs: int,
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
        train_size (int): Training window size in days.
        test_size (int): Test window size in days.
        step (int): Step size in days for rolling window.
        expanding (bool): Whether to use expanding window (True) or rolling (False).
        embargo (int): Embargo period in days to prevent leakage.
        direction_threshold (float): Threshold for classification labels.
        out_dir (str): Directory to save output CSV files.
        save_db (bool): Whether to also save predictions/metrics back to the SQLite DB.
    """
    logger = get_logger("fit_models")
    
    logger.info("="*80)
    logger.info(f"Starting model training with task={task}, model={model}")
    logger.info(f"Config: train_size={train_size}, test_size={test_size}, step={step}, expanding={expanding}, embargo={embargo}")
    logger.info(f"Perf: use_gpu={use_gpu}, skip_feature_importance={skip_feature_importance}, n_jobs_model={n_jobs_model}")
    logger.info(f"DL: model={model}, seq_len={seq_len}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, lr={lr}, batch_size={batch_size}, epochs={epochs}")
    logger.info("="*80)
    
    # Load data
    logger.info(f"Loading features from {X_table} (DB: {db}, CSV fallback: {X_csv})")
    X = read_table(db, X_table, csv_fallback=X_csv)
    logger.info(f"Loading targets from {y_table} (DB: {db}, CSV fallback: {y_csv})")
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
        use_gpu=use_gpu,
        skip_feature_importance=skip_feature_importance,
        n_jobs_model=n_jobs_model,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
    )

    # Run walk-forward Cross Validation and fit models
    logger.info("Starting walk-forward cross-validation")
    preds, metrics, feature_importances = run_walkforward(X, y, cfg)

    # Save outputs
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{task}_{model}"
    
    preds_path = out_dir / f"predictions_{suffix}.csv"
    metrics_path = out_dir / f"metrics_{suffix}.csv"
    feature_importances_path = out_dir / f"feature_importances_{suffix}.csv"

    preds.to_csv(preds_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    feature_importances.to_csv(feature_importances_path, index=False)

    logger.info(f"Saved predictions ({len(preds)} rows) to {preds_path}")
    logger.info(f"Saved metrics ({len(metrics)} rows) to {metrics_path}")
    logger.info(f"Saved feature importances ({len(feature_importances)} rows) to {feature_importances_path}")

    print(f"Saved predictions         -> {preds_path}")
    print(f"Saved metrics             -> {metrics_path}")
    print(f"Saved feature importances -> {feature_importances_path}")

    if save_db:
        try:
            with sqlite3.connect(db) as conn:
                preds.to_sql(f"predictions_{suffix}", conn, if_exists="replace", index=False)
                metrics.to_sql(f"metrics_{suffix}", conn, if_exists="replace", index=False)
                feature_importances.to_sql(f"feature_importances_{suffix}", conn, if_exists="replace", index=False)
            logger.info(f"Also wrote to SQLite tables: predictions_{suffix}, metrics_{suffix}, feature_importances_{suffix}")
            print(f"Also wrote to SQLite tables: predictions_{suffix}, metrics_{suffix}, feature_importances_{suffix}")
        except Exception as e:
            logger.error(f"Failed to save to SQLite: {e}")
    
    logger.info("="*80)
    logger.info("Model training completed successfully!")
    logger.info("="*80)





if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Baseline models with walk-forward CV (regression & classification)")
    args.add_argument("--db", default="data/data.db")
    args.add_argument("--X_table", default="X_features")
    args.add_argument("--y_table", default="y_targets")
    args.add_argument("--X_csv", default="data/processed/X_features.csv")
    args.add_argument("--y_csv", default="data/processed/y_targets.csv")

    # Task: regression or classification
    args.add_argument("--task", choices=["regression", "classification"], default="regression")

    # Models: regression -> ridge|rf|xgb|lstm|transformer ; classification -> logit|rf|xgb|lstm|transformer
    args.add_argument("--model", choices=["ridge", "logit", "rf", "xgb", "lstm", "transformer"], default="ridge")

    # Hyperparams
    args.add_argument("--ridge_alpha", type=float, default=10.0)
    args.add_argument("--rf_estimators", type=int, default=100)
    args.add_argument("--rf_depth", type=int, default=6)
    args.add_argument("--logit_C", type=float, default=1.0)

    # Performance options
    args.add_argument("--use_gpu", action="store_true", help="Use GPU for XGBoost if available")
    args.add_argument("--skip_feature_importance", action="store_true", help="Skip computing feature importances to speed up")
    args.add_argument("--n_jobs_model", type=int, default=-1, help="n_jobs for models (scikit-learn/XGBoost)")

    # Deep learning options
    args.add_argument("--seq_len", type=int, default=365, help="Sequence length for LSTM/Transformer")
    args.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for LSTM/Transformer")
    args.add_argument("--num_layers", type=int, default=1, help="Number of layers for LSTM/Transformer")
    args.add_argument("--dropout", type=float, default=0.1, help="Dropout for LSTM/Transformer")
    args.add_argument("--lr", type=float, default=1e-3, help="Learning rate for LSTM/Transformer")
    args.add_argument("--batch_size", type=int, default=64, help="Batch size for LSTM/Transformer")
    args.add_argument("--epochs", type=int, default=30, help="Training epochs for LSTM/Transformer")

    # Walk-forward
    args.add_argument("--train_size", type=int, default=365)
    args.add_argument("--test_size", type=int, default=1)
    args.add_argument("--step", type=int, default=1)
    args.add_argument("--expanding", action="store_true")
    args.add_argument("--rolling", dest="expanding", action="store_false")
    args.set_defaults(expanding=True)
    args.add_argument("--embargo", type=int, default=0)

    # Classification-specific
    args.add_argument("--direction_threshold", type=float, default=0.0, help="Label threshold: y > thr => class 1")

    # Output
    args.add_argument("--out_dir", default="data/predictions")
    args.add_argument("--save_db", action="store_true", help="Also write predictions/metrics to SQLite")
    args = args.parse_args()

    main(args.db, args.X_table, args.y_table, args.X_csv, args.y_csv, 
        args.task, args.model,
        args.ridge_alpha, args.rf_estimators, args.rf_depth, args.logit_C,
        args.train_size, args.test_size, args.step, args.expanding, args.embargo, 
        args.direction_threshold, args.use_gpu, args.skip_feature_importance, args.n_jobs_model,
        args.seq_len, args.hidden_dim, args.num_layers, args.dropout, args.lr, args.batch_size, args.epochs,
        args.out_dir, args.save_db)
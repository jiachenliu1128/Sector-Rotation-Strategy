import argparse
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


def walkforward_splits(
    index: pd.Index,
    train_size: int = 60,
    test_size: int = 12,
    step: int = 12,
    expanding: bool = True,
    embargo: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, test_idx) integer arrays for walk-forward CV.

    Args:
        index : sequence of labels (DataFrame.index)
        train_size, test_size, step, expanding, embargo : see `WalkForwardConfig` below.

    Yields:
    (train_idx, test_idx) : tuple of np.ndarray
        Integer positions into your arrays/DataFrames.
    """
    # Basic check
    n = len(index)
    if train_size <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("train_size, test_size, and step must be positive integers")
    if train_size + test_size > n:
        return 

    # Generate splits if expanding or rolling
    if expanding:
        # Initial expanding windows
        train_end = train_size  
        test_end = train_end + test_size
        # Expanding train windows
        while test_end <= n:
            train_start = 0
            # Apply embargo by trimming from train_end
            emb_end = max(train_end - embargo, 0)
            train = np.arange(train_start, emb_end)
            test = np.arange(train_end, test_end)
            if len(train) > 0 and len(test) > 0:
                yield train, test
            # Next fold
            train_end = min(train_end + step, n)
            test_end = min(train_end + test_size, n)
    else:
        # Rolling train windows
        train_start = 0
        train_end = train_start + train_size
        test_end = train_end + test_size
        # Fixed-length rolling
        while test_end <= n:
            # Apply embargo by trimming from train_end
            emb_end = max(train_end - embargo, train_start)
            train = np.arange(train_start, emb_end)
            test = np.arange(train_end, test_end)
            if len(train) > 0 and len(test) > 0:
                yield train, test
            # Next fold
            train_start = train_start + step
            train_end = train_start + train_size
            test_end = train_end + test_size
            
            
      
            

@dataclass
class WFConfig:
    """
    Configuration for walk-forward cross-validation.

    Parameters：
        train_size : int
            Number of time steps in each training window (in months if your index is monthly).
            If `expanding=True`, this is the *minimum* initial train length; later folds expand.
        test_size : int
            Number of time steps in each test window.
        step : int
            How far to advance the window between folds (e.g., 12 for yearly steps on monthly data).
        expanding : bool
            If True, use an expanding training window; if False, use a rolling (fixed-length) window.
        embargo : int
            Number of observations to exclude at the *end* of the training set before the test period
            (helps reduce leakage when features target near-future info). 0 disables embargo.
    """
    train_size: int = 60
    test_size: int = 12
    step: int = 12
    expanding: bool = True
    embargo: int = 0





class WFCV:
    """
    Class for Scikit-learn compatible walk-forward splitter.
    """
    def __init__(self, config: Optional[WFConfig] = None):
        self.config = config or WFConfig()

    def get_n_splits(self, X: Optional[pd.DataFrame]=None, y=None, groups=None) -> int:
        if X is None:
            raise ValueError("get_n_splits requires X with an index to determine length")
        count = 0
        for _ in walkforward_splits(
            X.index,
            train_size=self.config.train_size,
            test_size=self.config.test_size,
            step=self.config.step,
            expanding=self.config.expanding,
            embargo=self.config.embargo,
        ):
            count += 1
        return count

    def split(self, X: Optional[pd.DataFrame]=None, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if X is None:
            raise ValueError("get_n_splits requires X with an index to determine length")
        for train, test in walkforward_splits(
            X.index,
            train_size=self.config.train_size,
            test_size=self.config.test_size,
            step=self.config.step,
            expanding=self.config.expanding,
            embargo=self.config.embargo,
        ):
            yield train, test





def demo_from_csv(csv_path: str, date_col: str = "date", train_size: int = 60,
    test_size: int = 12, step: int = 12, expanding: bool = True, embargo: int = 0) -> pd.DataFrame:
    """Utility to preview folds from a CSV that has a `date` column.
    
    Args:
        csv_path (str): Path to CSV file.
        date_col (str): Name of the date column in the CSV.
        train_size (int): Size of the training set.
        test_size (int): Size of the test set.
        step (int): Step size for the rolling window.
        expanding (bool): Whether to use an expanding window.
        embargo (int): Embargo period to prevent leakage.

    Return:
    A small DataFrame listing fold numbers and date spans for train/test.
    """
    # Read CSV with date parsing and sorting
    df = pd.read_csv(csv_path, parse_dates=[date_col]).set_index(date_col).sort_index()
    idx = df.index
    rows = []
    # Generate splits
    splits = walkforward_splits(idx, train_size=train_size, test_size=test_size, step=step, expanding=expanding, embargo=embargo)

    # Log each fold 
    for k, (train, test) in enumerate(splits):
        rows.append(
            {
                "fold": k + 1,
                "train_start": idx[train[0]],
                "train_end": idx[train[-1]],
                "test_start": idx[test[0]],
                "test_end": idx[test[-1]],
                "n_train": len(train),
                "n_test": len(test),
            }
        )
    return pd.DataFrame(rows)





def main(csv_path: str, date_col: str, train_size: int, test_size: int, step: int, expanding: bool, embargo: int) -> None:
    """Run the walk-forward cross-validation demo.

    Args:
        csv_path (str): Path to the CSV file containing the data.
        date_col (str): Name of the date column in the CSV.
        train_size (int): Size of the training set.
        test_size (int): Size of the test set.
        step (int): Step size for the rolling window.
        expanding (bool): Whether to use an expanding window.
        embargo (int): Embargo period to prevent leakage.
    """
    summary = demo_from_csv(csv_path, date_col=date_col, train_size=train_size, 
                            test_size=test_size, step=step, expanding=expanding,
                            embargo=embargo)
    
    # Print a summary of the folds
    if summary.empty:
        print("No folds produced — check sizes vs. series length.")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(summary)





if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Walk-forward Cross Validation generator")
    
    args.add_argument("--csv", default="data/processed/X_features.csv", help="CSV with a 'date' column to preview folds")
    args.add_argument("--date_col", default="date", help="Name of date column in CSV")
    
    args.add_argument("--train_size", type=int, default=60, help="Training window size (months)")
    args.add_argument("--test_size", type=int, default=12, help="Testing window size (months)")
    args.add_argument("--step", type=int, default=12, help="Step size (months), how far to move window each fold")

    args.add_argument("--expanding", action="store_true", help="Use expanding window, includes all available past data if True")
    args.add_argument("--rolling", dest="expanding", action="store_false", help="Use rolling window instead")
    args.set_defaults(expanding=True)

    args.add_argument("--embargo", type=int, default=0, help="Embargo period (months) to prevent leakage, 0 disables embargo")

    args = args.parse_args()
    main(args.csv, args.date_col, args.train_size, args.test_size, args.step, args.expanding, args.embargo)
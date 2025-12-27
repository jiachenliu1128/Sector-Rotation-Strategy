# Machine Learning-driven Sector Rotation Strategy

Machine-learning research pipeline for building, training, and backtesting a daily U.S. sector rotation strategy driven by ETF and macro features.

## Project Layout
- `config.yaml` — universe, benchmark, and FRED series configuration.
- `src/data_loader.py` - fetch macro + ETF data
- `src/fred_data_prep.py`, `src/etf_data_prep.py` — clean and align macro + ETF data.
- `src/etf_compute_features.py` — create momentum, volatility, RSI, and MACD factors.
- `src/build_feature_and_target.py` — align features with next-day excess returns.
- `src/fit_models.py` — walk-forward training for ridge, RF, XGBoost, and logistic models.
- `src/backtest.py` — portfolio construction and performance reporting.
- `data/` — processed datasets, predictions, and backtest outputs.
- `notebooks/` — exploratory analysis and model evaluation.

## Setup
1. Create a Python 3.11+ environment.
2. Install dependencies: `pip install pandas numpy scikit-learn xgboost sqlalchemy`.
3. Optional: populate the SQLite store at `data/data.db` if you prefer DB inputs over CSV.
4. Create `.env` file with any API keys needed (e.g., `FRED_API_KEY=your_key_here`).

## Quick Start
1. **Download & prep data**
   - `python src/data_loader.py`
   - `python src/fred_data_prep.py`
   - `python src/etf_data_prep.py`
   - `python src/etf_compute_features.py`
2. **Build modeling tables**
   - `python src/build_feature_and_target.py`
3. **Train & score models**
   - `python src/fit_models.py --task regression --model ridge`
   - Results appear in `data/predictions/`.
4. **Run backtests**
   - `python src/backtest.py --task regression --model ridge`
   - Portfolio stats/curves saved to `data/backtest/`.

Adjust CLI flags (e.g., window sizes, model type, long-short mode) to experiment with other configurations.


## TODO:
1. Use daily data instead of monthly
2. Use deep learning models (LSTM, GRU, Transformer)
3. Create dashboard
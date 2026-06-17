"""alphacore: pure-compute quant strategy engine (signals, portfolio, ml, backtest).

Data loading is intentionally out of scope: callers fill BacktestData from their own
source. See README for the column contract.
"""
from alphacore.factors import factor_exposures_at, gics_sector_map
from alphacore.ml.governed import build_governed_alpha
from alphacore.portfolio.backtest import BacktestResult, run_backtest
from alphacore.portfolio.construction import (
    BacktestConfig,
    BacktestData,
    construct_portfolios,
)
from alphacore.portfolio.optimizer import OptimizerParams
from alphacore.signals.pipeline import (
    DEFAULT_SLEEVES,
    assemble_cross_section,
    compute_ticker_signals,
)
from alphacore.signals.regime import (
    RegimeParams,
    c_market_regime_signal,
    c_regime_cash_weight,
)

__all__ = [
    "BacktestConfig",
    "BacktestData",
    "BacktestResult",
    "DEFAULT_SLEEVES",
    "OptimizerParams",
    "RegimeParams",
    "assemble_cross_section",
    "build_governed_alpha",
    "c_market_regime_signal",
    "c_regime_cash_weight",
    "compute_ticker_signals",
    "construct_portfolios",
    "factor_exposures_at",
    "gics_sector_map",
    "run_backtest",
]

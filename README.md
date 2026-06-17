# alphacore

Pure-compute quant strategy engine: signals, portfolio, ml, backtest.

## What it is / is not

`alphacore` is **pure compute on in-memory DataFrames**. It scores a universe,
constructs a `date x ticker` weight panel, and runs a fast research backtest — all
on `pandas`/`numpy` objects you pass in.

Data loading is **out of scope**. The package never touches the network, a database,
or the filesystem for market data: callers build a `BacktestData` from their own
source and hand it in. This keeps the engine deterministic and trivially testable.

It is a **private package**. Install over SSH at a tagged version:

```bash
pip install git+ssh://git@github.com/AyrtonSoler/alphacore@v0.1.0
```

## Install (dev)

```bash
python3.12 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```

## The `BacktestData` contract

`BacktestData` (`alphacore.portfolio.construction`) is a frozen dataclass holding the
cross-date market data the rebalance loop reads. The caller is responsible for
producing each field with the exact index/columns below.

| Field | Type | Index / columns expected |
| --- | --- | --- |
| `sectors` | `pd.Series[str]` | Indexed by ticker -> GICS sector (drives the sector-neutral z-score). |
| `factor_matrices` | `Mapping[str, pd.DataFrame]` | Factor name -> a `date x ticker` matrix (date-indexed, ticker-columned) from the consumer's factor-matrix loader. |
| `benchmark_weights` | `pd.DataFrame` | `date x ticker`: KN600 daily constituent weights; the optimizer's `w_bench` (read as-of each rebalance date). |
| `volatility` *(optional)* | `pd.Series[float] \| None` | Indexed by ticker -> volatility for the diagonal risk model. Defaults to 1.0 when omitted. |
| `adv` *(optional)* | `pd.Series[float] \| None` | Indexed by ticker -> average dollar volume for the liquidity cap. Defaults to deep liquidity (no binding cap). |
| `cash_schedule` *(optional)* | `pd.Series[float] \| None` | Indexed by date -> cash weight from the regime overlay; read as-of each rebalance date. When absent, the fixed `BacktestConfig.cash` is used. |

> Note: `BacktestData` does **not** carry a price panel. Per-name signals arrive via
> the `signal_frames` argument to `construct_portfolios`, and forward returns are
> supplied directly to `run_backtest`.

The companion `BacktestConfig` (also in `construction.py`) holds run configuration:
`params: OptimizerParams`, `sleeves: Mapping[str, Sequence[str]] | None`,
`cash: float = 0.01`, `aum: float = 100_000_000.0`.

## Minimal usage

```python
import pandas as pd

from alphacore import BacktestData, BacktestConfig, construct_portfolios, run_backtest

dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
tickers = ["AAA", "BBB", "CCC"]

# Per-ticker signal frames (from your signal pipeline), keyed by ticker.
signal_frames = {t: pd.DataFrame(index=dates) for t in tickers}

data = BacktestData(
    sectors=pd.Series({"AAA": "Tech", "BBB": "Tech", "CCC": "Health"}),
    factor_matrices={"value": pd.DataFrame(0.0, index=dates, columns=tickers)},
    benchmark_weights=pd.DataFrame(1 / 3, index=dates, columns=tickers),
)

weights = construct_portfolios(signal_frames, dates, data, config=BacktestConfig())

forward_returns = pd.DataFrame(0.0, index=dates, columns=tickers)
benchmark_returns = pd.Series(0.0, index=dates)
result = run_backtest(weights, forward_returns, benchmark_returns, cost_bps=5.0)
print(result.summary())
```

## Tests

```bash
PYTHONHASHSEED=0 pytest tests -q
```

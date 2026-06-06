"""Prepare KaxaNuk attribution-analysis inputs from our KN600 pipeline artifacts.

The attribution package loads, per factor, a ``[date, asset1, asset2, …]`` CSV and keeps
only the columns that intersect the portfolio's assets (see
``kaxanuk.attribution_analysis.modules.data_wrangling.load_by_factor_returns_arrow``). The
official KN factor files are date-indexed matrices whose first header cell is empty; the
loader takes ``available_cols[0]`` as the date column and feeds it to pyarrow's column
selector, which needs a real name. These helpers slim each KN factor file to the portfolio
universe and name the date column, so the loader reads a small, well-formed file instead of
a multi-hundred-MB matrix with an unnamed index.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATE = "date"


def assets_from_weights(weights_csv: Path | str, date_column: str = "date_column") -> list[str]:
    """Return the asset (ticker) columns of a Backtest-Engine-format weights file.

    Parameters
    ----------
    weights_csv:
        Path to the wide weights file (``date_column`` + one column per ticker).
    date_column:
        Header of the date column to exclude from the asset list.

    Returns
    -------
    list[str]
        The ticker column names, in file order.

    """
    header = pd.read_csv(weights_csv, nrows=0)
    return [c for c in header.columns if c != date_column]


def slim_factor_frame(frame: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """Subset a date-indexed factor matrix to ``assets`` and name the date index ``date``.

    Parameters
    ----------
    frame:
        Date-indexed factor matrix (index = dates, columns = tickers).
    assets:
        Tickers to keep. Columns absent from the frame are skipped; the original column
        order is preserved.

    Returns
    -------
    pandas.DataFrame
        The column-subset frame with its index named ``date``, so a CSV round-trip yields
        a named date column. Idempotent.

    """
    wanted = set(assets)
    keep = [c for c in frame.columns if c in wanted]
    out = frame.loc[:, keep].copy()
    out.index.name = _DATE
    return out


def slim_factor_csv(src: Path | str, assets: list[str], out: Path | str) -> Path:
    """Read a KN factor CSV (date in column 0), slim it to ``assets``, write to ``out``.

    Parameters
    ----------
    src:
        Source KN factor CSV (date-indexed matrix; first header cell may be empty).
    assets:
        Tickers to keep.
    out:
        Destination CSV path; parent directories are created.

    Returns
    -------
    pathlib.Path
        The written ``out`` path. The written file's first column header is ``date``.

    """
    frame = pd.read_csv(src, index_col=0)
    slim = slim_factor_frame(frame, assets)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(out_path)
    return out_path


def prepare_factor_returns(
    factor_files: dict[str, Path | str], assets: list[str], out_dir: Path | str
) -> list[Path]:
    """Slim a set of KN factor files into ``out_dir`` for the attribution loader.

    Parameters
    ----------
    factor_files:
        Mapping of ``factor_name -> source CSV path``. The output file is
        ``out_dir/<factor_name>.csv`` (the loader derives the factor name from the file
        stem).
    assets:
        Tickers to keep in every factor file.
    out_dir:
        Destination directory; created if missing.

    Returns
    -------
    list[pathlib.Path]
        The written factor CSV paths, in ``factor_files`` iteration order.

    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for factor_name, src in factor_files.items():
        written.append(slim_factor_csv(src, assets, out_dir_path / f"{factor_name}.csv"))
    return written

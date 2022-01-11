from pandas import DataFrame
from typing import Iterable


def exclude_cols_from_df(df: DataFrame, exclude_cols: Iterable[str]):
    return [f for f in df.columns if f not in exclude_cols]

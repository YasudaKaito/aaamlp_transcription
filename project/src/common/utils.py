import re
import string
from typing import Iterable

from pandas import DataFrame


def exclude_cols_from_df(df: DataFrame, exclude_cols: Iterable[str]):
    return [f for f in df.columns if f not in exclude_cols]


def clean_text(s: str):
    # 1つ以上の半角スペースで分割
    result = s.split()
    # 得られたトークンを半角スペースで結合
    result = " ".join(result)
    # 句読点を削除
    # []の中のどれかにマッチ
    result = re.sub(f"[{re.escape(string.punctuation)}]", "", result)
    return result

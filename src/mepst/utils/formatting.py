"""
Reformat dataframes or any other data structures.
"""

from pathlib import Path
from typing import List
import pandas as pd

import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    """Credits to:
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class GenericEncoder(json.JSONEncoder):
    """Credits to:
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return obj.resolve().as_posix()
        return super(NpEncoder, self).default(obj)


def rearrange_cols_first(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Rearrange the dataframe's columns,
    by setting `cols` as the first columns"""
    return df[cols + [c for c in df.columns if c not in cols]]


def rearrange_cols_last(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Rearrange the dataframe's columns,
    by setting `cols` as the last columns"""
    return df[[c for c in df.columns if c not in cols] + cols]

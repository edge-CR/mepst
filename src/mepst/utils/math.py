from typing import TypeAlias

import numpy as np
import pandas as pd

_ArrayOrSeries: TypeAlias = np.ndarray | pd.Series


def norm(v: _ArrayOrSeries) -> _ArrayOrSeries:
    """The function that for some reason does not exist in numpy"""
    return v / v.sum()


def jaccard(s1: set, s2: set, undefined_when_empty: bool = True) -> float:
    """Intersection over union (cardinals)."""
    num = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom:
        return num / denom
    return np.nan if undefined_when_empty else 0.0

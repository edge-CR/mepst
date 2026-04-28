import mpbn
import numpy as np
import pandas as pd

from .utils import cast_trap_space_to_marker


def check_single_target_reachable_from_source(bn, source, target):
    """ """
    source = cast_trap_space_to_marker(source)
    success = False
    for i, a in enumerate(bn.attractors(reachable_from=source)):
        if i >= 1:
            success = False
            break
        if target == a:
            success = True
    return success

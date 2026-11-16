"""E0, kalibracja stalych zlozonosci: przesiew (ten sam protokol co kalibracja_przesiew.py).

Kalibrujemy 3 stale grupy complexity: P(depth) (depth_mode_prob), c (depth_low_coef,
d_min = ceil(c * log2 n)) i leaf_max. K, splitter i BR zostaja jak w A_p.

Konfiguracje: A_p (obecne stale), kontrola A_p z jednostajnym rozkladem glebokosci
(depth_distribution = 'uniform'), N konfiguracji z LHS i RF domyslny jako punkt
odniesienia (poza rankingiem). Protokol, ziarna, ranking i podsumowanie jak w
kalibracja_przesiew.py (16 zbiorow dev, foldy 0-4, B = 150).

    python kalibracja_przesiew_zlozonosc.py [--configs 20] [--trees 150] [--folds 5] [--summary-only]
"""

from dataclasses import replace

import numpy as np

from hdrf import SamplingConfig
from kalibracja_przesiew import REFERENCE, lhs_points, main

# leaf_max losujemy w skali log (ciagle w [ln 2, ln 16)) i zaokraglamy w dol: {2, ..., 15}.
RANGES = {"p_depth": (0.2, 0.8), "c": (0.5, 1.2), "leaf_max": (2, 15)}


def make_configs(n_lhs):
    base = SamplingConfig()
    configs = {REFERENCE: base, "A_p depth uniform": replace(base, depth_distribution="uniform")}
    scaled = dict(RANGES, leaf_max=(np.log(2), np.log(16)))
    for i, (p_depth, c, log_leaf) in enumerate(lhs_points(scaled, n_lhs)):
        configs[f"LHS{i:02d}"] = replace(base, depth_mode_prob=float(p_depth),
                                         depth_low_coef=float(c),
                                         leaf_max=int(min(15, np.floor(np.exp(log_leaf)))))
    return configs


if __name__ == "__main__":
    main(make_configs, RANGES, "results/kalibracja_zlozonosc")

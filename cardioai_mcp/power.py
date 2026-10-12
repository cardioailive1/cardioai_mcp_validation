"""Sample-size / power for the pre-specified performance goals.

Success = lower 95% CI bound >= goal (upper bound <= goal for 'le').  Power is
computed for an assumed TRUE external performance; we report two scenarios:
  * 'phase1'   — true external performance equals the Phase 1 estimate
  * 'degraded' — true performance halfway between Phase 1 and the goal
                 (typical external-validation shrinkage)
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

Z = stats.norm.ppf(0.975)


def _k_crit(n: int, goal: float) -> int:
    """Smallest k whose Wilson lower bound >= goal (lower bound is monotone in k)."""
    from .metrics import wilson

    lo, hi = 0, n + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if mid <= n and wilson(mid, n)[0] >= goal:
            hi = mid
        else:
            lo = mid + 1
    return lo


def power_proportion(n: int, p_true: float, goal: float) -> float:
    """Exact binomial power that the Wilson 95% lower bound clears `goal`."""
    k = _k_crit(n, goal)
    return float(stats.binom.sf(k - 1, n, p_true)) if k <= n else 0.0


def n_for_proportion(p_true: float, goal: float, power: float = 0.80, n_max: int = 50000) -> int | None:
    """Smallest n with power >= target that stays powered for the next 10 n (binomial saw-tooth)."""
    if p_true <= goal:
        return None
    n0 = max(10, int(((Z + stats.norm.ppf(power)) ** 2 * p_true * (1 - p_true)) / (p_true - goal) ** 2 * 0.6))
    for n in range(n0, n_max):
        if power_proportion(n, p_true, goal) >= power and all(
            power_proportion(n + j, p_true, goal) >= power for j in range(1, 11)
        ):
            return n
    return None


def auc_se(a: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg))


def power_auc(a_true: float, goal: float, n_pos: int, n_neg: int) -> float:
    se = auc_se(a_true, n_pos, n_neg)
    return float(stats.norm.cdf((a_true - goal) / se - Z))


def n_events_for_auc(a_true: float, goal: float, prevalence: float, power: float = 0.80) -> int | None:
    if a_true <= goal:
        return None
    for n_pos in range(10, 20000):
        n_neg = int(round(n_pos * (1 - prevalence) / prevalence))
        if power_auc(a_true, goal, n_pos, n_neg) >= power:
            return n_pos
    return None


def power_correlation(r_true: float, goal: float, n: int) -> float:
    se = 1 / math.sqrt(n - 3)
    return float(stats.norm.cdf((math.atanh(r_true) - math.atanh(goal)) / se - Z))


def n_for_mae(mae_true: float, goal: float, sd_abs_err: float, power: float = 0.80) -> int:
    return math.ceil(((Z + stats.norm.ppf(power)) * sd_abs_err / (goal - mae_true)) ** 2)


def power_kappa(k_true: float, goal: float, n: int, cats: int = 3) -> float:
    """Analytic (Fleiss large-sample) power, equal marginals over `cats` categories."""
    pe = 1 / cats
    po = k_true * (1 - pe) + pe
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    return float(stats.norm.cdf((k_true - goal) / se - Z))


def power_kappa_sim(k_true: float, goal: float, n: int, cats: int = 3, sims: int = 400, seed: int = 1) -> float:
    """Simulation: equal-marginal categories, agreement tuned to kappa=k_true; bootstrap-free
    large-sample SE (Fleiss) for the decision."""
    rng = np.random.default_rng(seed)
    pe = 1 / cats
    po = k_true * (1 - pe) + pe
    wins = 0
    for _ in range(sims):
        a = rng.integers(0, cats, n)
        agree = rng.random(n) < (po - pe) / (1 - pe)
        b = np.where(agree, a, rng.integers(0, cats, n))
        po_hat = (a == b).mean()
        pa = np.bincount(a, minlength=cats) / n
        pb = np.bincount(b, minlength=cats) / n
        pe_hat = (pa * pb).sum()
        k = (po_hat - pe_hat) / (1 - pe_hat)
        se = math.sqrt(po_hat * (1 - po_hat) / (n * (1 - pe_hat) ** 2))
        wins += (k - Z * se) >= goal
    return wins / sims


def scenario(phase1: float, goal: float, kind: str = "degraded") -> float:
    return phase1 if kind == "phase1" else (phase1 + goal) / 2

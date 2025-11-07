import numpy as np
from typing import Tuple, Literal

def _probs(c: np.ndarray) -> np.ndarray:
    """
    Return normalized probabilities p_i = |c_i|^2 / sum_j |c_j|^2.
    Handles complex c and the all-zero edge case.
    """
    c = np.asarray(c)
    w = np.abs(c)**2
    Z = w.sum()
    if Z == 0:
        # convention: uniform if the vector is identically zero
        return np.ones_like(w, dtype=float) / max(1, w.size)
    return w / Z

def ipr(c: np.ndarray) -> float:
    """
    Inverse Participation Ratio: sum_i p_i^2, where p_i = |c_i|^2 / sum |c|^2.
    Larger IPR => more concentrated.
    """
    p = _probs(c)
    return float(np.sum(p**2))

def pr(c: np.ndarray) -> float:
    """
    Participation number: 1 / IPR. Interpreted as the 'effective' number of components.
    """
    I = ipr(c)
    return float(1.0 / I) if I > 0 else np.inf

def ipr_sparsity_score(c: np.ndarray) -> float:
    """
    Normalized IPR sparsity score in [0,1]:
        S_IPR = (IPR - 1/n) / (1 - 1/n)
    Returns 1 for a basis vector, 0 for the uniform vector.
    """
    n = max(1, int(np.size(c)))
    if n == 1:
        return 1.0
    I = ipr(c)
    return float((I - 1.0/n) / (1.0 - 1.0/n))

def shannon_entropy(c: np.ndarray, base: float = np.e) -> float:
    """
    Shannon entropy H = -sum p_i log p_i (with log in the given base).
    Zeros are handled with the usual 0*log0 -> 0 convention.
    """
    p = _probs(c)
    # mask zeros to avoid log issues
    mask = p > 0
    logp = np.log(p[mask])
    if base != np.e:
        logp = logp / np.log(base)
    H = -float(np.dot(p[mask], logp))
    return H

def shannon_sparsity_score(c: np.ndarray, base: float = np.e) -> float:
    """
    Normalized Shannon sparsity score in [0,1]:
        S_Sh = 1 - H / log_base(n)
    Returns 1 for a basis vector, 0 for the uniform vector.
    """
    n = max(1, int(np.size(c)))
    if n == 1:
        return 1.0
    H = shannon_entropy(c, base=base)
    logn = np.log(n) / (np.log(base) if base != np.e else 1.0)
    return float(1.0 - H / logn)

def top_k_coverage(c: np.ndarray, alpha: float = 0.9) -> Tuple[int, float]:
    """
    Smallest K such that sum of the K largest p_i >= alpha.
    Returns (K, K/n). If alpha <= 0, returns (0, 0.0). If alpha > 1, clamps to 1.
    """
    p = _probs(c)
    n = int(p.size)
    if n == 0:
        return 0, 0.0
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0:
        return 0, 0.0
    ps = np.sort(p)[::-1]
    cs = np.cumsum(ps)
    K = int(np.searchsorted(cs, a, side="left") + 1)
    K = min(K, n)
    return K, K / n

def epsilon_support_size(
    c: np.ndarray,
    epsilon: float,
    mode: Literal["amplitude", "prob"] = "amplitude"
) -> int:
    """
    Count of indices above a threshold:
      - mode="amplitude": |c_i| >= epsilon     (threshold on amplitudes)
      - mode="prob":      p_i   >= epsilon     (threshold on probabilities)
    """
    if mode == "amplitude":
        return int(np.count_nonzero(np.abs(c) >= epsilon))
    elif mode == "prob":
        p = _probs(c)
        return int(np.count_nonzero(p >= epsilon))
    else:
        raise ValueError("mode must be 'amplitude' or 'prob'")



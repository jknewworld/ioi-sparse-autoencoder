"""Choose a feature, find similar features, and measure score changes."""
import numpy as np
from sklearn.metrics import roc_auc_score


def select_feature(dev_features, dev_rows, min_active=0.03, max_active=0.97):
    """Use the development set to find a feature that tracks name order."""
    y = np.asarray([e['io_first'] for e in dev_rows], dtype=int)
    freq = (dev_features > 0).mean(axis=0)
    a, b = dev_features[y == 1], dev_features[y == 0]
    diff = a.mean(axis=0) - b.mean(axis=0)
    score = np.abs(diff) / np.maximum(dev_features.std(axis=0), 1e-4)
    valid = (freq >= min_active) & (freq <= max_active)
    if not valid.any():
        raise ValueError('No feature was active often enough within the allowed range. Try more data or training steps.')
    top = int(np.argmax(np.where(valid, score, -np.inf)))
    sign = 1 if diff[top] > 0 else -1
    return top, sign, score, freq


def heldout_auc(z, rows, feature, sign):
    y = [e['io_first'] for e in rows]
    return float(roc_auc_score(y, z[:, feature] * sign))


def bootstrap_ci(values, seed=42, n_boot=1000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (n_boot, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def matched_controls(dev_features, selected, score, freq, n=12, seed=42):
    """Match how often and how strongly features respond. Skip the top 5% for name order."""
    rng = np.random.default_rng(seed)
    sizes = np.mean(dev_features, axis=0)
    active = (freq >= 0.03) & (freq <= 0.97)
    cutoff = np.quantile(score[active], 0.95)
    candidates = np.where(active & (score < cutoff))[0]
    candidates = candidates[candidates != selected]
    if not len(candidates):
        raise ValueError('No features are available for comparison.')
    target_freq = freq[selected]
    target_size = sizes[selected]
    distances = (freq[candidates] - target_freq) ** 2 + (
        (sizes[candidates] - target_size) / (sizes.std() + 1e-6)) ** 2
    shortlist = candidates[np.argsort(distances)[:max(n * 4, n)]]
    return rng.choice(shortlist, size=min(n, len(shortlist)), replace=False).tolist()

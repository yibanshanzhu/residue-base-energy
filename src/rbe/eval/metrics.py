from __future__ import annotations

import numpy as np

from rbe.data.pwm import normalize_pwm


def pwm_mae(
    target: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    target = normalize_pwm(target)
    pred = normalize_pwm(pred)
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        target = target[valid]
        pred = pred[valid]
    per_position_l1 = np.sum(np.abs(target - pred), axis=1)
    return float(np.mean(per_position_l1))


def pwm_kl(target: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    target = normalize_pwm(target, eps=eps)
    pred = normalize_pwm(pred, eps=eps)
    return float(np.mean(np.sum(target * (np.log(target) - np.log(pred)), axis=1)))


def information_content(pwm: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    pwm = normalize_pwm(pwm, eps=eps)
    return 2.0 + np.sum(pwm * np.log2(pwm), axis=1)


def weighted_pcc(target: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    x = normalize_pwm(target).reshape(-1)
    y = normalize_pwm(pred).reshape(-1)
    w = np.repeat(weights.astype(np.float64), 4)
    if np.all(w == 0):
        w = np.ones_like(w)
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    denom = np.sqrt(vx * vy)
    return float(cov / denom) if denom > 0 else 0.0


def reverse_complement_pwm(pwm: np.ndarray) -> np.ndarray:
    pwm = normalize_pwm(pwm)
    return pwm[::-1][:, [3, 2, 1, 0]]


def pwm_metrics(target: np.ndarray, pred: np.ndarray) -> dict:
    weights = information_content(target)
    direct_kl = pwm_kl(target, pred)
    rc_kl = pwm_kl(target, reverse_complement_pwm(pred))
    return {
        "mae": pwm_mae(target, pred),
        "kl": direct_kl,
        "ic_pcc": weighted_pcc(target, pred, weights),
        "rc_aware_kl": min(direct_kl, rc_kl),
    }


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.reshape(-1).astype(np.int64)
    y_score = y_score.reshape(-1).astype(np.float64)
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    rank = np.arange(1, len(y_sorted) + 1)
    precision = tp / rank
    return float(np.sum(precision * y_sorted) / positives)


def top_l_precision(y_true: np.ndarray, y_score: np.ndarray, top_l: int) -> float:
    y_true = y_true.reshape(-1).astype(np.int64)
    y_score = y_score.reshape(-1).astype(np.float64)
    top_l = max(1, min(int(top_l), y_true.size))
    order = np.argsort(-y_score)[:top_l]
    return float(y_true[order].mean())


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = y_true.reshape(-1).astype(np.int64)
    y_score = y_score.reshape(-1).astype(np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return {
        "ap": average_precision(y_true, y_score),
        "mcc": float(mcc),
        "f1": float(f1),
    }


def best_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = y_true.reshape(-1).astype(np.int64)
    y_score = y_score.reshape(-1).astype(np.float64)
    thresholds = np.unique(np.concatenate([y_score, np.asarray([0.5])]))

    best_f1 = -1.0
    best_f1_threshold = 0.5
    best_mcc = -1.0
    best_mcc_threshold = 0.5

    for threshold in thresholds:
        metrics = binary_metrics(y_true, y_score, threshold=float(threshold))
        f1 = metrics["f1"]
        mcc = metrics["mcc"]
        if f1 > best_f1 or (
            np.isclose(f1, best_f1)
            and abs(float(threshold) - 0.5) < abs(best_f1_threshold - 0.5)
        ):
            best_f1 = f1
            best_f1_threshold = float(threshold)
        if mcc > best_mcc or (
            np.isclose(mcc, best_mcc)
            and abs(float(threshold) - 0.5) < abs(best_mcc_threshold - 0.5)
        ):
            best_mcc = mcc
            best_mcc_threshold = float(threshold)

    at_05 = binary_metrics(y_true, y_score, threshold=0.5)
    return {
        "ap": at_05["ap"],
        "f1_at_0.5": at_05["f1"],
        "mcc_at_0.5": at_05["mcc"],
        "best_f1_diagnostic": float(best_f1),
        "best_f1_threshold_diagnostic": best_f1_threshold,
        "best_mcc_diagnostic": float(best_mcc),
        "best_mcc_threshold_diagnostic": best_mcc_threshold,
    }

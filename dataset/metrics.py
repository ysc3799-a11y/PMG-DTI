from typing import Dict, Iterable, Tuple

import numpy as np
import torch

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:
    average_precision_score = None
    roc_auc_score = None


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    z = np.clip(logits.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def gather_outputs(logits_list: Iterable[torch.Tensor], labels_list: Iterable[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    logits = torch.cat(list(logits_list), dim=0).view(-1)
    labels = torch.cat(list(labels_list), dim=0).view(-1)
    return _to_numpy(labels).astype(np.float32), _to_numpy(logits).astype(np.float32)


def _confusion(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5):
    y = (y_true >= 0.5).astype(np.int64)
    p = (y_prob >= float(thr)).astype(np.int64)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return tp, tn, fp, fn


def _metrics_from_conf(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    eps = 1e-12
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, eps)
    bacc = 0.5 * (recall + specificity)
    mcc_den = max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0)
    mcc = ((tp * tn - fp * fn) / np.sqrt(mcc_den + eps)) if mcc_den > 0 else 0.0
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "bacc": float(bacc),
        "mcc": float(mcc),
    }


def compute_metrics(y_true, y_pred, threshold: float = 0.5) -> Dict[str, float]:
    y_true = _to_numpy(y_true).astype(np.float32).reshape(-1)
    y_pred = _to_numpy(y_pred).astype(np.float32).reshape(-1)
    y_prob = _sigmoid_np(y_pred)

    out = _metrics_from_conf(*_confusion(y_true, y_prob, threshold))
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob)) if roc_auc_score is not None else float("nan")
    except Exception:
        out["auc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_prob)) if average_precision_score is not None else float("nan")
    except Exception:
        out["auprc"] = float("nan")

    # compatibility aliases
    out["auroc"] = out["auc"]
    out["aupr"] = out["auprc"]
    return out


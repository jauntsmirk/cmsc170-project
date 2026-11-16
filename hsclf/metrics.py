"""Evaluation metrics written from scratch with numpy only."""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    """cm[i, j] = number of samples with true class i predicted as class j."""
    idx = np.asarray(y_true, dtype=np.int64) * n_classes + np.asarray(y_pred, dtype=np.int64)
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class_prf(cm):
    """Precision, recall, F1 and support per class (0 where undefined)."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    return prec, rec, f1, support


def macro_f1(cm):
    """Unweighted mean F1 over the classes that actually occur in y_true (support > 0)."""
    _, _, f1, support = per_class_prf(cm)
    return float(f1[support > 0].mean())


def weighted_f1(cm):
    _, _, f1, support = per_class_prf(cm)
    return float((f1 * support).sum() / support.sum())


def top_k_accuracy(y_true, scores, k=3):
    """Share of samples whose true class is among the k highest-scoring classes."""
    top = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    return float(np.mean((top == np.asarray(y_true)[:, None]).any(axis=1)))


def value_weighted_error(y_true, y_pred, value):
    """Share of total dutiable value (PHP) carried by misclassified rows."""
    value = np.nan_to_num(np.asarray(value, dtype=np.float64))
    total = value.sum()
    return float(value[np.asarray(y_true) != np.asarray(y_pred)].sum() / total) if total > 0 else float("nan")


def evaluate(y_true, y_pred, n_classes, scores=None, value=None):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    out = {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(cm), "weighted_f1": weighted_f1(cm)}
    if scores is not None:
        out["top3_accuracy"] = top_k_accuracy(y_true, scores, 3)
    if value is not None:
        out["value_weighted_error"] = value_weighted_error(y_true, y_pred, value)
    return out, cm

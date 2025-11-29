# bench/numeric_eval.py
import numpy as np


def compute_nmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    numerator = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return float("inf")
    return float(numerator / denominator)


def compute_acc01(y_true, y_pred, tol=0.1):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    rel = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)
    return float(np.mean(rel < tol))

# -*- coding: utf-8 -*-
"""Addressee classifier (v1): a small L2-regularized logistic regression over interpretable
features (numpy, no sklearn dep). Predicts P(is_for_agent) per utterance. Save/load via JSON.

Swap for a fine-tuned small LM later (v2) — the interface (fit / predict_proba) stays the same.
"""
from __future__ import annotations

import json
from typing import List

import numpy as np


class AddresseeClassifier:
    def __init__(self, l2: float = 1.0):
        self.w = None
        self.l2 = l2
        self.mu = None
        self.sd = None

    def _norm(self, X):
        return (X - self.mu) / self.sd

    def fit(self, X: List[List[float]], y: List[int], epochs: int = 500, lr: float = 0.2):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        self.mu[-1] = 0.0; self.sd[-1] = 1.0            # keep bias column intact
        Xn = self._norm(X)
        n, d = Xn.shape
        self.w = np.zeros(d)
        # class weight to counter imbalance (addressed-to-agent is the minority)
        pos = max(1.0, y.sum()); neg = max(1.0, n - y.sum())
        cw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-Xn @ self.w))
            g = Xn.T @ ((p - y) * cw) / n + self.l2 * np.r_[self.w[:-1], 0.0] / n
            self.w -= lr * g
        return self

    def predict_proba(self, X: List[List[float]]) -> np.ndarray:
        Xn = self._norm(np.asarray(X, float))
        return 1 / (1 + np.exp(-Xn @ self.w))

    def predict(self, X, thresh: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= thresh).astype(int)

    def save(self, path: str):
        json.dump({"w": self.w.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                   "l2": self.l2}, open(path, "w"))

    @staticmethod
    def load(path: str) -> "AddresseeClassifier":
        d = json.load(open(path))
        c = AddresseeClassifier(d.get("l2", 1.0))
        c.w = np.array(d["w"]); c.mu = np.array(d["mu"]); c.sd = np.array(d["sd"])
        return c

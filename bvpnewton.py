"""Модуль решения краевых задач (BVP) методом Ньютона без метода стрельбы.

Подключение:
    from bvpnewton import solve_bvp_newton, build_linear_guess
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np


ArrayLike = np.ndarray


@dataclass
class NewtonResult:
    x: np.ndarray
    y: np.ndarray
    iterations: int
    residual_norm: float
    converged: bool


def _as_2d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values[:, None]
    return values


def build_linear_guess(
    a: float,
    b: float,
    y_a: Iterable[float],
    y_b: Iterable[float],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Линейная начальная аппроксимация между граничными значениями."""
    x = np.linspace(a, b, n + 1)
    y_a = np.asarray(list(y_a), dtype=float)
    y_b = np.asarray(list(y_b), dtype=float)
    y = np.zeros((n + 1, y_a.size), dtype=float)
    for i, xi in enumerate(x):
        t = (xi - a) / (b - a)
        y[i] = (1 - t) * y_a + t * y_b
    return x, y


def solve_bvp_newton(
    f: Callable[[float, np.ndarray], np.ndarray],
    bc: Callable[[np.ndarray, np.ndarray], np.ndarray],
    a: float,
    b: float,
    y_guess: np.ndarray | Callable[[np.ndarray], np.ndarray],
    n: int,
    tol: float = 1e-8,
    max_iter: int = 20,
    jac_eps: float = 1e-6,
) -> NewtonResult:
    """Решает систему краевой задачи методом Ньютона на сетке.

    Решаемая форма:
        y'(x) = f(x, y), x in [a, b]
        bc(y(a), y(b)) = 0

    bc должна возвращать вектор длины m (m - размерность системы).
    """
    if callable(y_guess):
        x = np.linspace(a, b, n + 1)
        y0 = y_guess(x)
    else:
        x = np.linspace(a, b, n + 1)
        y0 = y_guess

    y = _as_2d(np.asarray(y0, dtype=float))
    if y.shape[0] != n + 1:
        raise ValueError("y_guess должен иметь форму (n+1, m).")

    m = y.shape[1]
    h = (b - a) / n

    def residual(y_flat: np.ndarray) -> np.ndarray:
        y_local = y_flat.reshape(n + 1, m)
        res = []
        bc_res = bc(y_local[0], y_local[-1])
        bc_res = np.asarray(bc_res, dtype=float)
        if bc_res.size != m:
            raise ValueError("bc должна возвращать m условий.")
        res.append(bc_res)
        for i in range(1, n):
            diff = (y_local[i + 1] - y_local[i - 1]) / (2 * h)
            res.append(diff - f(x[i], y_local[i]))
        return np.concatenate(res)

    def jacobian(y_flat: np.ndarray, r0: np.ndarray) -> np.ndarray:
        size = y_flat.size
        J = np.zeros((size, size), dtype=float)
        for i in range(size):
            y_step = y_flat.copy()
            y_step[i] += jac_eps
            r1 = residual(y_step)
            J[:, i] = (r1 - r0) / jac_eps
        return J

    y_flat = y.reshape(-1)
    converged = False
    res_norm = np.inf

    for it in range(1, max_iter + 1):
        r0 = residual(y_flat)
        res_norm = float(np.linalg.norm(r0, ord=np.inf))
        if res_norm < tol:
            converged = True
            break
        J = jacobian(y_flat, r0)
        step = np.linalg.solve(J, -r0)

        damping = 1.0
        for _ in range(10):
            candidate = y_flat + damping * step
            r_candidate = residual(candidate)
            if np.linalg.norm(r_candidate, ord=np.inf) < res_norm:
                y_flat = candidate
                break
            damping *= 0.5
        else:
            y_flat = y_flat + step

    y = y_flat.reshape(n + 1, m)
    return NewtonResult(
        x=x,
        y=y,
        iterations=it,
        residual_norm=res_norm,
        converged=converged,
    )

from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional, Tuple, Union


class AdaptiveUKF:
    """
    Production-grade Scaled Unscented Kalman Filter (UKF) for 2D kinematic tracking:
      State: x = [pos_x, pos_y, vel_x, vel_y]^T (m, m/s)
    Features:
      - Scaled unscented transform (van der Merwe / Julier-Uhlmann formulation)
      - Positive semi-definite (PSD) repair on covariance via eigenvalue floor
      - Innovation gating via chi-square / Mahalanobis threshold (nu^T * S^-1 * nu)
      - Joseph-style stabilized covariance update (P - K * S * K^T)
      - Support for both general non-linear h(x) and multi-anchor range vectors
    """

    def __init__(
        self,
        n: int = 4,
        x0: Optional[Union[List[float], np.ndarray]] = None,
        P0: Optional[np.ndarray] = None,
        alpha: float = 0.5,
        beta: float = 2.0,
        kappa: float = 0.0,
        R0: float = 0.03**2,
        Q: float = 0.05,
    ) -> None:
        self.n = int(n)
        self.x = np.array(
            x0 if x0 is not None else [0.50, 0.35, 0.0, 0.0][: self.n],
            dtype=float,
        )
        if P0 is not None:
            self.P = np.array(P0, dtype=float)
        else:
            self.P = np.eye(self.n) * (0.05**2)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.R0 = float(R0)
        self.q_scale = float(Q)

        # Scaled Unscented Transform parameters
        self.lambda_ = (self.alpha**2) * (self.n + self.kappa) - self.n
        self.c = self.n + self.lambda_

        # 2n + 1 sigma point weights
        self.wm = np.full(2 * self.n + 1, 1.0 / (2.0 * self.c))
        self.wc = np.full(2 * self.n + 1, 1.0 / (2.0 * self.c))
        self.wm[0] = self.lambda_ / self.c
        self.wc[0] = self.lambda_ / self.c + (1.0 - self.alpha**2 + self.beta)

    def predict(self, dt: float = 0.025) -> None:
        """Propagates state and covariance forward using constant-velocity motion model."""
        dt = max(1e-4, float(dt))
        F = np.eye(self.n)
        if self.n >= 4:
            F[0, 2] = dt
            F[1, 3] = dt

        # Piecewise continuous white noise acceleration model for Q
        if self.n == 4:
            G = np.array([[0.5 * dt**2, 0.0], [0.0, 0.5 * dt**2], [dt, 0.0], [0.0, dt]])
            Q_mat = G @ (np.eye(2) * (self.q_scale**2)) @ G.T
        else:
            Q_mat = np.eye(self.n) * ((self.q_scale * dt) ** 2)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q_mat
        self.P = 0.5 * (self.P + self.P.T) + np.eye(self.n) * 1e-9

    def generate_sigma_points(self) -> np.ndarray:
        """Generates 2n+1 sigma points with numerical Cholesky PSD recovery."""
        sigma = np.zeros((2 * self.n + 1, self.n))
        sigma[0] = self.x

        P_sym = 0.5 * (self.P + self.P.T)
        try:
            L = np.linalg.cholesky(self.c * P_sym)
        except np.linalg.LinAlgError:
            # Reconstruct PSD matrix using eigenvalue floor
            eigvals, eigvecs = np.linalg.eigh(P_sym)
            eigvals = np.maximum(eigvals, 1e-8)
            P_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(self.c * P_psd)

        for i in range(self.n):
            sigma[i + 1] = self.x + L[:, i]
            sigma[self.n + i + 1] = self.x - L[:, i]

        return sigma

    def update_ranges(
        self,
        anchors: Union[List[List[float]], np.ndarray],
        ranges: Union[List[float], np.ndarray],
        R_diag: Optional[Union[List[float], np.ndarray]] = None,
        gate_chi2: float = 11.34,
    ) -> bool:
        """
        Updates filter using multi-anchor direct range measurements:
          z_i = ||pos - anchor_i|| + v_i
        """
        try:
            anchors_np = np.asarray(anchors, dtype=float)
            z = np.asarray(ranges, dtype=float).flatten()
        except (TypeError, ValueError, OverflowError):
            return False
        m = len(z)
        if (
            anchors_np.ndim != 2
            or anchors_np.shape[1] != 2
            or m == 0
            or len(anchors_np) != m
            or not np.all(np.isfinite(anchors_np))
            or not np.all(np.isfinite(z))
            or not np.isfinite(gate_chi2)
        ):
            return False

        if R_diag is None:
            R = np.eye(m) * self.R0
        else:
            try:
                r_diag = np.asarray(R_diag, dtype=float).flatten()
            except (TypeError, ValueError, OverflowError):
                return False
            if (
                r_diag.size != m
                or not np.all(np.isfinite(r_diag))
                or np.any(r_diag < 0.0)
            ):
                return False
            R = np.diag(r_diag**2)

        sigmas = self.generate_sigma_points()
        Z_sig = np.zeros((2 * self.n + 1, m))
        for k in range(2 * self.n + 1):
            pos = sigmas[k, :2]
            for j in range(m):
                Z_sig[k, j] = np.linalg.norm(pos - anchors_np[j])

        z_hat = np.sum(self.wm[:, None] * Z_sig, axis=0)
        Z_diff = Z_sig - z_hat[None, :]

        Pzz = np.zeros((m, m))
        for k in range(2 * self.n + 1):
            Pzz += self.wc[k] * np.outer(Z_diff[k], Z_diff[k])
        S = Pzz + R

        X_diff = sigmas - self.x[None, :]
        Pxz = np.zeros((self.n, m))
        for k in range(2 * self.n + 1):
            Pxz += self.wc[k] * np.outer(X_diff[k], Z_diff[k])

        nu = z - z_hat
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(nu.T @ S_inv @ nu)
            if d2 > gate_chi2:
                # Outlier rejected
                return False

            K = Pxz @ S_inv
            self.x = self.x + K @ nu
            self.P = self.P - K @ S @ K.T
            self.P = 0.5 * (self.P + self.P.T) + np.eye(self.n) * 1e-9
            return True
        except np.linalg.LinAlgError:
            return False

    def update(
        self,
        z: Union[float, List[float], np.ndarray],
        h: Callable[[np.ndarray], Union[float, np.ndarray]],
        R_val: Optional[Union[float, np.ndarray]] = None,
        gate_chi2: float = 9.21,
    ) -> bool:
        """
        General measurement update for non-linear measurement model h(x).
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        m = len(z_arr)

        if R_val is None:
            R = np.eye(m) * self.R0
        elif np.isscalar(R_val):
            R = np.eye(m) * float(R_val)
        else:
            R = np.atleast_2d(np.asarray(R_val, dtype=float))

        sigmas = self.generate_sigma_points()
        Z_sig = np.zeros((2 * self.n + 1, m))
        for k in range(2 * self.n + 1):
            eval_h = np.atleast_1d(h(sigmas[k]))
            Z_sig[k] = eval_h

        z_hat = np.sum(self.wm[:, None] * Z_sig, axis=0)
        Z_diff = Z_sig - z_hat[None, :]

        Pzz = np.zeros((m, m))
        for k in range(2 * self.n + 1):
            Pzz += self.wc[k] * np.outer(Z_diff[k], Z_diff[k])
        S = Pzz + R

        X_diff = sigmas - self.x[None, :]
        Pxz = np.zeros((self.n, m))
        for k in range(2 * self.n + 1):
            Pxz += self.wc[k] * np.outer(X_diff[k], Z_diff[k])

        nu = z_arr - z_hat
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(nu.T @ S_inv @ nu)
            if d2 > gate_chi2:
                return False

            K = Pxz @ S_inv
            self.x = self.x + K @ nu
            self.P = self.P - K @ S @ K.T
            self.P = 0.5 * (self.P + self.P.T) + np.eye(self.n) * 1e-9
            return True
        except np.linalg.LinAlgError:
            return False


# Backward compatibility alias
RobustUKF = AdaptiveUKF

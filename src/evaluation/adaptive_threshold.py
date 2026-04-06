import numpy as np
from typing import Dict, List, Tuple, Optional


class AdaptiveThresholdEstimator:
    """
    Per-user adaptive authentication threshold.

    Computes T_user = μ_user + k·σ_user where:
      - μ_user = mean distance from enrolment embeddings to their centroid
      - σ_user = std of those distances
      - k      = hyperparameter, optimized on validation set via two-stage grid search

    Consistent users get smaller T_user (tighter security).
    Inconsistent users get larger T_user (fewer false rejects).
    """

    def __init__(self, k: float = 2.0, sigma_floor: float = 0.0):
        """
        Args:
            k:           Threshold multiplier. Optimized on validation set.
            sigma_floor: Minimum σ_user value. When E=1, σ_user=0 so the
                         threshold degrades to T=μ_user. Setting a floor
                         derived from the validation population prevents this.
                         If 0.0, no floor is applied (E=1 documented as edge case).
        """
        self.k = k
        self.sigma_floor = sigma_floor

    def estimate_user_threshold(
        self, enrolment_embeddings: np.ndarray
    ) -> Dict:
        """
        Compute per-user threshold from enrolment embeddings.

        Args:
            enrolment_embeddings: shape (E, 64)

        Returns:
            Template dict: {z_mean, mu_user, sigma_user, T_user, E}
        """
        E = enrolment_embeddings.shape[0]
        assert E >= 1, "Need at least 1 enrolment session"

        # Centroid
        z_mean = enrolment_embeddings.mean(axis=0)  # (64,)

        # Distances from each enrolment embedding to centroid
        dists = np.linalg.norm(
            enrolment_embeddings - z_mean, axis=1
        )  # (E,)

        mu_user = float(dists.mean())
        sigma_user = float(dists.std()) if E > 1 else 0.0

        # Apply sigma floor (handles E=1 edge case)
        effective_sigma = max(sigma_user, self.sigma_floor)

        T_user = mu_user + self.k * effective_sigma

        return {
            "z_mean": z_mean,
            "mu_user": mu_user,
            "sigma_user": sigma_user,
            "sigma_effective": effective_sigma,
            "T_user": T_user,
            "E": E,
        }

    def predict(
        self, z_test: np.ndarray, template: Dict
    ) -> Tuple[str, float]:
        """
        Make authentication decision for a test embedding.

        Returns:
            (decision, distance)
        """
        d = float(np.linalg.norm(z_test - template["z_mean"]))
        decision = "accept" if d <= template["T_user"] else "reject"
        return decision, d

    def optimize_k(
        self,
        val_data: Dict[str, Dict],
        k_range: Optional[np.ndarray] = None,
        refine: bool = True,
    ) -> Tuple[float, float]:
        """
        Two-stage grid search for optimal k on validation set.

        Stage 1: Coarse search with 0.25 step over [0.5, 4.0]
        Stage 2: Fine search with 0.05 step around ±0.25 of best coarse k

        Args:
            val_data: {user_id: {'enrol': (E,64), 'genuine': (5,64), 'impostor': (N,64)}}
            k_range:  Override for stage 1 range. If None, uses default.
            refine:   Whether to do stage 2 refinement.

        Returns:
            (best_k, best_eer)
        """
        # Stage 1: Coarse search
        if k_range is None:
            k_range = np.arange(0.5, 4.25, 0.25)

        best_k, best_eer = self._grid_search(val_data, k_range, stage_label="Coarse")

        # Stage 2: Fine refinement around best coarse k
        if refine:
            fine_range = np.arange(
                max(0.05, best_k - 0.25),
                best_k + 0.30,
                0.05,
            )
            best_k, best_eer = self._grid_search(
                val_data, fine_range, stage_label="Fine"
            )

        self.k = best_k
        return best_k, best_eer

    def _grid_search(
        self,
        val_data: Dict[str, Dict],
        k_range: np.ndarray,
        stage_label: str = "",
    ) -> Tuple[float, float]:
        """Run grid search over k_range and return (best_k, best_eer)."""
        from src.evaluation.metrics import compute_eer

        best_k = self.k
        best_eer = 1.0
        results = []

        for k_cand in k_range:
            self.k = k_cand
            user_eers = []

            for user_id, data in val_data.items():
                template = self.estimate_user_threshold(data["enrol"])

                genuine_scores = [
                    float(np.linalg.norm(z - template["z_mean"]))
                    for z in data["genuine"]
                ]
                impostor_scores = [
                    float(np.linalg.norm(z - template["z_mean"]))
                    for z in data["impostor"]
                ]

                user_eer = compute_eer(genuine_scores, impostor_scores)
                user_eers.append(user_eer)

            avg_eer = float(np.mean(user_eers))
            results.append((k_cand, avg_eer))

            if avg_eer < best_eer:
                best_eer = avg_eer
                best_k = k_cand

        self.k = best_k

        print(f"\n  [{stage_label} search] k optimization (best k={best_k:.2f}, EER={best_eer*100:.4f}%):")
        for k_val, eer in results:
            marker = " ◀" if abs(k_val - best_k) < 1e-6 else ""
            print(f"    k={k_val:.2f}: EER={eer*100:.4f}%{marker}")

        return best_k, best_eer

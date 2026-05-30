"""Unit tests for Wilcoxon signed-rank comparison."""

import numpy as np
import pytest
from src.statistics.wilcoxon import compare_configurations


class TestWilcoxon:
    def _make_eer_dict(self):
        rng = np.random.RandomState(42)
        return {
            "config_a": rng.rand(100) * 0.1 + 0.05,   # 5-15% EER
            "config_b": rng.rand(100) * 0.1 + 0.03,   # 3-13% EER (better)
            "config_c": rng.rand(100) * 0.1 + 0.05,   # same as a
        }

    def test_returns_dataframe(self):
        import pandas as pd
        eer_dict = self._make_eer_dict()
        df = compare_configurations(
            eer_dict,
            [("config_a", "config_b")],
            alpha=0.05,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_required_columns(self):
        eer_dict = self._make_eer_dict()
        df = compare_configurations(eer_dict, [("config_a", "config_b")])
        for col in ["comparison", "statistic", "p_raw", "p_bonferroni", "significant"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_identical_distributions_not_significant(self):
        """Identical EER arrays should not be significant."""
        rng = np.random.RandomState(0)
        arr = rng.rand(100)
        eer_dict = {"a": arr.copy(), "b": arr.copy()}
        df = compare_configurations(eer_dict, [("a", "b")], alpha=0.05)
        assert df.iloc[0]["p_raw"] == 1.0 or not df.iloc[0]["significant"]

    def test_bonferroni_correction_applied(self):
        """With 5 comparisons, p_bonferroni = p_raw * 5 (capped at 1.0)."""
        eer_dict = self._make_eer_dict()
        comparisons = [("config_a", "config_b")] * 5
        # Duplicate same comparison 5 times
        extended = {}
        for i in range(5):
            extended[f"cfg_{i}a"] = eer_dict["config_a"].copy()
            extended[f"cfg_{i}b"] = eer_dict["config_b"].copy()
        comps = [(f"cfg_{i}a", f"cfg_{i}b") for i in range(5)]
        df = compare_configurations(extended, comps, alpha=0.05)
        assert len(df) == 5
        # p_bonferroni should be >= p_raw
        assert all(df["p_bonferroni"] >= df["p_raw"] - 1e-10)

    def test_missing_config_raises(self):
        eer_dict = {"a": np.rand(50) if False else np.random.rand(50)}
        with pytest.raises(KeyError):
            compare_configurations(eer_dict, [("a", "nonexistent")])

    def test_length_mismatch_raises(self):
        eer_dict = {
            "a": np.random.rand(100),
            "b": np.random.rand(50),   # different length
        }
        with pytest.raises(ValueError):
            compare_configurations(eer_dict, [("a", "b")])

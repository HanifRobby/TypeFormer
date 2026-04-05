import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.kdprint_preprocess import KDPrintPreprocessor
from src.preprocessing.typeformer_preprocess import typeformer_preprocess

def make_fake_session(hold_mu=100, hold_sd=15, inter_mu=160, inter_sd=30,
                       seq_len=50, seed=0):
    rng = np.random.default_rng(seed)
    hold    = np.clip(rng.normal(hold_mu, hold_sd, seq_len), 20, 500)
    inter   = np.clip(rng.normal(inter_mu, inter_sd, seq_len), 5, 800)
    press   = hold + inter + rng.normal(0, 8, seq_len)
    release = press + rng.normal(0, 5, seq_len)
    ascii_  = rng.integers(65, 123, seq_len).astype(float) / 255.0
    return np.column_stack([hold, inter, press, release, ascii_])

class TestKDPrintPreprocessor:
    
    def test_fit_produces_correct_shapes(self):
        prep = KDPrintPreprocessor()
        sessions = [make_fake_session(seed=i) for i in range(5)]
        prep.fit(sessions)
        
        assert prep.mu.shape    == (4,), f"mu shape: {prep.mu.shape}"
        assert prep.sigma.shape == (4,), f"sigma shape: {prep.sigma.shape}"
    
    def test_standardize_centers_genuine_data(self):
        prep = KDPrintPreprocessor()
        enrol = [make_fake_session(hold_mu=100, seed=i) for i in range(5)]
        prep.fit(enrol)
        
        test_session = make_fake_session(hold_mu=100, seed=99)
        result = prep.standardize(test_session)
        
        for i in range(4):
            mean = result[:, i].mean()
            std  = result[:, i].std()
            assert abs(mean) < 0.5, f"Feature {i} mean not near 0: {mean:.3f}"
            assert 0.5 < std < 2.0, f"Feature {i} std unexpected: {std:.3f}"
    
    def test_standardize_detects_impostor(self):
        prep = KDPrintPreprocessor()
        enrol_A = [make_fake_session(hold_mu=70, inter_mu=90, seed=i) for i in range(5)]
        prep.fit(enrol_A)
        
        impostor_session = make_fake_session(hold_mu=145, inter_mu=280, seed=99)
        result = prep.standardize(impostor_session)
        
        impostor_hold_mean = result[:, 0].mean()
        assert impostor_hold_mean > 3.0, f"Impostor should be >3σ away, got {impostor_hold_mean:.2f}σ"
    
    def test_buffer_reduces_variance(self):
        prep = KDPrintPreprocessor(buffer_size=5)
        sessions = [make_fake_session(seed=i) for i in range(3)]
        processed = prep.fit_transform(sessions, use_buffer=False)
        buffered  = prep.apply_buffer(processed)
        
        for i, (proc, buff) in enumerate(zip(processed, buffered)):
            for feat_idx in range(4):
                std_before = proc[:, feat_idx].std()
                std_after  = buff[:, feat_idx].std()
                # A buffer inherently reduces variance. Give slight margin for artifacts.
                assert std_after <= std_before * 1.1, "Buffer did not reduce variance expectedly"
    
    def test_transform_uses_enrolment_stats(self):
        prep = KDPrintPreprocessor()
        enrol = [make_fake_session(hold_mu=70, seed=i) for i in range(3)]
        prep.fit(enrol)
        
        mu_before = prep.mu.copy()
        test = make_fake_session(hold_mu=200, seed=99)
        _ = prep.transform(test)
        
        np.testing.assert_array_equal(prep.mu, mu_before, err_msg="transform() must not modify mu")
    
    def test_ascii_normalization_preserved(self):
        prep = KDPrintPreprocessor()
        sessions = [make_fake_session(seed=i) for i in range(3)]
        processed = prep.fit_transform(sessions)
        
        for s in processed:
            assert np.all(s[:, 4] >= 0) and np.all(s[:, 4] <= 1), "ASCII values out of [0, 1] range"

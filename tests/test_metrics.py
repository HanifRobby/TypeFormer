import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_eer, find_eer_threshold

def test_compute_eer_perfect_separation():
    genuine = [0.1, 0.2, 0.3]
    impostor = [0.7, 0.8, 0.9]
    eer = compute_eer(genuine, impostor)
    assert eer == 0.0, f"Expected 0.0 EER for perfect separation, got {eer}"

def test_compute_eer_total_overlap():
    genuine = [0.1, 0.2, 0.3, 0.4, 0.5]
    impostor = [0.1, 0.2, 0.3, 0.4, 0.5]
    eer = compute_eer(genuine, impostor)
    # With identical distributions, EER should be around 50%
    assert 0.4 <= eer <= 0.6, f"Expected ~0.5 EER for total overlap, got {eer}"

def test_find_eer_threshold():
    genuine = [0.1, 0.2, 0.3]
    impostor = [0.7, 0.8, 0.9]
    threshold = find_eer_threshold(genuine, impostor)
    # Threshold should sit somewhere between 0.3 and 0.7
    assert 0.3 <= threshold <= 0.7, f"Threshold should separate classes, got {threshold}"

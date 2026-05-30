"""Unit tests for FiLM head (film_head.py)."""

import torch
import numpy as np
import pytest
from src.models.film_head import FiLMHead


D, S = 64, 20


class TestFiLMHeadIdentityInit:
    def test_fresh_output_equals_input(self):
        """At identity init, FiLM(e, s_u) == e for any e, s_u."""
        film = FiLMHead(stats_dim=S, embedding_dim=D)
        film.eval()
        torch.manual_seed(0)
        e = torch.randn(4, D)
        s = torch.randn(4, S)
        with torch.no_grad():
            out = film(e, s)
        assert torch.allclose(out, e, atol=1e-6), \
            f"Identity init failed: max_diff={( out - e).abs().max():.2e}"

    def test_gamma_one_beta_zero(self):
        """fc2 output at identity init: first D elements = 1, last D = 0."""
        film = FiLMHead(stats_dim=S, embedding_dim=D)
        film.eval()
        with torch.no_grad():
            s = torch.randn(2, S)
            gb = film.fc2(torch.relu(film.fc1(s)))
            gamma = gb[:, :D]
            beta = gb[:, D:]
        assert (gamma - 1.0).abs().max() < 1e-6
        assert beta.abs().max() < 1e-6


class TestFiLMHeadForward:
    def test_output_shape(self):
        film = FiLMHead(stats_dim=S, embedding_dim=D)
        e = torch.randn(8, D)
        s = torch.randn(8, S)
        out = film(e, s)
        assert out.shape == (8, D)

    def test_gradient_flow(self):
        film = FiLMHead(stats_dim=S, embedding_dim=D)
        e = torch.randn(4, D)
        s = torch.randn(4, S)
        out = film(e, s)
        loss = out.sum()
        loss.backward()
        for name, p in film.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    def test_dropout_zero_leaves_output_unchanged(self):
        film_no_drop = FiLMHead(stats_dim=S, embedding_dim=D, dropout=0.0)
        film_with_drop = FiLMHead(stats_dim=S, embedding_dim=D, dropout=0.5)
        # Copy same weights
        film_with_drop.load_state_dict(film_no_drop.state_dict())
        film_no_drop.eval()
        film_with_drop.eval()

        torch.manual_seed(0)
        e = torch.randn(4, D)
        s = torch.randn(4, S)
        with torch.no_grad():
            out1 = film_no_drop(e, s)
            out2 = film_with_drop(e, s)
        assert torch.allclose(out1, out2, atol=1e-6)

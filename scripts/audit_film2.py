"""
audit_film2.py — Definitive collapse test for FiLM.

TEST A: Apply ENROLMENT user's s_u to IMPOSTOR probes.
    Under a legitimate model: impostors have different backbone embeddings
        → cosine stays low even with same s_u → EER stays low.
    Under beta-collapse: FiLM(impostor_embedding, s_u_u) → beta(s_u_u)
        → impostor looks like genuine → EER → 50%.

TEST B: Constant s_u (same vector for ALL users).
    Under legitimate model: all users mapped to same cluster → EER → 50%.
    Under beta-collapse: same s_u → same beta → all embeddings same point
        → genuine AND impostor both ≈ 1.0 → EER → 50%.

TEST C: Zero s_u for all users.
    Under legitimate model: gamma≈1, beta≈0 (near identity) → raw cosine EER.
    Under beta-collapse: FiLM with s_u=0 → some fixed beta(0), applied to all
        → all embeddings mapped to same direction → EER → 50%.

Run: conda run -n Typeformer python scripts/audit_film2.py --exp f1_film_only
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.film_head import FiLMHead
from src.statistics.user_stats import compute_user_stats
from src.evaluation.metrics import compute_eer
from src.scoring.cosine_raw import RawCosineScorer


def load_film(exp: str, seed: int) -> FiLMHead:
    ckpt = ROOT / "results" / exp / "checkpoints" / f"film_head_seed{seed}.pt"
    film = FiLMHead(stats_dim=20, embedding_dim=64, hidden_dim=64)
    film.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
    film.eval()
    return film


def apply_film(film: FiLMHead, embs: np.ndarray, s_u: np.ndarray) -> np.ndarray:
    """Apply FiLM to a batch of embeddings with ONE s_u."""
    with torch.no_grad():
        e_t = torch.from_numpy(embs).float()
        s_t = torch.from_numpy(s_u).float().unsqueeze(0).expand(len(embs), -1)
        return film(e_t, s_t).numpy()


def eval_protocol(
    film: FiLMHead,
    embeddings: np.ndarray,   # (N, 15, 64)
    sessions: np.ndarray,     # (N, 15, 50, 5)
    n_users: int,
    mode: str,                # "correct", "impostor_with_enrol_su", "constant_su", "zero_su"
    E: int = 5,
) -> tuple[float, float, np.ndarray]:
    """Evaluate EER under different s_u assignment protocols."""
    scorer = RawCosineScorer()
    N = min(n_users, len(embeddings))

    # Pre-compute s_u for all users
    s_u_dict = {}
    for u in range(N):
        s_u_dict[u] = compute_user_stats(sessions[u, :E])

    # Pre-apply FiLM to all sessions under each mode
    if mode == "constant_su":
        # Same random s_u for every user
        s_u_const = np.random.RandomState(99).randn(20).astype(np.float32)
    elif mode == "zero_su":
        s_u_const = np.zeros(20, dtype=np.float32)

    film_embs = np.zeros((N, 15, 64), dtype=np.float32)
    for u in range(N):
        if mode == "correct":
            s_u = s_u_dict[u]
        elif mode in ("constant_su", "zero_su"):
            s_u = s_u_const
        else:
            s_u = s_u_dict[u]  # for impostor_with_enrol_su, handled below
        film_embs[u] = apply_film(film, embeddings[u], s_u)

    # Evaluate
    per_user_eers = []
    all_genuine, all_impostor = [], []

    for u in range(N):
        e_u = film_embs[u, :E].mean(axis=0)   # enrolment centroid
        genuine_embs = film_embs[u, -5:]       # sessions 10-14

        genuine_scores = [scorer.score(e_u, p) for p in genuine_embs]

        if mode == "impostor_with_enrol_su":
            # Apply ENROLMENT user u's s_u to IMPOSTOR embeddings
            # This is the critical test: under collapse, all → beta(s_u_u) → score ≈ 1.0
            impostor_scores = []
            for j in range(N):
                if j == u:
                    continue
                # Apply user u's s_u to user j's embedding (session 14)
                imp_emb_modulated = apply_film(
                    film, embeddings[j, -1:], s_u_dict[u]
                )
                impostor_scores.append(scorer.score(e_u, imp_emb_modulated[0]))
        else:
            impostor_scores = [
                scorer.score(e_u, film_embs[j, -1])
                for j in range(N) if j != u
            ]

        gen_arr = np.array(genuine_scores)
        imp_arr = np.array(impostor_scores)
        eer, _ = compute_eer(gen_arr, imp_arr)
        per_user_eers.append(eer)
        all_genuine.extend(genuine_scores)
        all_impostor.extend(impostor_scores)

    per_user_eers = np.array(per_user_eers)
    global_eer, _ = compute_eer(np.array(all_genuine), np.array(all_impostor))
    return global_eer, per_user_eers.mean(), per_user_eers


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="f1_film_only")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-users", type=int, default=150)
    args = p.parse_args()

    print(f"\n{'='*65}")
    print(f"DEFINITIVE COLLAPSE TEST: {args.exp}  seed={args.seed}  N={args.n_users}")
    print(f"{'='*65}")

    emb_path = ROOT / "results" / "embeddings_cache" / "test" / "test_embeddings.npy"
    sess_path = ROOT / "data" / "processed" / "test_sessions.npz"
    embeddings = np.load(str(emb_path))[:args.n_users]
    sessions = np.load(str(sess_path))["sessions"][:args.n_users]

    film = load_film(args.exp, args.seed)

    raw_scorer = RawCosineScorer()
    N = args.n_users

    # Raw cosine baseline (no FiLM)
    per_user_raw, gen_r, imp_r = [], [], []
    for u in range(N):
        e_u = embeddings[u, :5].mean(axis=0)
        gs = [raw_scorer.score(e_u, embeddings[u, s]) for s in range(10, 15)]
        is_ = [raw_scorer.score(e_u, embeddings[j, -1]) for j in range(N) if j != u]
        eer, _ = compute_eer(np.array(gs), np.array(is_))
        per_user_raw.append(eer)
        gen_r.extend(gs); imp_r.extend(is_)
    raw_global, _ = compute_eer(np.array(gen_r), np.array(imp_r))
    raw_ps = np.mean(per_user_raw)

    print(f"\n{'─'*65}")
    print(f"BASELINE (raw cosine, no FiLM):")
    print(f"  per_subj={raw_ps*100:.2f}%  global={raw_global*100:.2f}%")
    print(f"{'─'*65}")

    tests = [
        ("correct",                "TEST 0: Correct s_u (standard protocol)"),
        ("impostor_with_enrol_su", "TEST A: Impostor probe with ENROLMENT user's s_u [DEFINITIVE]"),
        ("constant_su",            "TEST B: Constant s_u (same vector for all users)"),
        ("zero_su",                "TEST C: Zero s_u for all users"),
    ]

    predictions = {
        "correct":                "legitimate≈same, collapse≈same (reference)",
        "impostor_with_enrol_su": "legitimate→LOW EER, collapse→EER≈50%",
        "constant_su":            "legitimate→HIGH EER, collapse→EER≈50%",
        "zero_su":                "legitimate→raw cosine EER, collapse→EER≈50%",
    }

    results = {}
    for mode, label in tests:
        print(f"\n{label}")
        print(f"  Prediction: {predictions[mode]}")
        g_eer, ps_eer, pu = eval_protocol(film, embeddings, sessions, N, mode)
        results[mode] = (g_eer, ps_eer)
        n_zero = int((pu == 0).sum())
        print(f"  global={g_eer*100:.2f}%  per_subj={ps_eer*100:.2f}%  "
              f"EER=0%: {n_zero}/{N} users")

    # Interpretation
    print(f"\n{'='*65}")
    print("INTERPRETATION")
    print(f"{'='*65}")

    ref_global = results["correct"][0]
    test_a_global = results["impostor_with_enrol_su"][0]
    test_b_global = results["constant_su"][0]
    test_c_global = results["zero_su"][0]

    print(f"\n  TEST A (impostor with enrol s_u): {test_a_global*100:.2f}%")
    ratio_a = test_a_global / (ref_global + 1e-10)
    print(f"  Degradation vs correct: {ratio_a:.1f}x (1.0x = no routing, >>1x = partial routing, large = full collapse)")
    if test_a_global > 0.40:
        print("  → FULL COLLAPSE: impostors mapped to enrolment cluster (EER≈50%)")
    elif ratio_a > 3.0:
        print("  → PARTIAL ROUTING: s_u acts as routing label — both backbone AND β contribute")
    elif ratio_a > 1.5:
        print("  → MILD ROUTING: moderate s_u influence, backbone still dominant")
    else:
        print("  → NO ROUTING: FiLM uses backbone embedding without s_u-based routing")

    print(f"\n  TEST B (constant s_u): {test_b_global*100:.2f}%")
    if abs(test_b_global - ref_global) < 0.05:
        print("  → FiLM IGNORES s_u content: constant s_u gives same EER as correct s_u")
    elif test_b_global > 0.40:
        print("  → FiLM needs unique s_u per user: constant s_u destroys discrimination")
    else:
        print(f"  → Moderate sensitivity to s_u uniqueness")

    print(f"\n  TEST C (zero s_u): {test_c_global*100:.2f}%")
    if abs(test_c_global - raw_global) < 0.03:
        print("  → FiLM with zero s_u ≈ raw cosine: identity-like behavior")
    elif test_c_global > 0.40:
        print("  → Zero s_u destroys discrimination: FiLM needs non-trivial s_u")
    else:
        print(f"  → Partial degradation with zero s_u")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()

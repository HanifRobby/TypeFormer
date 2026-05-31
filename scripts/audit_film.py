"""
audit_film.py — Audit FiLM checkpoint for potential leaks and artifacts.

Checks:
  1. gamma/beta deviation from identity (is FiLM actually doing anything?)
  2. Per-user EER distribution (how many users have exactly 0% EER?)
  3. Score distribution: genuine vs impostor histograms
  4. Cross-user s_u: does applying user A's s_u to user B's embedding
     produce high similarity with user A's template? (tests label-collapse)
  5. Session-split test: s_u from sessions 0-4, enrolment from sessions 5-9,
     probes from sessions 10-14 (no temporal overlap between s_u and enrolment)

Usage:
    conda run -n Typeformer python scripts/audit_film.py
    conda run -n Typeformer python scripts/audit_film.py --exp fn_full_system
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.film_head import FiLMHead
from src.statistics.user_stats import compute_user_stats
from src.evaluation.metrics import compute_eer
from src.scoring.cosine_raw import RawCosineScorer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="f1_film_only")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-users", type=int, default=200,
                   help="How many test users to use for audit (default 200)")
    return p.parse_args()


def load_film(exp: str, seed: int, cfg_root: Path) -> FiLMHead:
    ckpt = cfg_root / "results" / exp / "checkpoints" / f"film_head_seed{seed}.pt"
    film = FiLMHead(stats_dim=20, embedding_dim=64, hidden_dim=64)
    film.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
    film.eval()
    return film


def compute_gamma_beta_stats(film: FiLMHead, s_u_batch: np.ndarray) -> dict:
    """Compute gamma/beta for a batch of s_u vectors."""
    with torch.no_grad():
        s_t = torch.from_numpy(s_u_batch).float()
        gb = film.fc2(torch.relu(film.fc1(s_t)))
        gamma = gb[:, :64].numpy()
        beta = gb[:, 64:].numpy()
    return {
        "gamma_mean": float(gamma.mean()),
        "gamma_std": float(gamma.std()),
        "gamma_per_user_deviation": float(np.abs(gamma - 1.0).mean()),
        "beta_mean": float(beta.mean()),
        "beta_std": float(beta.std()),
        "beta_per_user_norm": float(np.abs(beta).mean()),
        "gamma_user_to_user_var": float(gamma.var(axis=0).mean()),
        "beta_user_to_user_var": float(beta.var(axis=0).mean()),
    }


def apply_film_batch(film, embs, s_u):
    """Apply FiLM: embs (N, D), s_u (D_s,) → (N, D)"""
    with torch.no_grad():
        e_t = torch.from_numpy(embs).float()
        s_t = torch.from_numpy(s_u).float().unsqueeze(0).expand(len(embs), -1)
        return film(e_t, s_t).numpy()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"AUDIT: {args.exp}  seed={args.seed}  n_users={args.n_users}")
    print(f"{'='*60}\n")

    # Load data
    emb_path = ROOT / "results" / "embeddings_cache" / "test" / "test_embeddings.npy"
    sess_path = ROOT / "data" / "processed" / "test_sessions.npz"

    if not emb_path.exists() or not sess_path.exists():
        print("ERROR: test embeddings or sessions not found.")
        sys.exit(1)

    embeddings = np.load(str(emb_path))[:args.n_users]  # (N, 15, 64)
    sessions_data = np.load(str(sess_path))
    sessions = sessions_data["sessions"][:args.n_users]  # (N, 15, 50, 5)
    n_users, n_sess, L, C = sessions.shape
    D = embeddings.shape[-1]

    film = load_film(args.exp, args.seed, ROOT)

    # ----------------------------------------------------------------
    # CHECK 1: gamma/beta statistics
    # ----------------------------------------------------------------
    print("CHECK 1: gamma/beta deviation from identity")
    print("-" * 50)

    s_u_all = np.stack([
        compute_user_stats(sessions[u, :5]) for u in range(n_users)
    ])  # (N, 20)

    stats = compute_gamma_beta_stats(film, s_u_all)
    print(f"  gamma mean:                   {stats['gamma_mean']:.4f}  (expected: 1.0 at init)")
    print(f"  gamma std across all dims:    {stats['gamma_std']:.4f}")
    print(f"  ||gamma-1|| mean (per dim):   {stats['gamma_per_user_deviation']:.4f}  (<0.01 → near-identity)")
    print(f"  beta mean:                    {stats['beta_mean']:.4f}  (expected: 0.0 at init)")
    print(f"  ||beta|| mean (per dim):      {stats['beta_per_user_norm']:.4f}  (<0.01 → near-identity)")
    print(f"  gamma user-to-user variance:  {stats['gamma_user_to_user_var']:.6f}  (>0 → user-specific)")
    print(f"  beta user-to-user variance:   {stats['beta_user_to_user_var']:.6f}  (>0 → user-specific)")

    # ----------------------------------------------------------------
    # CHECK 2: per-user EER distribution
    # ----------------------------------------------------------------
    print("\nCHECK 2: per-user EER distribution from test evaluation")
    print("-" * 50)

    eer_path = ROOT / "results" / args.exp / "per_user_eers_E5.csv"
    if eer_path.exists():
        per_user_eers = np.loadtxt(str(eer_path), delimiter=",")[:args.n_users]
        n_zero = int((per_user_eers == 0.0).sum())
        n_nonzero = int((per_user_eers > 0.0).sum())
        print(f"  Users with EER=0.0%:   {n_zero}/{len(per_user_eers)} ({100*n_zero/len(per_user_eers):.1f}%)")
        print(f"  Users with EER>0%:     {n_nonzero}/{len(per_user_eers)}")
        print(f"  EER percentiles: p50={np.percentile(per_user_eers,50)*100:.3f}%  "
              f"p90={np.percentile(per_user_eers,90)*100:.3f}%  "
              f"p99={np.percentile(per_user_eers,99)*100:.3f}%")
        print(f"  Max per-user EER: {per_user_eers.max()*100:.2f}%")
    else:
        print("  per_user_eers_E5.csv not found — run 07b_evaluate_film.py first")

    # ----------------------------------------------------------------
    # CHECK 3: score distribution genuine vs impostor
    # ----------------------------------------------------------------
    print("\nCHECK 3: genuine vs impostor score distributions (E=5, n=100 users)")
    print("-" * 50)

    n_audit = min(100, n_users)
    scorer = RawCosineScorer()

    film_embs = np.zeros((n_audit, n_sess, D), dtype=np.float32)
    for u in range(n_audit):
        s_u = compute_user_stats(sessions[u, :5])
        film_embs[u] = apply_film_batch(film, embeddings[u], s_u)

    genuine_scores, impostor_scores = [], []
    for u in range(n_audit):
        e_u = film_embs[u, :5].mean(axis=0)
        for probe in film_embs[u, -5:]:
            genuine_scores.append(scorer.score(e_u, probe))
        imp_probe = film_embs[:, -1, :]  # last session of each user
        for j in range(n_audit):
            if j != u:
                impostor_scores.append(scorer.score(e_u, imp_probe[j]))

    gen = np.array(genuine_scores)
    imp = np.array(impostor_scores)
    eer_audit, _ = compute_eer(gen, imp)

    print(f"  Genuine scores:  mean={gen.mean():.4f}  std={gen.std():.4f}  "
          f"min={gen.min():.4f}  max={gen.max():.4f}")
    print(f"  Impostor scores: mean={imp.mean():.4f}  std={imp.std():.4f}  "
          f"min={imp.min():.4f}  max={imp.max():.4f}")
    print(f"  Score separation (gen_mean - imp_mean): {gen.mean()-imp.mean():.4f}")
    print(f"  EER from this sample: {eer_audit*100:.2f}%")

    # ----------------------------------------------------------------
    # CHECK 4: cross-user s_u test (label collapse detection)
    # ----------------------------------------------------------------
    print("\nCHECK 4: cross-user s_u test (does wrong s_u give low scores?)")
    print("-" * 50)

    n_cross = min(50, n_users)
    genuine_correct, genuine_wrong_su = [], []

    for u in range(n_cross):
        e_u_correct = film_embs[u, :5].mean(axis=0)  # modulated with own s_u

        # Probe with CORRECT s_u (sessions 10-14, modulated with own s_u)
        for probe in film_embs[u, -5:]:
            genuine_correct.append(scorer.score(e_u_correct, probe))

        # Probe with WRONG s_u: apply user (u+1)'s s_u to user u's embedding
        wrong_user = (u + 1) % n_cross
        s_u_wrong = compute_user_stats(sessions[wrong_user, :5])
        probes_wrong = apply_film_batch(film, embeddings[u, -5:], s_u_wrong)
        for probe in probes_wrong:
            genuine_wrong_su.append(scorer.score(e_u_correct, probe))

    gc = np.array(genuine_correct)
    gw = np.array(genuine_wrong_su)
    print(f"  Genuine (correct s_u):    mean={gc.mean():.4f}  std={gc.std():.4f}")
    print(f"  Genuine (wrong s_u):      mean={gw.mean():.4f}  std={gw.std():.4f}")
    print(f"  Score drop when s_u wrong: {gc.mean()-gw.mean():.4f}")
    if gc.mean() - gw.mean() > 0.1:
        print("  → Large drop: FiLM IS using s_u content for modulation")
    elif gc.mean() - gw.mean() > 0.01:
        print("  → Moderate drop: FiLM uses s_u partially")
    else:
        print("  → Tiny/no drop: FiLM may be ignoring s_u content (label collapse)")

    # ----------------------------------------------------------------
    # CHECK 5: session-split test (s_u sessions ≠ enrolment sessions)
    # ----------------------------------------------------------------
    print("\nCHECK 5: session-split test")
    print("  Protocol: s_u from sessions 0-4, enrolment from 5-9, probes from 10-14")
    print("  (Eliminates any shared sessions between s_u and enrolment templates)")
    print("-" * 50)

    n_split = min(100, n_users)
    per_user_eers_split = []

    # Apply FiLM with s_u from sessions 0-4 to ALL sessions
    film_embs_split = np.zeros((n_split, n_sess, D), dtype=np.float32)
    for u in range(n_split):
        s_u = compute_user_stats(sessions[u, :5])  # sessions 0-4 only for s_u
        film_embs_split[u] = apply_film_batch(film, embeddings[u], s_u)

    # Enrolment: sessions 5-9 (5 sessions, different from s_u sessions)
    # Probes: sessions 10-14
    gen_s, imp_s = [], []
    for u in range(n_split):
        e_u = film_embs_split[u, 5:10].mean(axis=0)   # sessions 5-9
        genuine = film_embs_split[u, 10:15]            # sessions 10-14
        impostors = film_embs_split[[j for j in range(n_split) if j != u], -1]

        gen_sc = [scorer.score(e_u, p) for p in genuine]
        imp_sc = [scorer.score(e_u, p) for p in impostors]
        eer, _ = compute_eer(np.array(gen_sc), np.array(imp_sc))
        per_user_eers_split.append(eer)
        gen_s.extend(gen_sc)
        imp_s.extend(imp_sc)

    pu_split = np.array(per_user_eers_split)
    global_eer_split, _ = compute_eer(np.array(gen_s), np.array(imp_s))
    print(f"  Mean per-subject EER: {pu_split.mean()*100:.2f}%  std={pu_split.std()*100:.2f}%")
    print(f"  Global EER:           {global_eer_split*100:.2f}%")
    print(f"  Users with EER=0%:    {(pu_split==0).sum()}/{n_split}")

    # Compare with standard protocol (s_u and enrolment both from sessions 0-4)
    print(f"\n  Standard protocol (s_u=sessions 0-4, enrolment=sessions 0-4):")
    per_user_std = []
    gen_std, imp_std = [], []
    for u in range(n_split):
        e_u = film_embs[u, :5].mean(axis=0)  # enrolment from 0-4 (already FiLM'd)
        genuine = film_embs[u, 10:15]
        impostors = film_embs[[j for j in range(n_split) if j != u], -1]
        gen_sc = [scorer.score(e_u, p) for p in genuine]
        imp_sc = [scorer.score(e_u, p) for p in impostors]
        eer, _ = compute_eer(np.array(gen_sc), np.array(imp_sc))
        per_user_std.append(eer)
        gen_std.extend(gen_sc)
        imp_std.extend(imp_sc)
    pu_std = np.array(per_user_std)
    global_std, _ = compute_eer(np.array(gen_std), np.array(imp_std))
    print(f"  Mean per-subject EER: {pu_std.mean()*100:.2f}%  std={pu_std.std()*100:.2f}%")
    print(f"  Global EER:           {global_std*100:.2f}%")

    print(f"\n  EER change (split vs standard):")
    print(f"  Per-subject: {pu_split.mean()*100:.2f}% vs {pu_std.mean()*100:.2f}%")
    print(f"  Global:      {global_eer_split*100:.2f}% vs {global_std*100:.2f}%")

    if abs(pu_split.mean() - pu_std.mean()) < 0.005:
        print("  → SIMILAR: s_u computation not biased by session overlap")
    else:
        print("  → DIFFERENT: session overlap between s_u and enrolment inflates results")

    # ----------------------------------------------------------------
    # CHECK 6: compare with RAW (non-FiLM) cosine at same protocol
    # ----------------------------------------------------------------
    print("\nCHECK 6: raw cosine (no FiLM) at session-split protocol for reference")
    print("-" * 50)
    per_user_raw = []
    gen_raw, imp_raw = [], []
    for u in range(n_split):
        e_u_raw = embeddings[u, 5:10].mean(axis=0)
        genuine_raw = embeddings[u, 10:15]
        imp_raw_u = embeddings[[j for j in range(n_split) if j != u], -1]
        gen_sc = [scorer.score(e_u_raw, p) for p in genuine_raw]
        imp_sc = [scorer.score(e_u_raw, p) for p in imp_raw_u]
        eer, _ = compute_eer(np.array(gen_sc), np.array(imp_sc))
        per_user_raw.append(eer)
        gen_raw.extend(gen_sc)
        imp_raw.extend(imp_sc)
    pu_raw = np.array(per_user_raw)
    global_raw, _ = compute_eer(np.array(gen_raw), np.array(imp_raw))
    print(f"  Raw cosine per-subject EER: {pu_raw.mean()*100:.2f}%  global: {global_raw*100:.2f}%")
    print(f"  FiLM split per-subject EER: {pu_split.mean()*100:.2f}%  global: {global_eer_split*100:.2f}%")

    improvement_global = (global_raw - global_eer_split) / global_raw * 100
    print(f"  FiLM improvement (global): {improvement_global:.1f}% relative reduction")
    if improvement_global > 10:
        print("  → FiLM shows genuine improvement even with separated s_u/enrolment sessions")
    else:
        print("  → FiLM improvement marginal in clean protocol — results may be inflated")

    print(f"\n{'='*60}")
    print("AUDIT COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

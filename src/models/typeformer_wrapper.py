"""
TypeFormerWrapper — frozen backbone for embedding extraction.

The TypeFormer model architecture is loaded from TypeFormer/model/Model.py
using the architecture parameters from TypeFormer/utils/config.yaml.
Only the model file is imported; the full TypeFormer config module is
NOT imported to avoid side-effects (directory creation, file-system writes).

CUDA requirement: HARTrans.__init__() hardcodes .to('cuda') for two Conv1d
layers stored in Python lists (not nn.ModuleList), so CUDA must be available
at construction time.  The wrapper always runs on CUDA.
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
_TYPEFORMER_DIR = _ROOT / "TypeFormer"


def _load_typeformer_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Import HARTrans, instantiate with config, load pretrained weights."""
    # Temporarily add TypeFormer/ to path for the import
    tf_str = str(_TYPEFORMER_DIR)
    if tf_str not in sys.path:
        sys.path.insert(0, tf_str)
        _added = True
    else:
        _added = False

    try:
        from model.Model import HARTrans  # noqa: PLC0415
    finally:
        if _added and tf_str in sys.path:
            sys.path.remove(tf_str)

    # Read only model architecture params from TypeFormer config YAML
    tf_config_path = _TYPEFORMER_DIR / "utils" / "config.yaml"
    with open(tf_config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    mc = raw["model"]
    args = SimpleNamespace(
        sequence_length=mc["sequence_length"],
        dimensionality=mc["dimensionality"],
        output_dim=mc["output_dim"],
        K=mc["K"],
        hlayers=mc["hlayers"],
        hlayers_rec=mc["hlayers_rec"],
        hlayers_pos=mc["hlayers_pos"],
        hheads=mc["hheads"],
        vlayers=mc["vlayers"],
        vheads=mc["vheads"],
    )

    # HARTrans.__init__ hard-codes .to('cuda') for self.encoders & self.encoder_v
    # so CUDA must be available before instantiation.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TypeFormerWrapper requires CUDA. "
            "HARTrans.__init__() contains hard-coded .to('cuda') calls."
        )

    model = HARTrans(args).float()
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    # If weights were saved as double, load then implicitly cast via .float() above
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    logger.info(
        "TypeFormer loaded: %s  device=%s  params=%d  (all frozen)",
        checkpoint_path.name, device,
        sum(p.numel() for p in model.parameters()),
    )
    return model


class TypeFormerWrapper:
    """Frozen TypeFormer backbone for keystroke embedding extraction.

    Args:
        checkpoint_path: Path to TypeFormer_pretrained.pt.
        device: Torch device string (must be 'cuda' or 'cuda:N').
    """

    EMBEDDING_DIM: int = 64

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        if checkpoint_path is None:
            checkpoint_path = _ROOT / "TypeFormer" / "pretrained" / "TypeFormer_pretrained.pt"
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model = _load_typeformer_model(self.checkpoint_path, device)

    @torch.no_grad()
    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        """Encode a batch of keystroke sequences.

        Args:
            sequences: (B, L, 5) float32 Tensor.

        Returns:
            (B, 64) float32 Tensor on CPU.
        """
        sequences = sequences.to(self.device, dtype=torch.float32)
        embeddings = self.model(sequences)
        return embeddings.cpu()

    @torch.no_grad()
    def encode_numpy(self, sequences: np.ndarray) -> np.ndarray:
        """Encode a numpy array of sequences.

        Args:
            sequences: (B, L, 5) float32 array.

        Returns:
            (B, 64) float32 numpy array.
        """
        t = torch.from_numpy(sequences).float()
        return self.encode(t).numpy()

    def encode_dataset(
        self,
        sessions: np.ndarray,
        batch_size: int = 64,
        desc: str = "Encoding",
    ) -> np.ndarray:
        """Encode all sessions in a dataset array.

        Args:
            sessions: (N_users, N_sessions, L, 5) or (N_total, L, 5) float32.
            batch_size: Number of sequences per GPU batch.
            desc: tqdm description string.

        Returns:
            Same leading dimensions, last dim replaced by 64.
            e.g. (N_users, N_sessions, 64) or (N_total, 64).
        """
        original_shape = sessions.shape
        flat = sessions.reshape(-1, original_shape[-2], original_shape[-1])  # (N, L, 5)
        n = len(flat)

        results = []
        for start in tqdm(range(0, n, batch_size), desc=desc, unit="batch"):
            batch = flat[start: start + batch_size]
            embs = self.encode_numpy(batch)
            results.append(embs)

        all_embs = np.concatenate(results, axis=0)  # (N, 64)
        out_shape = original_shape[:-2] + (self.EMBEDDING_DIM,)
        return all_embs.reshape(out_shape)

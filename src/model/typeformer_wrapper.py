import torch
import numpy as np
from pathlib import Path
from typing import List
import sys

# Ensure TypeFormer modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model.Model import HARTrans
from utils.train_config import configs

class TypeFormerWrapper:
    """Wrapper around pretrained TypeFormer for embedding extraction."""
    
    def __init__(self, weights_path: str, device: str = 'auto', batch_size: int = 64):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.batch_size = batch_size
        self.model = self._load_model(weights_path)
        self.model.eval()
        
    def _load_model(self, weights_path: str):
        # We need to set configs required by the model
        model_args = configs
        
        # Initialize model
        model = HARTrans(model_args).double()
        
        # Load pretrained weights
        checkpoint = torch.load(weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        return model.to(self.device)
        
    def extract_embeddings(self, sessions: List[np.ndarray]) -> np.ndarray:
        if len(sessions) == 0:
            return np.empty((0, 64))
            
        all_embeddings = []
        for start in range(0, len(sessions), self.batch_size):
            batch = sessions[start : start + self.batch_size]
            
            # The model expects reshape (N, dimension, sequence_length) internally,
            # but input to model should be shape (batch, seq_len, dim) = (N, 50, 5)
            # as it will do data.view(data.shape[0], self.dimension, -1)
            batch_tensor = torch.tensor(np.stack(batch, axis=0), dtype=torch.float64, device=self.device)
            
            with torch.no_grad():
                embeddings = self.model(batch_tensor)
                
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)
        
    def extract_single(self, session: np.ndarray) -> np.ndarray:
        return self.extract_embeddings([session])[0]

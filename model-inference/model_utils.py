"""
model_utils.py

Model architecture definitions for BiLSTM-CRF.
Extracted from train_bilstm_crf.py for standalone inference.
"""
import torch
import torch.nn as nn
import pandas as pd
from TorchCRF import CRF

class BiLSTMCRF(nn.Module):
    """BiLSTM-CRF model for surgical phase recognition."""
    def __init__(self, input_dim=512, hidden_dim=256, n_layers=2, n_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=0.2 if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, n_classes)
        self.crf = CRF(n_classes)
    
    def forward(self, x, mask=None):
        out, _ = self.lstm(x)
        emissions = self.fc(out)
        # TorchCRF expects (seq_len, batch, num_tags)
        emissions = emissions.transpose(0, 1)
        return emissions

def get_label_mapping(labels_csv):
    """Get label to index mapping from phases CSV."""
    df = pd.read_csv(labels_csv)
    unique_phases = sorted(df['phase'].unique())
    label_to_idx = {phase: idx for idx, phase in enumerate(unique_phases)}
    # Add background class if not present
    if "background" not in label_to_idx:
        label_to_idx["background"] = len(label_to_idx)
    return label_to_idx

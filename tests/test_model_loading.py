"""
Test Model Loading for Resona
"""

import sys
import os
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from resona.core.model import ResonaModel

def test_model_instantiation():
    print("Testing ResonaModel instantiation on CPU...")
    model = ResonaModel(device="cpu")
    assert isinstance(model, torch.nn.Module)
    assert len(model.vocab) > 50
    print("[PASS] ResonaModel loaded successfully.")

if __name__ == "__main__":
    test_model_instantiation()

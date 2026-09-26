"""
Test Voice Registry and Style Vectors
"""

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from resona.voices import VoiceManager

def test_registry():
    voices = VoiceManager.list_voices()
    assert len(voices) >= 20, f"Expected at least 20 voices, got {len(voices)}"
    
    for v in voices:
        assert v.grade == "A", f"Voice {v.id} is not Grade A"
        tensor = VoiceManager.load_voice_tensor(v.id)
        assert tensor.shape[-1] == 256, f"Invalid tensor shape for {v.id}: {tensor.shape}"

    print(f"[PASS] All {len(voices)} Grade A voices verified with valid style tensors.")

if __name__ == "__main__":
    test_registry()

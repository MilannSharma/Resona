"""
Resona Audio Utilities
Silence injection, natural breathing cadence, and broadcast-quality file output.
"""

import os
import re
from typing import List, Tuple
import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000

def create_silence(duration_sec: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generates zero-amplitude silence array."""
    num_samples = int(sample_rate * max(0.0, duration_sec))
    return np.zeros(num_samples, dtype=np.float32)


def split_into_clauses(text: str) -> List[Tuple[str, str]]:
    """
    Splits text into clause segments and identifies trailing punctuation
    (e.g., comma, period, question mark) for natural pause calculation.
    """
    if not text:
        return []

    # Match tokens ending with punctuation or end of string
    parts = re.split(r'([,;:.!?।]+)', text.strip())
    clauses = []
    
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        punct = ""
        if i + 1 < len(parts):
            punct = parts[i + 1].strip()
            i += 2
        else:
            i += 1
        
        if chunk or punct:
            clauses.append((chunk, punct))

    return clauses


def save_wav(output_path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Saves float32 numpy audio array to 24kHz WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # Clip to prevent digital distortion
    audio_clipped = np.clip(audio, -1.0, 1.0)
    sf.write(output_path, audio_clipped, sample_rate)

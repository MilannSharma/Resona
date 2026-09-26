"""
Resona Engine Configuration
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ResonaConfig:
    """Configuration for Resona-82M Neural Audio Engine."""
    sample_rate: int = 24000
    hidden_dim: int = 256
    style_dim: int = 256
    n_layer: int = 3
    n_token: int = 178
    max_dur: int = 50
    dropout: float = 0.2
    text_encoder_kernel_size: int = 5
    n_mels: int = 80
    default_speed: float = 0.95
    comma_pause_sec: float = 0.18
    sentence_pause_sec: float = 0.35
    paragraph_pause_sec: float = 0.50
    device: str = "cpu"

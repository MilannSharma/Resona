"""
Resona: Lightweight Offline Neural Text-to-Speech Engine
"""

__version__ = "1.0.0"

from .core.model import ResonaModel
from .core.pipeline import ResonaPipeline, SynthesisResult
from .core.config import ResonaConfig
from .voices import VoiceManager, VoiceInfo

__all__ = [
    "ResonaModel",
    "ResonaPipeline",
    "SynthesisResult",
    "ResonaConfig",
    "VoiceManager",
    "VoiceInfo",
]

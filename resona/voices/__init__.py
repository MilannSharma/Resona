"""
Resona Voice Manager & Registry Interface
Single source of truth for voice discovery, tensor loading, and style vector blending.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch

@dataclass
class VoiceInfo:
    id: str
    display_name: str
    gender: str
    grade: str
    status: str
    character: str
    accent: str
    primary_language: str
    supported_languages: List[str]
    f0_pitch_hz: float
    recommended_speed: float
    style_tensor_path: str
    sample_audio_path: str
    description: str


class VoiceManager:
    """Manages available Resona voices, metadata, and style vector loading."""

    _registry: Optional[Dict[str, Any]] = None
    _cached_tensors: Dict[str, torch.FloatTensor] = {}

    @classmethod
    def _ensure_registry(cls):
        if cls._registry is None:
            reg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "registry.json"))
            if not os.path.isfile(reg_path):
                raise FileNotFoundError(f"Resona voice registry not found at {reg_path}")
            with open(reg_path, "r", encoding="utf-8") as f:
                cls._registry = json.load(f)

    @classmethod
    def list_voices(cls, language: Optional[str] = None) -> List[VoiceInfo]:
        """Returns all registered voices, optionally filtered by supported language."""
        cls._ensure_registry()
        voices = []
        for v in cls._registry.get("voices", []):
            if language:
                lang_clean = language.lower().strip()
                if lang_clean not in [l.lower() for l in v.get("supported_languages", [])]:
                    continue
            voices.append(VoiceInfo(
                id=v["id"],
                display_name=v["display_name"],
                gender=v["gender"],
                grade=v["grade"],
                status=v["status"],
                character=v["character"],
                accent=v["accent"],
                primary_language=v["primary_language"],
                supported_languages=v["supported_languages"],
                f0_pitch_hz=v.get("f0_pitch_hz", 150.0),
                recommended_speed=v.get("recommended_speed", 0.95),
                style_tensor_path=v["style_tensor"],
                sample_audio_path=v["sample_audio"],
                description=v["description"],
            ))
        return voices

    @classmethod
    def get_voice(cls, voice_id: str) -> Optional[VoiceInfo]:
        """Retrieves metadata for a specific voice ID."""
        voices = cls.list_voices()
        for v in voices:
            if v.id.lower() == voice_id.lower():
                return v
        return None

    @classmethod
    def get_default_voice(cls, language: str) -> str:
        """Returns default voice ID for a given language."""
        cls._ensure_registry()
        defaults = cls._registry.get("default_voices", {})
        return defaults.get(language.lower(), "anjura")

    @classmethod
    def load_voice_tensor(cls, voice_id: str, device: str = "cpu") -> torch.FloatTensor:
        """Loads and caches the [510, 1, 256] style tensor for a voice."""
        if voice_id in cls._cached_tensors:
            return cls._cached_tensors[voice_id].to(device)

        voice = cls.get_voice(voice_id)
        if not voice:
            # Fallback to anjura
            voice = cls.get_voice("anjura")

        # Resolve asset path relative to package root
        asset_rel = voice.style_tensor_path
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        tensor_path = os.path.join(base_dir, asset_rel)

        if not os.path.isfile(tensor_path):
            # Direct search in assets/
            alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", f"{voice_id}.pt"))
            if os.path.isfile(alt_path):
                tensor_path = alt_path
            else:
                raise FileNotFoundError(f"Voice style tensor for '{voice_id}' not found at {tensor_path}")

        tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
        cls._cached_tensors[voice_id] = tensor
        return tensor.to(device)

    @classmethod
    def blend_voices(
        cls,
        voice_id_a: str,
        voice_id_b: str,
        weight_a: float = 0.5,
        device: str = "cpu"
    ) -> torch.FloatTensor:
        """Interpolates between two voice style vectors."""
        t_a = cls.load_voice_tensor(voice_id_a, device=device)
        t_b = cls.load_voice_tensor(voice_id_b, device=device)
        weight_b = 1.0 - weight_a
        return (weight_a * t_a + weight_b * t_b)

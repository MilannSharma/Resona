"""
Resona Speech Pipeline
Orchestrates text normalization, bilingual code-switching, phonemization,
neural vocoding, and natural breathing cadence.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
import torch

from .model import ResonaModel
from .config import ResonaConfig
from ..preprocessing.espeak_finder import configure_espeak
from ..preprocessing.normalizer import normalize_text
from ..preprocessing.hinglish import ResonaHinglishEngine
from ..preprocessing.phonemizer import ResonaPhonemizer
from ..voices import VoiceManager
from ..utils.audio import create_silence, save_wav, SAMPLE_RATE


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    output_path: Optional[str] = None


class ResonaPipeline:
    """High-level synthesis pipeline for Resona Audio Engine."""

    def __init__(
        self,
        voice: str = "anjura",
        language: str = "hinglish",
        model: Optional[Union[str, ResonaModel]] = None,
        device: str = "cpu",
        config: Optional[ResonaConfig] = None,
    ):
        configure_espeak()

        self.voice = voice
        self.language = language.lower().strip()
        self.device = device
        self.config = config or ResonaConfig(device=device)

        # Preprocessing engines
        self.hinglish_engine = ResonaHinglishEngine()

        # Phonemizer mapping
        lang_map = {
            "h": "h", "hi": "h", "hindi": "h", "hinglish": "h", "mr": "mr",
            "a": "a", "en": "a", "en-us": "a", "english": "a",
            "b": "b", "en-gb": "b", "british": "b",
            "e": "e", "es": "e", "spanish": "e",
            "f": "f", "fr": "f", "french": "f",
            "i": "i", "it": "i", "italian": "i",
            "p": "p", "pt": "p", "portuguese": "p",
            "j": "j", "ja": "j", "japanese": "j",
            "z": "z", "zh": "z", "cmn": "z", "chinese": "z", "mandarin": "z",
        }
        lang_code = lang_map.get(self.language, "a")
        self.phonemizer = ResonaPhonemizer(lang_code=lang_code)

        # Model resolution
        if isinstance(model, ResonaModel):
            self.model = model
        else:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
            cand_indic = os.path.join(models_dir, "resona-indic-v1.pth")
            cand_base = os.path.join(models_dir, "resona-v1.pth")

            if model == "base":
                m_path = cand_base if os.path.isfile(cand_base) else cand_indic
            elif model == "indic" or self.language in ("hinglish", "hi", "hindi", "mr"):
                m_path = cand_indic if os.path.isfile(cand_indic) else cand_base
            else:
                m_path = cand_base if os.path.isfile(cand_base) else cand_indic

            c_path = os.path.join(models_dir, "config.json")
            self.model = ResonaModel(model_path=m_path, config_path=c_path, device=device)

    def _split_sentences(self, text: str) -> List[str]:
        """Splits paragraph into sentences on period, exclamation, question mark, or danda."""
        splits = re.split(r'(?<=[.!?।])\s+', text.strip())
        return [s.strip() for s in splits if s.strip()]

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        output_path: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Synthesizes text into high-fidelity speech with natural cadence.
        """
        if not text or not text.strip():
            empty_arr = np.zeros(0, dtype=np.float32)
            return SynthesisResult(audio=empty_arr, sample_rate=SAMPLE_RATE, duration_seconds=0.0)

        target_voice = voice or self.voice
        target_speed = speed or self.config.default_speed

        # Load voice style vector
        ref_s = VoiceManager.load_voice_tensor(target_voice, device=self.device)

        # Text normalization & Hinglish code-switching
        cleaned_text = normalize_text(text)
        if self.language == "hinglish":
            processed_text = self.hinglish_engine.normalize(cleaned_text)
        else:
            processed_text = cleaned_text

        # Sentence-level processing with breathing cadence
        sentences = self._split_sentences(processed_text)
        all_chunks = []

        for i, s in enumerate(sentences):
            # Clause-level processing for comma pauses
            clauses = re.split(r'([,;])', s)
            for c_idx in range(0, len(clauses), 2):
                clause_text = clauses[c_idx].strip()
                if not clause_text:
                    continue

                # Generate phonemes
                phonemes = self.phonemizer.phonemize(clause_text)
                if not phonemes:
                    continue

                # Forward through neural vocoder
                out = self.model.forward(phonemes, ref_s, speed=target_speed)
                audio_t = out.audio.cpu().numpy()
                all_chunks.append(audio_t)

                # Add clause pause (comma / semicolon)
                if c_idx + 1 < len(clauses):
                    all_chunks.append(create_silence(self.config.comma_pause_sec, SAMPLE_RATE))

            # Add sentence breathing pause
            if i < len(sentences) - 1:
                all_chunks.append(create_silence(self.config.sentence_pause_sec, SAMPLE_RATE))

        if not all_chunks:
            full_audio = np.zeros(0, dtype=np.float32)
        else:
            full_audio = np.concatenate(all_chunks)

        dur_sec = len(full_audio) / SAMPLE_RATE

        if output_path:
            save_wav(output_path, full_audio, SAMPLE_RATE)

        return SynthesisResult(
            audio=full_audio,
            sample_rate=SAMPLE_RATE,
            duration_seconds=dur_sec,
            output_path=output_path,
        )

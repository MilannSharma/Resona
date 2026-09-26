"""
Resona Phonemizer
G2P Routing and Phoneme Sanitization for Resona Neural Vocoder.
"""

import os
import re
import shutil
import subprocess
from typing import Optional

def sanitize_phonemes(phonemes: str) -> str:
    """Removes invalid punctuation and metadata tokens from phoneme stream."""
    if not isinstance(phonemes, str):
        return ""
    clean = re.sub(r'\([a-zA-Z\-]+\)', '', phonemes)
    clean = re.sub(r'[\(\)\[\]]', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


class ResonaPhonemizer:
    """Wrapper around eSpeak-NG phonemization with language-specific routing."""

    def __init__(self, lang_code: str = "h"):
        self.lang_code = lang_code
        self._espeak_lang = self._resolve_espeak_lang(lang_code)

    def _resolve_espeak_lang(self, code: str) -> str:
        code_clean = code.lower().strip()
        mapping = {
            "h": "hi",
            "hi": "hi",
            "hindi": "hi",
            "hinglish": "hi",
            "mr": "mr",
            "a": "en-us",
            "en": "en-us",
            "en-us": "en-us",
            "english": "en-us",
            "b": "en-gb",
            "en-gb": "en-gb",
            "british": "en-gb",
            "e": "es",
            "es": "es",
            "spanish": "es",
            "f": "fr-fr",
            "fr": "fr-fr",
            "french": "fr-fr",
            "i": "it",
            "it": "it",
            "italian": "it",
            "p": "pt-br",
            "pt": "pt-br",
            "portuguese": "pt-br",
            "j": "ja",
            "ja": "ja",
            "japanese": "ja",
            "z": "cmn",
            "zh": "cmn",
            "cmn": "cmn",
            "chinese": "cmn",
            "mandarin": "cmn",
        }
        return mapping.get(code_clean, "en-us")

    def phonemize(self, text: str) -> str:
        """Converts text into IPA phonemes via eSpeak-NG C-lib or CLI fallback."""
        if not text or not text.strip():
            return ""

        # For Hindi on Windows, use CLI directly to prevent mbrola.dll console warning
        if self._espeak_lang in ("hi", "h", "hindi") and os.name == "nt":
            try:
                espeak_bin = shutil.which("espeak-ng") or r"C:\Program Files\eSpeak NG\espeak-ng.exe"
                if os.path.isfile(espeak_bin):
                    r = subprocess.run(
                        [espeak_bin, "-v", "inc/hi", "-q", "--ipa", text],
                        capture_output=True,
                        text=True,
                        encoding="utf-8"
                    )
                    if r.returncode == 0 and r.stdout.strip():
                        return sanitize_phonemes(r.stdout.strip())
            except Exception:
                pass

        # Attempt 1: Fast in-process EspeakBackend
        try:
            from phonemizer.backend import EspeakBackend
            backend = EspeakBackend(
                language=self._espeak_lang,
                preserve_punctuation=True,
                with_stress=True
            )
            raw_ps = backend.phonemize([text], strip=True)
            if raw_ps:
                return sanitize_phonemes(raw_ps[0])
        except Exception:
            pass

        # Attempt 2: Direct espeak-ng CLI execution (bypasses Windows mbrola issue)
        try:
            espeak_bin = shutil.which("espeak-ng")
            if not espeak_bin:
                for cand in [r"C:\Program Files\eSpeak NG\espeak-ng.exe", r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe"]:
                    if os.path.isfile(cand):
                        espeak_bin = cand
                        break

            if espeak_bin:
                voice = "inc/hi" if self._espeak_lang in ("hi", "h", "hindi") else self._espeak_lang
                r = subprocess.run(
                    [espeak_bin, "-v", voice, "-q", "--ipa", text],
                    capture_output=True,
                    text=True,
                    encoding="utf-8"
                )
                if r.returncode == 0 and r.stdout.strip():
                    return sanitize_phonemes(r.stdout.strip())
        except Exception:
            pass

        return ""

"""
Resona Text Normalizer
Normalizes numbers, percentages, abbreviations, and punctuation for speech synthesis.
"""

import re

def normalize_text(text: str) -> str:
    """Basic text cleanup and symbol expansion."""
    if not text:
        return ""

    t = text

    # Currency
    t = re.sub(r'₹\s*(\d+)', r'\1 rupees', t)
    t = re.sub(r'\$\s*(\d+)', r'\1 dollars', t)

    # Percentages
    t = re.sub(r'(\d+)\s*%', r'\1 percent', t)

    # Dashes and bullet points
    t = re.sub(r'—|–', ' - ', t)
    t = re.sub(r'^\s*[\*\-•]\s+', '', t, flags=re.MULTILINE)

    # Clean multiple spaces
    t = re.sub(r'\s+', ' ', t).strip()

    return t

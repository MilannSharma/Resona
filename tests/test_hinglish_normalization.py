"""
Test Hinglish Normalizer and Lexicon Decoupling
"""

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from resona.preprocessing.hinglish import ResonaHinglishEngine

def test_normalization():
    engine = ResonaHinglishEngine()

    assert len(engine.compound_phrases) >= 100, "Phrases not loaded"
    assert len(engine.words_map) >= 800, "Lexicon words not loaded"
    assert len(engine.whitelist) >= 700, "Whitelist not loaded"

    # Test sentence
    inp = "Top bar mein File menu par click kijiye, aur Export button dabaiye."
    out = engine.normalize(inp)

    assert "में" in out, "Failed transliteration for 'mein'"
    assert "File" in out, "Protected English term 'File' corrupted"
    assert "Export" in out, "Protected English term 'Export' corrupted"
    assert "कीजिए" in out, "Failed transliteration for 'kijiye'"

    print("[PASS] Hinglish normalization & vocabulary decoupling verified:", flush=True)
    print(f"  Input : {inp}", flush=True)
    print(f"  Output: {out}", flush=True)

if __name__ == "__main__":
    test_normalization()

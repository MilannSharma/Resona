"""
Resona Latency and Real-Time Factor (RTF) Benchmark
"""

import sys
import os
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from resona import ResonaPipeline

def run_benchmark():
    print("=" * 60)
    print("RESONA: INFERENCE SPEED & REAL-TIME FACTOR BENCHMARK")
    print("=" * 60)

    pipeline = ResonaPipeline(voice="anjura", language="hinglish", device="cpu")

    test_sentences = [
        "Resona ke iss tutorial mein aapka swagat hai.",
        "File menu par click kijiye aur Export option choose kijiye.",
        "Beginners ke liye Excel me Monthly Budget Banana step by step seekhenge.",
        "Apne expenses aur income ko organize karke financial freedom achieve kijiye.",
    ]

    total_words = 0
    total_audio_sec = 0.0
    total_synth_sec = 0.0

    # Warmup
    pipeline.synthesize("Warmup sentence.")

    for i, s in enumerate(test_sentences, 1):
        words = len(s.split())
        t0 = time.time()
        res = pipeline.synthesize(s)
        elapsed = time.time() - t0

        rtf = elapsed / max(0.01, res.duration_seconds)
        total_words += words
        total_audio_sec += res.duration_seconds
        total_synth_sec += elapsed

        print(f"Sentence {i}: {words:2d} words -> {res.duration_seconds:4.1f}s audio in {elapsed:4.2f}s (RTF: {rtf:4.2f}x)")

    avg_rtf = total_synth_sec / max(0.01, total_audio_sec)
    wps = total_words / max(0.01, total_synth_sec)

    print("\n" + "-" * 60)
    print(f"Average Real-Time Factor (RTF): {avg_rtf:.2f}x (Lower is faster)")
    print(f"Words Per Second (WPS)        : {wps:.1f} words/sec")
    print(f"Speech Rate                   : {total_words / (total_audio_sec / 60):.1f} WPM")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_benchmark()

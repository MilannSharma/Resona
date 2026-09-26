"""
Resona Quickstart Example
5-line synthesis script using the Resona Audio Engine.
"""

import sys
import os

# Add repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from resona import ResonaPipeline

def main():
    print("Initializing Resona Pipeline...")
    pipeline = ResonaPipeline(voice="anjura", language="hinglish")

    text = "Resona audio engine se speech generate karna super fast aur natural hai."
    output_wav = os.path.join(os.path.dirname(__file__), "quickstart_output.wav")

    print(f"Synthesizing: '{text}'")
    result = pipeline.synthesize(text, output_path=output_wav, speed=0.95)

    print(f"[SUCCESS] Audio generated in {result.duration_seconds:.2f}s!")
    print(f"Saved to: {output_wav}")

if __name__ == "__main__":
    main()

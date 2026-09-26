"""
Resona Hinglish Code-Switching Tutorial
Demonstrates technical term protection alongside native Hindi transliteration.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from resona import ResonaPipeline

def main():
    pipeline = ResonaPipeline(voice="arjun", language="hinglish")

    tutorial_script = (
        "Namaskar doston! Aaj ke tutorial mein hum dekhenge ki File menu par jaakar "
        "Project Settings kaise open karte hain. Sabse pehle resolution 1080p select kijiye, "
        "aur Export button click kar lijiye."
    )

    output_wav = os.path.join(os.path.dirname(__file__), "hinglish_tutorial_output.wav")
    print("Synthesizing Hinglish technical walkthrough...")
    result = pipeline.synthesize(tutorial_script, output_path=output_wav, speed=0.95)

    print(f"[SUCCESS] Tutorial audio generated! Duration: {result.duration_seconds:.2f}s")

if __name__ == "__main__":
    main()

"""
Resona Command-Line Interface (resona-tts)
"""

import argparse
import sys
import json
import os

from .core.pipeline import ResonaPipeline
from .voices import VoiceManager

def main():
    parser = argparse.ArgumentParser(description="Resona: Offline Neural Text-to-Speech Engine")
    parser.add_argument("--text", type=str, help="Text to speak")
    parser.add_argument("--voice", type=str, default="anjura", help="Voice ID (e.g., anjura, arjun, divya)")
    parser.add_argument("--lang", type=str, default="hinglish", help="Language code (hinglish, hi, en)")
    parser.add_argument("--speed", type=float, default=0.95, help="Speech speed multiplier (0.5 to 2.0)")
    parser.add_argument("--output", type=str, help="Output WAV file path")
    parser.add_argument("--list-voices", action="store_true", help="List all available voices and exit")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or 'cuda')")

    args = parser.parse_args()

    if args.list_voices:
        voices = VoiceManager.list_voices()
        print(f"\nResona Registered Voices ({len(voices)} available):\n" + "=" * 55)
        for v in voices:
            print(f"  • {v.display_name:<10} [{v.gender:<6}] Grade {v.grade} | {v.accent:<25} (ID: {v.id})")
        print("=" * 55 + "\n")
        sys.exit(0)

    if not args.text:
        print("[ERROR] Please provide --text to synthesize or use --list-voices.", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        print("[ERROR] Please provide an --output file path.", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing Resona Engine [Voice: {args.voice}, Language: {args.lang}]...")
    pipeline = ResonaPipeline(
        voice=args.voice,
        language=args.lang,
        device=args.device,
    )

    print(f"Synthesizing: \"{args.text[:60]}...\"")
    res = pipeline.synthesize(
        text=args.text,
        voice=args.voice,
        speed=args.speed,
        output_path=args.output,
    )

    print(f"\n[SUCCESS] Audio saved to: {os.path.abspath(args.output)}")
    print(f"Duration: {res.duration_seconds:.2f}s | Sample Rate: {res.sample_rate}Hz\n")

if __name__ == "__main__":
    main()

"""
Resona Model Packaging & Checksum Verifier
Generates SHA-256 checksums for model weights and voice assets to ensure release integrity.
"""

import os
import hashlib
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def calculate_sha256(file_path: str) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    print("=" * 60)
    print("RESONA: GENERATING RELEASE CHECKSUMS (SHA-256)")
    print("=" * 60)

    checksums = {}

    # 1. Models
    models_dir = os.path.join(REPO_ROOT, "models")
    if os.path.isdir(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith((".pth", ".json")):
                p = os.path.join(models_dir, f)
                h = calculate_sha256(p)
                checksums[f"models/{f}"] = h
                print(f"  [MODEL] models/{f:<25} -> {h}")

    # 2. Voice Assets
    voices_dir = os.path.join(REPO_ROOT, "resona", "voices", "assets")
    if os.path.isdir(voices_dir):
        for f in sorted(os.listdir(voices_dir)):
            if f.endswith(".pt"):
                p = os.path.join(voices_dir, f)
                h = calculate_sha256(p)
                checksums[f"resona/voices/assets/{f}"] = h
                print(f"  [VOICE] assets/{f:<25} -> {h}")

    out_file = os.path.join(REPO_ROOT, "checksums.sha256")
    with open(out_file, "w", encoding="utf-8") as f:
        for k, v in checksums.items():
            f.write(f"{v}  {k}\n")

    print("\n" + "=" * 60)
    print(f"Checksums saved to {out_file}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

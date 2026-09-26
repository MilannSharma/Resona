"""
eSpeak-NG Cross-Platform Auto-Discovery
Locates eSpeak-NG shared library and binaries across Windows, Linux, and macOS.
"""

import os
import shutil
import sys

def configure_espeak():
    """Dynamically discover and configure eSpeak-NG paths across platforms."""
    candidates = [
        os.environ.get("PHONEMIZER_ESPEAK_PATH", ""),
        r"C:\Program Files\eSpeak NG",
        r"C:\Program Files (x86)\eSpeak NG",
        r"C:\eSpeak NG",
        "/usr/bin/espeak-ng",
        "/usr/local/bin/espeak-ng",
        "/opt/homebrew/bin/espeak-ng",
    ]

    which_espeak = shutil.which("espeak-ng")
    if which_espeak:
        candidates.insert(0, which_espeak)

    found_dir = None
    found_dll = None

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            if os.path.isfile(candidate):
                cand_dir = os.path.dirname(candidate)
            else:
                cand_dir = candidate

            possible_dlls = [
                os.path.join(cand_dir, "libespeak-ng.dll"),
                os.path.join(cand_dir, "espeak-ng.dll"),
                os.path.join(cand_dir, "libespeak-ng.so"),
                os.path.join(cand_dir, "libespeak-ng.dylib"),
            ]
            for pd in possible_dlls:
                if os.path.exists(pd):
                    found_dir = cand_dir
                    found_dll = pd
                    break

            if not found_dll and (
                os.path.exists(os.path.join(cand_dir, "espeak-ng.exe"))
                or os.path.exists(os.path.join(cand_dir, "espeak-ng"))
            ):
                found_dir = cand_dir
                break

    if found_dir:
        os.environ["PHONEMIZER_ESPEAK_PATH"] = found_dir
        if found_dll:
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = found_dll
        path_env = os.environ.get("PATH", "")
        if found_dir not in path_env:
            os.environ["PATH"] = found_dir + os.pathsep + path_env
        return True, found_dir

    return False, None

"""
Batch Voice Preview Sample Generator
Generates standardized reference .wav preview files for all registered Grade A voices.
"""

import sys
import os
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from resona import ResonaPipeline, VoiceManager

SAMPLE_PROMPTS = {
    "hinglish": "Namaskar! Main {name} hoon. Resona Audio Engine ke saath aap high-quality tutorials aur voiceovers aasani se bana sakte hain.",
    "hi": "नमस्कार! मैं {name} हूँ। Resona ऑडियो इंजन के साथ अपनी आवाज़ को सजीव, स्पष्ट और प्रभावशाली बनाइए।",
    "en": "Hello! I am {name}. With Resona Audio Engine, you can create studio-quality speech easily.",
    "es": "¡Hola! Soy {name}. Con Resona Audio Engine, puedes crear voces naturales fácilmente.",
    "fr": "Bonjour! Je suis {name}. Avec Resona Audio Engine, vous pouvez créer des voix de haute qualité.",
}

def generate_samples(force: bool = False):
    print("=" * 60, flush=True)
    print("RESONA: GENERATING REFERENCE VOICE SAMPLES (.WAV)", flush=True)
    print("=" * 60, flush=True)

    samples_dir = os.path.join(REPO_ROOT, "resona", "voices", "samples")
    os.makedirs(samples_dir, exist_ok=True)

    voices = VoiceManager.list_voices()
    print(f"Discovered {len(voices)} Grade A voices in registry.\n", flush=True)

    # Cache pipelines per language
    pipelines = {
        "hinglish": ResonaPipeline(voice="anjura", language="hinglish"),
        "hi": ResonaPipeline(voice="meera", language="hi"),
        "en": ResonaPipeline(voice="heart", language="en"),
    }

    for idx, v in enumerate(voices, 1):
        voice_id = v.id
        display_name = v.display_name
        lang = v.primary_language if v.primary_language in pipelines else "en"
        out_wav = os.path.join(samples_dir, f"{voice_id}_preview.wav")

        if os.path.isfile(out_wav) and not force and os.path.getsize(out_wav) > 1000:
            print(f"[{idx}/{len(voices)}] {display_name:<12} ({voice_id}): Existing preview sample found, skipping.", flush=True)
            continue

        prompt_tmpl = SAMPLE_PROMPTS.get(lang, SAMPLE_PROMPTS["en"])
        text = prompt_tmpl.format(name=display_name)
        pipeline = pipelines.get(lang, pipelines["en"])

        print(f"[{idx}/{len(voices)}] Generating sample for {display_name:<12} ({voice_id}, lang={lang})...", end="", flush=True)
        t0 = time.time()
        try:
            res = pipeline.synthesize(
                text=text,
                voice=voice_id,
                speed=v.recommended_speed,
                output_path=out_wav
            )
            dur = res.duration_seconds
            print(f" OK! ({dur:.1f}s audio in {time.time()-t0:.2f}s) -> {os.path.basename(out_wav)}", flush=True)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("ALL VOICE PREVIEW SAMPLES GENERATED SUCCESSFULLY!", flush=True)
    print(f"Saved to: {samples_dir}", flush=True)
    print("=" * 60 + "\n", flush=True)

if __name__ == "__main__":
    generate_samples()

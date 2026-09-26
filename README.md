# Resona: Lightweight Offline Neural Text-to-Speech Engine
> **Fast, Expressive & 100% Offline Speech Synthesis with Authentic Hinglish Code-Switching.**

**Resona** is an ultra-lightweight (82M parameter) neural text-to-speech engine optimized for sub-second edge synthesis, conversational warmth, and natural bilingual Indian English / Hinglish tutorial narration.

---

## 🌟 Key Features

- **100% Offline & Private:** Zero internet connection required. All weights, style vectors, and dictionaries run locally on CPU or GPU.
- **Authentic Hinglish & Indic Code-Switching:** Seamlessly transitions between English UI/technical terminology and native Devanagari Hindi diction without unnatural accent distortion.
- **Natural Breathing Cadence:** Automatically injects subtle human pauses across clauses (180ms) and sentence boundaries (350ms).
- **21 Studio Grade A Personas:** Curated studio-quality personas covering corporate explainers, warm native storytellers, and deep baritone narrators.
- **Sub-Second Latency:** Generates speech in real-time or faster on standard commodity CPUs.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/MilannSharma/Resona.git
cd Resona
pip install -e .
```

*Note: Requires `espeak-ng` installed on your operating system.*

### 2. Python API

```python
from resona import ResonaPipeline

# Initialize pipeline with our flagship Hinglish narrator (Anjura)
pipeline = ResonaPipeline(voice="anjura", language="hinglish")

# Synthesize speech
result = pipeline.synthesize(
    text="Resona ke iss quick tutorial mein aapka swagat hai. File menu par click karke export kijiye.",
    output_path="tutorial.wav",
    speed=0.95
)

print(f"Generated {result.duration_seconds:.2f}s audio at {result.sample_rate}Hz!")
```

### 3. Command-Line Interface (`resona-tts`)

```bash
# List all 21 registered Grade A voices
resona-tts --list-voices

# Synthesize from terminal
resona-tts --text "Welcome to Resona Audio Engine." --voice arjun --output welcome.wav

# Synthesize Hinglish tutorial with custom speed
resona-tts --text "Settings menu par click kijiye aur resolution 1080p set kijiye." \
           --voice anjura \
           --lang hinglish \
           --speed 0.95 \
           --output settings.wav
```

---

## 🎙️ Voice Roster (21 Studio Grade A Voices)

All 21 voices include reference `.wav` preview samples in [`resona/voices/samples/`](resona/voices/samples/):

### 1. Hinglish & Indic Flagship Voices
| Voice ID | Name | Gender | Accent / Persona | Pitch | Rec. Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `anjura` | **Anjura** | Female | Studio Flagship Explainer (Grade A) | 184 Hz | 0.95x |
| `divya` | **Divya** | Female | Deep Corporate Explainer | 184.6 Hz | 0.95x |
| `meera` | **Meera** | Female | Warm Conversational Hindi / Hinglish | 210 Hz | 1.00x |
| `priya` | **Priya** | Female | Dynamic Modern Tech Educator | 195 Hz | 1.00x |
| `arjun` | **Arjun** | Male | Authoritative Enterprise Baritone Narrator | 115 Hz | 0.95x |
| `kabir` | **Kabir** | Male | Executive Conversational Narrator | 128 Hz | 0.95x |
| `aman` | **Aman** | Male | Agile Tech Guide (Noise-Floor Stabilized) | 132 Hz | 1.00x |
| `atul` | **Atul** | Male | Classic Indic Corporate Narrator | 122 Hz | 0.98x |

### 2. Indic English & Native Hindi Voices
| Voice ID | Name | Gender | Accent / Persona | Pitch | Rec. Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ananya` | **Ananya** | Female | Friendly Academic / Educator | 190 Hz | 1.00x |
| `nisha` | **Nisha** | Female | Calm, Gentle Presenter | 182 Hz | 0.98x |
| `tara` | **Tara** | Female | Bright & Youthful Explainer | 205 Hz | 1.00x |
| `shivani` | **Shivani** | Female | Expressive Native Hindi | 215 Hz | 0.98x |
| `dev` | **Dev** | Male | Professional News & Tech Anchor | 120 Hz | 1.00x |
| `sameer` | **Sameer** | Male | Friendly Walkthrough Mentor | 125 Hz | 0.98x |
| `ravi` | **Ravi** | Male | Deep Resonance Native Hindi Narrator | 112 Hz | 0.95x |
| `soham` | **Soham** | Male | Classic Soothing Storyteller | 110 Hz | 0.92x |

### 3. Global & International Voices
| Voice ID | Name | Gender | Accent / Persona | Pitch | Rec. Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `heart` | **Heart** | Female | Benchmark Global Narrator (Flagship) | 190 Hz | 1.00x |
| `bella` | **Bella** | Female | Warm American English Storyteller | 198 Hz | 1.00x |
| `sarah` | **Sarah** | Female | Crisp, Articulate Presenter | 188 Hz | 1.00x |
| `adam` | **Adam** | Male | Confident Commercial Voiceover | 118 Hz | 1.00x |
| `michael` | **Michael** | Male | Natural Conversational Baritone | 114 Hz | 1.00x |

---

## 🌐 Supported Languages & G2P Routing

Resona supports 10 global and regional languages out of the box:
* **Hinglish** (`hinglish`): Intelligent code-switching + vocabulary protection
* **Hindi** (`hi` / `hindi`): Native Devanagari G2P
* **English (US)** (`en` / `en-us`): Standard American English
* **English (UK)** (`en-gb`): British English
* **Spanish** (`es`), **French** (`fr`), **Italian** (`it`), **Portuguese** (`pt`)
* **Japanese** (`ja`), **Mandarin Chinese** (`zh` / `cmn`)

---

## 📁 Repository Structure

```text
resona/
├── resona/                     # 100% Self-Contained Python package
│   ├── core/                   # Neural vocoder, ResonaModel, ResonaPipeline, CustomSTFT
│   ├── preprocessing/          # Normalizer, ResonaHinglishEngine, G2P phonemizer
│   ├── voices/                 # VoiceManager, registry.json, 21 style tensors & samples
│   ├── vocab/                  # Bundled lexicons, compound phrases & whitelist
│   └── cli.py                  # resona-tts command line tool
├── models/                     # Checkpoints (resona-indic-v1.pth, resona-v1.pth)
├── vocab/                      # Root vocab dictionaries
├── examples/                   # Working python code examples
├── tests/                      # Automated test suite
└── pyproject.toml              # Build & packaging specifications
```

---

## 📄 License & Attribution

Resona is licensed under the [Apache License 2.0](LICENSE). 
Upstream architectural foundations and legal notices are detailed in [NOTICE](NOTICE).

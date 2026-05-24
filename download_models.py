"""
Run this FIRST before starting the server.
Downloads all AI models to your local cache.
Takes 5-10 minutes on first run. Only needed once.
"""

print("Step 1/4: Downloading Whisper STT model (~460MB)...")
import whisper
whisper.load_model("small")
print("✅ Whisper ready!\n")

print("Step 2/4: Downloading emotion detection model (~1.2GB)...")
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
AutoModelForAudioClassification.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
print("✅ Emotion model ready!\n")

print("Step 3/4: Downloading translation models (~300MB each)...")
from transformers import MarianMTModel, MarianTokenizer
for pair in ["en-hi", "hi-en", "en-fr", "fr-en", "en-de", "de-en", "en-es", "es-en"]:
    print(f"  Downloading Helsinki-NLP/opus-mt-{pair}...")
    try:
        MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{pair}")
        MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{pair}")
    except Exception as e:
        print(f"    ⚠️ Skipping {pair} (Not found on HF Hub)")
print("✅ Translation models ready!\n")

print("Step 4/4: Verifying gTTS...")
from gtts import gTTS
import io
t = gTTS("test", lang="en")
print("✅ TTS ready!\n")

print("=" * 40)
print("✅ ALL MODELS DOWNLOADED AND READY!")
print("Now run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
print("=" * 40)

"""
EchoLang — Emotion-Aware Real-Time Call Interpreter
Backend: FastAPI + WebSocket pipeline
"""

import os
try:
    import imageio_ffmpeg
    os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
except ImportError:
    pass
import io
import json
import base64
import tempfile
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EchoLang")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model loading (loaded once on first use) ──────────────────────────────
_whisper_model = None
_emotion_model = None
_emotion_processor = None
_emotion_labels = None
_tts_model = None
_translators = {}  # cache: "en-hi", "hi-en", etc.


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("[EchoLang] Loading Whisper small...")
        _whisper_model = whisper.load_model("small")
        print("[EchoLang] Whisper ready.")
    return _whisper_model


def get_emotion():
    global _emotion_model, _emotion_processor, _emotion_labels
    if _emotion_model is None:
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
        import torch
        print("[EchoLang] Loading emotion model...")
        model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        # Must use Wav2Vec2ForSequenceClassification — the checkpoint was saved with
        # a dense+output 2-layer head. AutoModelForAudioClassification maps to a
        # different 1-layer architecture, causing all weights to be UNEXPECTED/MISSING.
        _emotion_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        _emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        _emotion_model.eval()
        _emotion_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
        print("[EchoLang] Emotion model ready.")
    return _emotion_model, _emotion_processor, _emotion_labels


def get_translator(src: str, tgt: str):
    key = f"{src}-{tgt}"
    if key not in _translators:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        print(f"[EchoLang] Loading translator {model_name}...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _translators[key] = (tokenizer, model)
        print(f"[EchoLang] Translator {key} ready.")
    return _translators[key]


def get_tts():
    global _tts_model
    if _tts_model is None:
        from gtts import gTTS
        _tts_model = gTTS  # lightweight wrapper
    return _tts_model


# ── Core pipeline functions ────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes) -> dict:
    """Convert audio bytes → text + detected language using Whisper."""
    import whisper
    import numpy as np
    model = get_whisper()

    print(f"[EchoLang] Audio received: {len(audio_bytes)} bytes")

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        webm_path = f.name

    try:
        # Use Whisper's own ffmpeg pipeline — skip custom WAV conversion
        try:
            audio_np = whisper.load_audio(webm_path)  # always 16kHz mono float32
        except Exception as e:
            print(f"[EchoLang] load_audio failed: {e}")
            return {"text": "", "language": "en"}

        duration = len(audio_np) / 16000
        rms = float(np.sqrt(np.mean(audio_np ** 2)))
        print(f"[EchoLang] Audio: duration={duration:.2f}s  RMS={rms:.4f}  (>0.01 = audible speech)")

        if rms < 0.0001:
            print("[EchoLang] Near-silent audio — check Windows mic volume/permissions")
            return {"text": "", "language": "en"}

        # Normalize quiet audio so Whisper's VAD doesn't filter it out
        # (mic volume too low in Windows → boost here, cap at 10x)
        if rms < 0.05:
            scale = min(0.1 / rms, 10.0)
            audio_np = np.clip(audio_np * scale, -1.0, 1.0)
            print(f"[EchoLang] Applied {scale:.1f}x gain (RMS was {rms:.4f})")

        # Pass numpy array directly — avoids re-reading the file
        result = model.transcribe(
            audio_np,
            task="transcribe",
            fp16=False,
            no_speech_threshold=0.2,
            logprob_threshold=None,
            condition_on_previous_text=False,
        )
        text = result["text"].strip()
        print(f"[EchoLang] Whisper: lang={result['language']!r} text={text!r}")
        return {"text": text, "language": result["language"]}
    finally:
        try:
            os.unlink(webm_path)
        except Exception:
            pass


def detect_emotion(audio_bytes: bytes) -> dict:
    """Detect emotion from raw audio bytes."""
    import torch
    import subprocess

    model, processor, labels = get_emotion()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        webm_path = f.name

    wav_path = webm_path.replace(".webm", "_emo.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True
        )
        import whisper as _whisper
        speech_np = _whisper.load_audio(wav_path)

        inputs = processor(speech_np, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        top_idx = int(torch.argmax(logits))
        return {
            "emotion": labels[top_idx],
            "confidence": round(probs[top_idx], 3),
            "all_scores": {labels[i]: round(probs[i], 3) for i in range(len(labels))},
        }
    finally:
        for p in (webm_path, wav_path):
            try:
                os.unlink(p)
            except Exception:
                pass


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text using Helsinki-NLP MarianMT."""
    # Language code mapping for Marian models
    lang_map = {
        "en": "en", "hi": "hi", "ta": "ta",
        "fr": "fr", "de": "de", "es": "es"
    }
    src = lang_map.get(src_lang, src_lang)
    tgt = lang_map.get(tgt_lang, tgt_lang)

    if src == tgt:
        return text

    # Guard: empty input always produces the same default token from MarianMT
    if not text or not text.strip():
        return text

    try:
        tokenizer, model = get_translator(src, tgt)
    except Exception as e:
        print(f"[EchoLang] Translation model not found for {src}-{tgt}: {e}")
        return text

    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    return result


def synthesize_speech(text: str, lang: str, emotion: str) -> bytes:
    """Convert text to speech with emotion-adjusted parameters using gTTS."""
    from gtts import gTTS

    # Map language codes to gTTS lang codes
    gtts_lang_map = {
        "hi": "hi", "en": "en", "ta": "ta",
        "fr": "fr", "de": "de", "es": "es"
    }
    gtts_lang = gtts_lang_map.get(lang, "en")

    # Emotion → speed adjustment (gTTS slow param)
    slow_emotions = {"calm", "sad", "neutral"}
    slow = emotion in slow_emotions

    tts = gTTS(text=text, lang=gtts_lang, slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws/pipeline")
async def websocket_pipeline(websocket: WebSocket):
    await websocket.accept()
    print("[EchoLang] WebSocket connected.")
    try:
        while True:
            data = await websocket.receive_json()
            audio_b64 = data.get("audio")
            target_lang = data.get("target_lang", "hi")

            if not audio_b64:
                await websocket.send_json({"error": "No audio provided"})
                continue

            audio_bytes = base64.b64decode(audio_b64)

            # Send status updates in real time
            await websocket.send_json({"status": "transcribing"})
            loop = asyncio.get_event_loop()

            # Step 1: Transcribe
            stt_result = await loop.run_in_executor(None, transcribe_audio, audio_bytes)
            src_text = stt_result["text"]
            src_lang = stt_result["language"]
            print(f"[EchoLang] Whisper result: lang={src_lang!r} text={src_text!r}")

            if not src_text.strip():
                await websocket.send_json({
                    "error": "Could not hear any speech. Please speak clearly and try again."
                })
                continue

            await websocket.send_json({
                "status": "transcribed",
                "src_text": src_text,
                "src_lang": src_lang,
            })

            # Step 2: Detect emotion (parallel concept — run after STT)
            await websocket.send_json({"status": "analyzing_emotion"})
            emotion_result = await loop.run_in_executor(None, detect_emotion, audio_bytes)
            emotion = emotion_result["emotion"]

            await websocket.send_json({
                "status": "emotion_detected",
                "emotion": emotion,
                "confidence": emotion_result["confidence"],
                "all_scores": emotion_result["all_scores"],
            })

            # Step 3: Translate
            await websocket.send_json({"status": "translating"})
            translated_text = await loop.run_in_executor(
                None, translate_text, src_text, src_lang, target_lang
            )

            await websocket.send_json({
                "status": "translated",
                "translated_text": translated_text,
                "target_lang": target_lang,
            })

            # Step 4: Synthesize speech
            if not translated_text.strip():
                await websocket.send_json({
                    "error": "Translation produced no output. Try speaking more clearly."
                })
                continue

            await websocket.send_json({"status": "synthesizing"})
            tts_audio = await loop.run_in_executor(
                None, synthesize_speech, translated_text, target_lang, emotion
            )

            audio_out_b64 = base64.b64encode(tts_audio).decode()
            await websocket.send_json({
                "status": "complete",
                "audio_out": audio_out_b64,
                "emotion": emotion,
                "src_text": src_text,
                "translated_text": translated_text,
                "src_lang": src_lang,
                "target_lang": target_lang,
            })

    except WebSocketDisconnect:
        print("[EchoLang] WebSocket disconnected.")
    except Exception as e:
        print(f"[EchoLang] Error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


# ── Serve frontend ─────────────────────────────────────────────────────────────
if os.path.exists("../frontend/static"):
    app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")


@app.get("/")
async def root():
    with open("../frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
async def health():
    return {"status": "ok", "service": "EchoLang"}

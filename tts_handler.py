import os
import tempfile
import asyncio
from TTS.api import TTS

# Load XTTS Arabic model
tts = TTS(
    model_path="/root/.local/share/tts/tts_models--ar--custom--egtts_v0.1",
    config_path="/root/.local/share/tts/tts_models--ar--custom--egtts_v0.1/config.json"
)

reference_wav = "/tmp/reference_voice.wav"  # can be replaced with user upload

async def stream_tts_chunks(text: str, language="ar"):
    words = text.split()
    chunk_size = 8
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        output_file = os.path.join(tempfile.gettempdir(), f"tts_{i}.wav")
        tts.tts_to_file(text=chunk, file_path=output_file, speaker_wav=reference_wav, language=language)

        with open(output_file, "rb") as f:
            audio_data = f.read()

        yield audio_data
        await asyncio.sleep(0.3)  # simulate realtime streaming

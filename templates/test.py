from tts_handler import stream_tts_chunks
from llm_handler import get_gemini_response
import numpy as np 
import json
import tempfile
import wave


async def process_audio(track, pc, websocket):
    """
    Receives user audio -> detects silence -> sends to Whisper -> Gemini -> sends back as text
    """
    print("🎙 Listening...")

    frames = []
    silence_threshold = 500  # adjust sensitivity
    while True:
        frame = await track.recv()
        pcm = frame.to_ndarray()

        # Calculate volume (RMS energy)
        volume = np.sqrt(np.mean(pcm**2))
        frames.append(pcm)

        # If silence detected & enough speech collected
        if volume < silence_threshold and len(frames) > 20:
            print("Detected pause, processing speech chunk...")

            # Save audio to WAV
            audio_data = np.concatenate(frames, axis=0)
            frames = []  # reset for next utterance
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with wave.open(temp_wav.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(48000)
                wf.writeframes(audio_data.tobytes())
            temp_wav.flush()

            # Step 1: Transcribe with Whisper
            import openai
            with open(temp_wav.name, "rb") as f:
                result = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            text_input = result.text
            print("📝 Transcript:", text_input)

            # Step 2: Send transcript to Gemini
            response_text = await get_gemini_response(text_input)
            print("Gemini Response:", response_text)

            # Step 3: Send back to client over WebSocket
            await websocket.send_text(json.dumps({
                "type": "llm_response",
                "user_input": text_input,
                "gemini_reply": response_text
            }))
            print("📡 Sent response to client")

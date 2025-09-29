import json
import asyncio
import os
import tempfile
import base64
import wave
import numpy as np
import av

from fastapi import FastAPI, WebSocket
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import AudioFrame

from llm_handler import get_gemini_response
from tts_handler import stream_tts_chunks

app = FastAPI()
pcs = set()

# ----------------- Custom Track for TTS -----------------
class TTSAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__() 
        self.queue = asyncio.Queue()

    async def recv(self):
        # Wait for next audio chunk
        pcm16 = await self.queue.get()

        # Convert PCM16 numpy -> AudioFrame
        frame = AudioFrame.from_ndarray(pcm16, layout="mono")
        frame.sample_rate = 24000  # match your TTS sample rate
        return frame

    async def push_chunk(self, pcm16: np.ndarray):
        """Push PCM samples into the queue"""
        await self.queue.put(pcm16)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "offer":
                # WebRTC setup
                offer = RTCSessionDescription(sdp=message["sdp"], type=message["type"])
                pc = RTCPeerConnection()
                pcs.add(pc)

                # Create an outbound TTS track
                tts_track = TTSAudioTrack()
                pc.addTrack(tts_track)

                # Handle incoming audio
                @pc.on("track")
                def on_track(track):
                    print(f"🎙 Received track: {track.kind}")
                    if track.kind == "audio":
                        asyncio.create_task(process_audio(track, tts_track))

                # Set remote + send answer
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await websocket.send_text(json.dumps({
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type
                }))

    except Exception as e:
        print("❌ Error:", e)

@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


# ----------------- Processing Pipeline -----------------
async def process_audio(track, tts_track: TTSAudioTrack):
    """
    Receives user audio -> STT -> Gemini -> TTS -> stream back to browser.
    """
    print("🎙 Starting audio processing pipeline...")

    # Collect ~5 sec of audio
    frames = []
    for _ in range(50):
        frame = await track.recv()
        pcm = frame.to_ndarray()
        frames.append(pcm)

    audio_data = np.concatenate(frames, axis=0)

    # Save to temp wav
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(audio_data.tobytes())
    temp_wav.flush()

    # -------- STT (Whisper API example) --------
    import openai
    with open(temp_wav.name, "rb") as f:
        result = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    text_input = result.text
    print("📝 Transcript:", text_input)

    # -------- Gemini LLM --------
    response_text = await get_gemini_response(text_input)
    print("🤖 LLM Response:", response_text)

    # -------- TTS Stream Back --------
    async for audio_chunk in stream_tts_chunks(response_text, "ar"):
        # audio_chunk -> PCM16 numpy
        pcm16 = np.frombuffer(audio_chunk, dtype=np.int16)
        await tts_track.push_chunk(pcm16)
        print(f"🔊 Pushed {len(pcm16)} samples to WebRTC track")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

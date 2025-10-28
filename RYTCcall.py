# full file: turn_based_call_version.py
import asyncio
import tempfile
import os
import time
import wave
import shutil
import torch
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from av import open as av_open
from TTS.api import TTS
from faster_whisper import WhisperModel
import google.generativeai as genai
import traceback

app = FastAPI()
pcs = set()

# -------------------------
# ConversationSession: queue TTS and play on existing pc
# -------------------------
class ConversationSession:
    def __init__(self, pc, speaker_wav="/app/ref.wav"):
        self.pc = pc
        self.speaker_wav = speaker_wav
        self.audio_queue = asyncio.Queue()
        self.active = True
        self._task = asyncio.create_task(self._run())

    async def enqueue(self, text: str):
        await self.audio_queue.put(text)

    async def _run(self):
        while self.active:
            text = await self.audio_queue.get()
            try:
                tmp = os.path.join(tempfile.gettempdir(), f"conv_{id(self.pc)}_{int(time.time()*1000)}.wav")
                # generate tts (same as original)
                tts.tts_to_file(text=text, file_path=tmp, speaker_wav=self.speaker_wav, language="ar")
                # play via MediaPlayer and addTrack
                player = MediaPlayer(tmp)
                if not player.audio:
                    print("ConversationSession: player has no audio")
                else:
                    self.pc.addTrack(player.audio)
                    # estimate wait time
                    try:
                        file_size = os.path.getsize(tmp)
                        duration_est = file_size / (16000 * 2)
                    except Exception:
                        duration_est = max(1.0, len(text)/10.0)
                    await asyncio.sleep(duration_est + 0.3)
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except Exception:
                    pass
            except Exception as e:
                print("❌ ConversationSession playback error:", e)
                traceback.print_exc()
                await asyncio.sleep(0.2)

    async def close(self):
        self.active = False
        try:
            self._task.cancel()
            await self._task
        except Exception:
            pass
        try:
            await self.pc.close()
        except Exception:
            pass

# -------------------------
# Load models at startup 
# -------------------------
print("🔊 Loading Arabic TTS model...")
tts = TTS(
    model_path="/root/.local/share/tts/tts_models--ar--custom--egtts_v0.1",
    config_path="/root/.local/share/tts/tts_models--ar--custom--egtts_v0.1/config.json"
)
print("✅ TTS model loaded.")

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel("medium", device=device)
print(f"✅ Loaded faster-whisper model on {device}")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCVMkDlunj2qvsmP8gf3ExrqoqXmY3aYa0")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
print("✅ Gemini LLM configured")

async def get_gemini_response(text: str) -> str:
    try:
        instruction = (
            "تخيل إنك موظف خدمة عملاء مصري ودود بيتكلم باللهجة المصرية الطبيعية، "
            "بترد باحترام وبلُطف على العميل، ومبتستخدمش فصحى رسمية. "
            "خلي إجابتك بسيطة، قريبة من كلام الناس العادي، وماتطولش."
        )
        prompt = f"{instruction}\n\nالعميل قال: {text}\n\nرد الموظف:"
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini Error:", e)
        traceback.print_exc()
        return "عذرًا يا فندم، حصل خطأ بسيط في النظام. ممكن تعيد سؤالك؟"

# -------------------------
# Helper: simple RMS-based energy check on a wav file
# -------------------------
def wav_rms_seconds(path):
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            if wf.getsampwidth() == 2:
                dtype = np.int16
            else:
                dtype = np.int16
            audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            if audio.size == 0:
                return 0.0
            # mono vs stereo
            if wf.getnchannels() == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
            rms = np.sqrt(np.mean(audio**2))
            return rms
    except Exception as e:
        return 0.0

# -------------------------
# Frontend UI (updated): Start Call / End Call buttons and new WebRTC flow
# -------------------------
@app.get("/ui")
async def index():
    return HTMLResponse("""
    <html>
    <head>
        <meta charset="UTF-8">
        <style> /* same styles as before (omitted for brevity) */ 
            body { font-family: Arial; padding: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            button { padding: 12px 24px; font-size: 16px; cursor: pointer; margin: 5px; border: none; border-radius: 5px; background: #4CAF50; color: white; }
            #audioContainer { margin: 20px 0; }
            #log { background: #f0f0f0; padding: 15px; max-height: 400px; overflow-y: auto; border-radius: 5px; font-family: monospace; font-size: 12px; }
        </style>
    </head>
    <body>
      <div class="container">
        <h3>🎙️ Arabic Voice Call (Turn-based)</h3>
        <p>Press <strong>Start Call</strong> to open the call. Speak; stop talking to let the AI respond. End call to finish.</p>
        <button id="startBtn" onclick="startCall()">📞 Start Call</button>
        <button id="endBtn" onclick="endCall()" disabled>🔌 End Call</button>
        <div id="audioContainer"></div>
        <h4>Logs:</h4>
        <pre id="log"></pre>
      </div>

      <script>
        let pc = null;
        let localStream = null;
        let audioEl = null;
        let localTracks = [];

        function log(msg) {
          const pre = document.getElementById("log");
          const timestamp = new Date().toLocaleTimeString();
          pre.textContent += `[${timestamp}] ${msg}\\n`;
          pre.scrollTop = pre.scrollHeight;
          console.log(msg);
        }

        async function startCall() {
          try {
            document.getElementById('startBtn').disabled = true;
            document.getElementById('endBtn').disabled = false;
            log("📞 Starting call... obtaining mic permission");
            localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            pc = new RTCPeerConnection({
              iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });

            // send mic tracks
            localStream.getAudioTracks().forEach(track => {
              localTracks.push(track);
              pc.addTrack(track, localStream);
            });

            pc.ontrack = (event) => {
              log("🎧 Remote audio received");
              if (!audioEl) {
                audioEl = document.createElement('audio');
                audioEl.autoplay = true;
                audioEl.controls = true;
                const container = document.getElementById('audioContainer');
                container.innerHTML = '<p><strong>AI Response Audio:</strong></p>';
                container.appendChild(audioEl);
              }
              audioEl.srcObject = event.streams[0];
              audioEl.onplay = () => log("▶️ AI speaking");
              audioEl.onended = () => log("✅ AI finished");
            };

            pc.oniceconnectionstatechange = () => log("ICE: " + pc.iceConnectionState);

            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            // wait for ICE gather
            await new Promise(resolve => {
              if (pc.iceGatheringState === 'complete') resolve();
              else pc.onicegatheringstatechange = () => {
                if (pc.iceGatheringState === 'complete') resolve();
              };
            });

            log("📤 Sending offer to /call");
            const resp = await fetch('/call', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
            });

            if (!resp.ok) {
              const t = await resp.text();
              log("❌ /call error: " + t);
              return;
            }

            const answer = await resp.json();
            await pc.setRemoteDescription(answer);
            log("✅ Call established. Speak and pause — AI will reply when you stop.");

          } catch (err) {
            console.error(err);
            log("❌ Start call error: " + (err.message || err));
            document.getElementById('startBtn').disabled = false;
            document.getElementById('endBtn').disabled = true;
          }
        }

        async function endCall() {
          try {
            document.getElementById('startBtn').disabled = false;
            document.getElementById('endBtn').disabled = true;
            log("🔌 Ending call...");
            if (localTracks.length) {
              localTracks.forEach(t => t.stop());
            }
            if (pc) {
              await fetch('/call_end', { method: 'POST' });
              pc.close();
              pc = null;
            }
            log("✅ Call ended.");
          } catch (err) {
            log("❌ End call error: " + err.message);
          }
        }
      </script>
    </body>
    </html>
    """)


# -------------------------
# /call endpoint: accept bidirectional offer and set up recording + session
# -------------------------
@app.post("/call")
async def call(request: Request):
    """
    New endpoint for continuous turn-based call.
    Receives browser offer (which includes the mic as outgoing track).
    Server will:
      - attach a ConversationSession to pc (for TTS playback)
      - set up a MediaRecorder to record incoming user audio to temporary files
      - run a small background loop that detects "silence" and triggers transcription -> LLM -> enqueue TTS
    """
    try:
        data = await request.json()
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
            ]
        )

        pc = RTCPeerConnection(configuration=config)
        pcs.add(pc)
        print("✅ New PC created for call:", id(pc))

        # attach ConversationSession for outgoing TTS playback
        ref_wav = "/app/ref.wav"
        session = ConversationSession(pc, speaker_wav=ref_wav)
        pc.session = session

        # Incoming recording state per pc
        pc._recorder = None
        pc._last_segment = None
        pc._recording_dir = os.path.join(tempfile.gettempdir(), f"call_{id(pc)}")
        os.makedirs(pc._recording_dir, exist_ok=True)
        pc._last_write_time = time.time()
        pc._processing_lock = asyncio.Lock()
        pc._running = True

        # ontrack -> start a recorder writing to a rolling file
        @pc.on("track")
        def on_track(track):
            print(f"pc {id(pc)}: track received: {track.kind}")
            # create unique chunk file
            chunk_path = os.path.join(pc._recording_dir, f"in_{int(time.time()*1000)}.wav")
            recorder = MediaRecorder(chunk_path, format="wav")
            pc._recorder = recorder
            # start recorder for this track
            asyncio.ensure_future(recorder.start())
            recorder.addTrack(track)

            # update last write time periodically by spawning a tiny watchdog task
            async def watchdog():
                try:
                    while pc._running:
                        await asyncio.sleep(0.8)
                        pc._last_write_time = time.time()
                except asyncio.CancelledError:
                    pass

            asyncio.ensure_future(watchdog())

        # monitor loop: watches recorder files and when silence detected -> process segment
        async def monitor_and_process():
            print("monitor_and_process started for pc", id(pc))
            last_seen_size = 0
            last_changed = time.time()
            pending_files = []  # files to process in order
            while pc._running:
                try:
                    await asyncio.sleep(0.9)
                    # list wav files in dir
                    files = sorted([os.path.join(pc._recording_dir, f) for f in os.listdir(pc._recording_dir) if f.endswith(".wav")])
                    # if no files yet, continue
                    if not files:
                        continue
                    # pick oldest file to check
                    candidate = files[0]
                    size = os.path.getsize(candidate)
                    # if size changed since last check -> update last_changed
                    if size != last_seen_size:
                        last_changed = time.time()
                        last_seen_size = size
                        continue
                    # if size hasn't changed for a threshold and size > min -> consider this a finished segment
                    silence_threshold = 1.1  # seconds of no change
                    min_bytes = 4000
                    if (time.time() - last_changed) > silence_threshold and size > min_bytes:
                        # mark this file for processing (move it)
                        proc_path = os.path.join(pc._recording_dir, f"proc_{int(time.time()*1000)}.wav")
                        try:
                            shutil.move(candidate, proc_path)
                        except Exception:
                            # maybe file got removed; skip
                            continue
                        print("Detected finished speech segment:", proc_path)
                        # transcribe and respond (do not block monitor)
                        asyncio.create_task(process_segment_and_respond(pc, proc_path))
                        # reset trackers
                        last_seen_size = 0
                        last_changed = time.time()
                except Exception as e:
                    print("monitor error:", e)
                    traceback.print_exc()
                    await asyncio.sleep(0.5)

        # processing each finished segment: STT -> LLM -> enqueue TTS
        async def process_segment_and_respond(pc_local, wav_path):
            async with pc_local._processing_lock:
                try:
                    print("Processing segment:", wav_path)
                    # transcribe using whisper
                    segments, info = whisper_model.transcribe(wav_path, language="ar")
                    text_input = " ".join([seg.text for seg in segments]).strip()
                    print("Transcription:", text_input)
                    if not text_input:
                        print("Empty transcription, skipping.")
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
                        return
                    # call Gemini
                    response_text = await get_gemini_response(text_input)
                    print("Gemini response:", response_text)
                    # enqueue to session for playback
                    if hasattr(pc_local, "session"):
                        await pc_local.session.enqueue(response_text)
                        print("Enqueued response to session.")
                    else:
                        print("No session to enqueue to.")
                except Exception as e:
                    print("process_segment error:", e)
                    traceback.print_exc()
                finally:
                    try:
                        if os.path.exists(wav_path):
                            os.remove(wav_path)
                    except Exception:
                        pass

        # set remote description and create answer
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # start monitor
        pc._monitor_task = asyncio.create_task(monitor_and_process())

        print("Call setup complete for pc", id(pc))
        return JSONResponse({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })

    except Exception as e:
        print("FATAL in /call:", e)
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)


# endpoint to end a call (frontend will call it)
@app.post("/call_end")
async def call_end():
    # find all pcs and close them (simple approach)
    try:
        for pc in list(pcs):
            pc._running = False
            # stop monitor task
            if hasattr(pc, "_monitor_task"):
                try:
                    pc._monitor_task.cancel()
                except Exception:
                    pass
            # close session
            if hasattr(pc, "session"):
                try:
                    await pc.session.close()
                except Exception:
                    pass
            try:
                await pc.close()
            except Exception:
                pass
            pcs.discard(pc)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -------------------------
# Keep your previous /voice-conversation and /offer endpoints intact
# (I left them as-is so you still can use recording-upload flows if needed)
# -------------------------

@app.post("/voice-conversation")
async def voice_conversation(audio: UploadFile = File(...)):
    """Handle voice input: STT → LLM → return text for TTS"""
    try:
        print("\n" + "="*60)
        print("🎤 VOICE CONVERSATION PIPELINE STARTED")
        print("="*60)

        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        content = await audio.read()
        temp_audio.write(content)
        temp_audio.close()

        print(f"✅ Audio received and saved: {temp_audio.name}")

        segments, info = whisper_model.transcribe(temp_audio.name, language="ar")
        text_input = " ".join([segment.text for segment in segments])

        print(f"✅ Transcription complete: '{text_input}'")

        response_text = await get_gemini_response(text_input)
        print(f"✅ LLM response: '{response_text}'")

        os.unlink(temp_audio.name)

        # enqueue on any active session if exists
        if pcs:
            pc_with_session = None
            for pc in pcs:
                if hasattr(pc, "session"):
                    pc_with_session = pc
                    break
            if pc_with_session:
                await pc_with_session.session.enqueue(response_text)

        return JSONResponse({"transcription": text_input, "llm_response": response_text})

    except Exception as e:
        print("Error in voice_conversation:", e)
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/offer")
async def offer(request: Request):
    """Original /offer kept for backward compatibility (play once then stop)"""
    try:
        data = await request.json()
        text = data.get("text", "مرحبا بكم")

        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
            ]
        )

        pc = RTCPeerConnection(configuration=config)
        pcs.add(pc)

        # generate initial TTS and play (your original flow)
        ref_wav = "/app/ref.wav"
        temp_audio = os.path.join(tempfile.gettempdir(), f"output_{id(pc)}.wav")
        if not os.path.exists(ref_wav):
            return JSONResponse({"error": f"Reference file not found: {ref_wav}"}, status_code=500)

        tts.tts_to_file(text=text, file_path=temp_audio, speaker_wav=ref_wav, language="ar")
        player = MediaPlayer(temp_audio)
        audio_track = pc.addTrack(player.audio)

        @pc.on("iceconnectionstatechange")
        async def on_ice_state():
            print("ICE state:", pc.iceConnectionState)

        @pc.on("connectionstatechange")
        async def on_connection_state():
            print("connection state:", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # attach session
        try:
            session = ConversationSession(pc, speaker_wav=ref_wav)
            pc.session = session
        except Exception as e:
            print("Failed to attach session:", e)

        return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    except Exception as e:
        print("FATAL /offer:", e)
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)


# graceful shutdown
@app.on_event("shutdown")
async def on_shutdown():
    print("Shutting down. Closing PCs...")
    coros = []
    for pc in list(pcs):
        if hasattr(pc, "session"):
            try:
                coros.append(pc.session.close())
            except Exception:
                pass
        try:
            coros.append(pc.close())
        except Exception:
            pass
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
    print("Closed.")


if __name__ == "__main__":
    import uvicorn
    print("Starting server on :5002")
    uvicorn.run(app, host="0.0.0.0", port=5002)

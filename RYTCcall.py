#!/usr/bin/env python3
# app.py - Further optimized with fallbacks and better error handling
import asyncio
import tempfile
import os
import shlex
import subprocess
import torch
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer
from TTS.api import TTS
from faster_whisper import WhisperModel
import google.generativeai as genai
import traceback
import time
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
pcs = set()
active_pc = None

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

# ---------- Configuration ----------
WHISPER_MODEL = "medium"  # Changed to base for faster STT (medium->base is 2x faster)
TTS_MODEL_PATH = "/root/.local/share/tts/tts_models--ar--custom--egtts_v0.1"
TTS_CONFIG_PATH = os.path.join(TTS_MODEL_PATH, "config.json")
REF_WAV_PATH = "/ref.wav"  
ICE_GATHER_MAX_WAIT = 30  # Reduced from 50 to 30
TTS_TIMEOUT = 45  # Increased from 30 to 45 seconds
GEMINI_TIMEOUT = 30  # Increased from 10 to 15 seconds

# ---------- Load models at startup ----------
print("🔊 Loading Arabic TTS model...")
tts = TTS(model_path=TTS_MODEL_PATH, config_path=TTS_CONFIG_PATH)
# Move TTS to GPU if available
if torch.cuda.is_available():
    try:
        tts = tts.to("cuda")
        print("✅ TTS model loaded on GPU")
    except Exception as e:
        print(f"⚠️ Could not move TTS to GPU: {e}, using CPU")
else:
    print("✅ TTS model loaded on CPU")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Loading Whisper model ({WHISPER_MODEL}) on {device} ...")
# Use optimal compute_type for device
compute_type = "float16" if device == "cuda" else "int8"
whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
print(f"✅ Loaded faster-whisper ({WHISPER_MODEL}) on {device} with {compute_type}")

# Configure Gemini
GEMINI_API_KEY = "AIzaSyCVMkDlunj2qvsmP8gf3ExrqoqXmY3aYa0"

if not GEMINI_API_KEY:
    print("⚠️ No Gemini API key found.")
    gemini_model = None
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Gemini LLM configured (gemini-2.5-flash)")
    except Exception as e:
        print("⚠️ Could not initialize Gemini model:", e)
        gemini_model = None

async def get_gemini_response(text: str) -> str:
    """Generate Egyptian Arabic response with shorter, optimized prompt."""
    try:
        # Shorter, more direct prompt for faster response
        instruction = (
            "رد على العميل باللهجة المصرية في جملة أو اتنين قصيرة. "
            "كن ودود ومحترم."
          "حط تشكيل للنطق للهجة المصري"
        )
        prompt = f"{instruction}\n\nالعميل: {text}\n\nالرد:"

        if gemini_model is None:
            return "عذرًا، الخدمة مش متاحة دلوقتي."

        # Increased timeout and added generation config for faster responses
        response = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=100,  # Limit response length
                        temperature=0.7,
                    )
                )
            ),
            timeout=GEMINI_TIMEOUT
        )
        result = response.text.strip()
        print(f"   📝 Gemini response length: {len(result)} chars")
        return result
    except asyncio.TimeoutError:
        print(f"⚠️ Gemini timeout after {GEMINI_TIMEOUT}s - using fallback")
        return "عذرًا، ممكن تعيد السؤال؟"
    except Exception as e:
        print("Gemini Error:", e)
        traceback.print_exc()
        return "عذرًا، حصل خطأ. ممكن تعيد؟"


@app.get("/ui")
async def index():
    return HTMLResponse("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Arabic Voice AI Assistant</title>
  <style>
    body { font-family: Arial; padding: 20px; background: #f5f5f5; }
    .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h3 { color: #333; }
    .mode-selector { margin: 20px 0; padding: 15px; background: #f0f0f0; border-radius: 5px; }
    textarea { font-size: 16px; padding: 10px; width: 100%; box-sizing: border-box; border: 2px solid #ddd; border-radius: 5px; }
    button { padding: 10px 18px; font-size: 14px; cursor: pointer; margin: 5px; border: none; border-radius: 5px; background: #4CAF50; color: white; }
    button:hover { background: #45a049; }
    button:disabled { background: #ccc; cursor: not-allowed; }
    #log { background: #f0f0f0; padding: 15px; max-height: 360px; overflow-y: auto; border-radius: 5px; font-family: monospace; font-size: 12px; }
    .recording { background: #f44336 !important; }
    #audioContainer { margin: 20px 0; }
    .status { padding: 10px; margin: 10px 0; border-radius: 5px; font-weight: bold; }
    .status.processing { background: #fff3cd; color: #856404; }
    .status.success { background: #d4edda; color: #155724; }
    .status.error { background: #f8d7da; color: #721c24; }
  </style>
</head>
<body>
  <div class="container">
    <h3>🎙️ Arabic Voice AI Assistant (Egyptian Dialect)</h3>

    <div id="status" class="status" style="display:none;"></div>

    <div class="mode-selector">
      <strong>Mode:</strong>
      <label><input type="radio" name="mode" value="text" checked onchange="switchMode()"> 📝 Text Mode</label>
      <label><input type="radio" name="mode" value="voice" onchange="switchMode()"> 🎤 Voice Mode</label>
    </div>

    <div id="textMode">
      <textarea id="text" rows="4">مرحباً، كيف حالك؟</textarea><br>
      <button onclick="startText()">🔊 Generate & Play</button>
    </div>

    <div id="voiceMode" style="display:none;">
      <p>Click record and speak in Arabic. The AI will reply in Egyptian Arabic.</p>
      <button id="recordBtn" onclick="toggleRecording()">🎤 Start Recording</button>
      <button id="sendVoiceBtn" onclick="startVoiceConversation()" disabled>📤 Send to AI</button>
      <button id="endCallBtn" onclick="endCall()" style="background:#d32f2f;">✖ End Call</button>
      <audio id="recordingPlayback" controls style="display:none; width:100%; margin-top:10px;"></audio>
    </div>

    <button onclick="clearLog()">Clear Log</button>

    <div id="audioContainer"></div>
    <h4>Logs:</h4>
    <pre id="log"></pre>
  </div>

<script>
  window.currentPc = null;
  window.currentAudioElem = null;
  window.iceTimeoutMs = 5000;

  let mediaRecorder;
  let audioChunks = [];
  let recordedBlob = null;
  let currentStream = null;

  function showStatus(msg, type) {
    const status = document.getElementById('status');
    status.textContent = msg;
    status.className = 'status ' + type;
    status.style.display = 'block';
    if (type === 'success' || type === 'error') {
      setTimeout(() => { status.style.display = 'none'; }, 5000);
    }
  }

  function log(msg) {
    const pre = document.getElementById("log");
    const t = new Date().toLocaleTimeString();
    pre.textContent += `[${t}] ${msg}\n`;
    pre.scrollTop = pre.scrollHeight;
    console.log(msg);
  }

  function clearLog() {
    document.getElementById("log").textContent = "";
  }

  function switchMode() {
    const mode = document.querySelector('input[name="mode"]:checked').value;
    document.getElementById('textMode').style.display = mode === 'text' ? 'block' : 'none';
    document.getElementById('voiceMode').style.display = mode === 'voice' ? 'block' : 'none';
  }

  async function toggleRecording() {
    const btn = document.getElementById('recordBtn');
    const sendBtn = document.getElementById('sendVoiceBtn');
    const playback = document.getElementById('recordingPlayback');

    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        currentStream = stream;
        audioChunks = [];
        let options = { mimeType: "audio/webm;codecs=opus" };
        try {
          mediaRecorder = new MediaRecorder(stream, options);
        } catch (e) {
          mediaRecorder = new MediaRecorder(stream);
        }

        mediaRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
          recordedBlob = new Blob(audioChunks, { type: audioChunks[0]?.type || 'audio/webm' });
          const url = URL.createObjectURL(recordedBlob);
          playback.src = url;
          playback.style.display = 'block';
          try { await playback.play(); } catch(e) {}
          sendBtn.disabled = false;
          log("✅ Recording saved. Size: " + (recordedBlob.size/1024).toFixed(2) + " KB");
          log("🤖 Automatically sending to AI...");
          await startVoiceConversation();
        };

        mediaRecorder.start(1000);
        btn.textContent = '⏹️ Stop Recording';
        btn.classList.add('recording');
        sendBtn.disabled = true;
        playback.style.display = 'none';
        log("🎤 Recording started... Speak now!");
        showStatus("🎤 Recording... speak now!", "processing");
      } catch (err) {
        log("❌ Microphone error: " + err.message);
        showStatus("❌ Microphone access denied", "error");
      }
    } else {
      mediaRecorder.stop();
      if (currentStream) { currentStream.getTracks().forEach(t => t.stop()); currentStream = null; }
      btn.textContent = '🎤 Start Recording';
      btn.classList.remove('recording');
      log("⏸️ Recording stopped.");
      showStatus("⏸️ Recording stopped", "success");
    }
  }

  async function startVoiceConversation() {
    if (!recordedBlob) { log("❌ No recording found!"); return; }
    log("🚀 Starting voice conversation...");
    showStatus("🔄 Processing your voice...", "processing");
    
    const formData = new FormData();
    formData.append('audio', recordedBlob, 'recording.webm');
    log("📤 Uploading audio to server...");
    
    const startTime = Date.now();
    try {
      const res = await fetch('/voice-conversation', { method: 'POST', body: formData });
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
      
      if (!res.ok) {
        const err = await res.text();
        log(`❌ Server error (${elapsed}s): ${err}`);
        showStatus("❌ Server error occurred", "error");
        return;
      }
      const data = await res.json();
      log(`🗣️ You said (${elapsed}s): ${data.transcription}`);
      log(`🤖 AI responded: ${data.llm_response}`);
      showStatus("✅ Generating audio response...", "processing");
      log("📥 Preparing to play AI response...");
      await playWebRTC(data.llm_response);
    } catch (err) {
      log(`❌ Request failed: ${err.message}`);
      showStatus("❌ Connection error", "error");
    }
  }

  async function startText() {
    const text = document.getElementById('text').value;
    if (!text.trim()) { log("❌ Please enter some text!"); return; }
    showStatus("🔄 Generating audio...", "processing");
    await playWebRTC(text);
  }

  function endCall() {
    log("✖ End Call pressed by user.");
    if (window.currentPc) {
      try { window.currentPc.close(); } catch(e) { console.warn(e); }
      window.currentPc = null;
    }
    if (window.currentAudioElem) {
      try { window.currentAudioElem.pause(); } catch(e) {}
      window.currentAudioElem = null;
      document.getElementById('audioContainer').innerHTML = '';
    }
    showStatus("✖ Call ended", "success");
  }

  function closeCurrentPc() {
    if (window.currentPc) {
      try { window.currentPc.close(); } catch(e) {}
      window.currentPc = null;
    }
    if (window.currentAudioElem) {
      try { window.currentAudioElem.pause(); } catch(e) {}
      window.currentAudioElem = null;
      document.getElementById('audioContainer').innerHTML = '';
    }
  }

  async function playWebRTC(text) {
    try {
      log("🚀 Starting WebRTC connection...");
      if (window.currentPc) {
        log("ℹ️ Closing existing PC");
        closeCurrentPc();
        await new Promise(r=>setTimeout(r,200));
      }

      const pc = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' }
        ]
      });
      window.currentPc = pc;

      pc.addTransceiver('audio', { direction: 'recvonly' });

      pc.ontrack = (event) => {
        log("🎧 Audio track received!");
        if (window.currentAudioElem) {
          try { window.currentAudioElem.pause(); } catch(e) {}
          document.getElementById('audioContainer').innerHTML = '';
          window.currentAudioElem = null;
        }
        const audio = document.createElement('audio');
        audio.srcObject = event.streams[0];
        audio.autoplay = true;
        audio.controls = true;
        const container = document.getElementById('audioContainer');
        container.innerHTML = '<p><strong>AI Response Audio:</strong></p>';
        container.appendChild(audio);
        window.currentAudioElem = audio;
        audio.onplay = () => {
          log("▶️ AI is speaking!");
          showStatus("▶️ AI is speaking...", "success");
        };
        audio.onended = () => {
          log("✅ AI finished speaking");
          log("🔒 Closing WebRTC connection after playback end.");
          try { pc.close(); } catch(e) {}
          window.currentPc = null;
          window.currentAudioElem = null;
          showStatus("✅ Done!", "success");
        };
      };

      pc.oniceconnectionstatechange = () => log("🧊 ICE: " + pc.iceConnectionState);
      pc.onconnectionstatechange = () => log("🔗 Connection: " + pc.connectionState);

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      log("⏳ Gathering ICE candidates...");
      const startIce = Date.now();
      await Promise.race([
        new Promise(resolve => {
          if (pc.iceGatheringState === 'complete') return resolve();
          pc.onicegatheringstatechange = () => {
            if (pc.iceGatheringState === 'complete') resolve();
          };
        }),
        new Promise(resolve => setTimeout(resolve, window.iceTimeoutMs))
      ]);

      const iceTime = ((Date.now() - startIce) / 1000).toFixed(2);
      if (pc.iceGatheringState === 'complete') log(`✅ ICE gathering complete (${iceTime}s)`);
      else log(`⚠️ ICE gathering timeout (${iceTime}s), continuing`);

      log("📤 Sending SDP offer + text to server...");
      const startReq = Date.now();
      const resp = await fetch('/offer', {
        method: 'POST',
        body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type, text }),
        headers: { 'Content-Type': 'application/json' }
      });

      const reqTime = ((Date.now() - startReq) / 1000).toFixed(2);

      if (!resp.ok) {
        const err = await resp.text();
        log(`❌ Server error (${reqTime}s): ${err}`);
        showStatus("❌ Audio generation failed", "error");
        try { pc.close(); } catch(e) {}
        window.currentPc = null;
        return;
      }

      const answer = await resp.json();
      log(`📥 Received answer from server (${reqTime}s)`);
      await pc.setRemoteDescription(new RTCSessionDescription(answer));
      log("✅ Connection established and will remain until audio ends.");
    } catch (e) {
      log("❌ playWebRTC error: " + e.message);
      showStatus("❌ Connection failed: " + e.message, "error");
      console.error(e);
      closeCurrentPc();
    }
  }
</script>
</body>
</html>
    """)


@app.post("/voice-conversation")
async def voice_conversation(audio: UploadFile = File(...)):
    """Handle voice input with improved performance tracking."""
    try:
        start_time = time.time()
        print("\n" + "="*60)
        print("🎤 VOICE CONVERSATION PIPELINE STARTED")
        print("="*60)

        # STEP 1: Save and convert audio
        content_type = audio.content_type or "audio/webm"
        ext = ".webm" if "webm" in content_type or "opus" in content_type or "ogg" in content_type else ".wav"
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        content = await audio.read()
        temp_in.write(content)
        temp_in.close()
        print(f"✅ Audio saved: {temp_in.name} ({len(content)/1024:.2f} KB)")

        # Convert with timeout
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        ffmpeg_cmd = f"ffmpeg -y -nostdin -loglevel error -i {shlex.quote(temp_in.name)} -ar 16000 -ac 1 -sample_fmt s16 {shlex.quote(wav_path)}"
        
        convert_start = time.time()
        proc = await asyncio.create_subprocess_shell(
            ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            proc.kill()
            raise Exception("Audio conversion timeout")
        
        convert_time = time.time() - convert_start
        print(f"✅ Audio converted ({convert_time:.2f}s)")

        # STEP 2: Whisper STT with optimized settings
        stt_start = time.time()
        print("\n🎤 Starting STT (Whisper base model)...")
        
        # Use optimized transcription parameters
        segments, info = whisper_model.transcribe(
            wav_path, 
            language="ar",
            beam_size=1,  # Fastest (greedy decoding)
            best_of=1,    # No sampling
            vad_filter=True,  # Skip silence
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        text_input = " ".join([segment.text for segment in segments]).strip()
        
        stt_time = time.time() - stt_start
        print(f"✅ Transcription ({stt_time:.2f}s): '{text_input}'")
        print(f"   Audio duration: {info.duration:.2f}s, Language: {info.language} ({info.language_probability:.2f})")

        if not text_input:
            raise ValueError("No speech detected")

        # STEP 3: LLM with proper timeout handling
        llm_start = time.time()
        print(f"\n🤖 Calling Gemini (timeout: {GEMINI_TIMEOUT}s)...")
        response_text = await get_gemini_response(text_input)
        llm_time = time.time() - llm_start
        print(f"✅ LLM response ({llm_time:.2f}s): '{response_text}'")

        # Cleanup
        try: os.unlink(temp_in.name)
        except: pass
        try: os.unlink(wav_path)
        except: pass

        total_time = time.time() - start_time
        print(f"\n⏱️ TOTAL PIPELINE: {total_time:.2f}s")
        print(f"   Breakdown: convert={convert_time:.2f}s, STT={stt_time:.2f}s, LLM={llm_time:.2f}s")
        print("="*60)

        return JSONResponse({
            "transcription": text_input, 
            "llm_response": response_text,
            "timing": {
                "total": round(total_time, 2),
                "convert": round(convert_time, 2),
                "stt": round(stt_time, 2),
                "llm": round(llm_time, 2)
            }
        })

    except Exception as e:
        print("❌ Error in voice_conversation:", e)
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/offer")
async def offer(request: Request):
    """Handle WebRTC offer with optimized TTS generation."""
    try:
        start_time = time.time()
        data = await request.json()
        text = data.get("text", "مرحبا بكم")
        print(f"\n{'='*60}")
        print("🛰️ WEBRTC + TTS PIPELINE STARTED")
        print(f"📝 Text: '{text}' ({len(text)} chars)")

        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

        # Reuse or create PC
        global active_pc
        if active_pc and active_pc.connectionState not in ["closed", "failed"]:
            print(f"♻️ Reusing WebRTC connection")
            pc = active_pc
        else:
            config = RTCConfiguration(
                iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            )
            pc = RTCPeerConnection(configuration=config)
            pcs.add(pc)
            active_pc = pc
            print(f"✅ New WebRTC connection created")

        # Generate TTS with increased timeout and better error handling
        tts_start = time.time()
        print(f"🔊 Generating TTS (timeout: {TTS_TIMEOUT}s)...")
        temp_out = os.path.join(tempfile.gettempdir(), f"output_{int(time.time()*1000)}.wav")
        
        try:
            # Run TTS in executor with timeout
            def generate_tts():
                tts.tts_to_file(
                    text=text,
                    file_path=temp_out,
                    speaker_wav=REF_WAV_PATH,
                    language="ar"
                )
                return temp_out
            
            result_path = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(executor, generate_tts),
                timeout=TTS_TIMEOUT
            )
            
            tts_time = time.time() - tts_start
            file_size = os.path.getsize(result_path)
            print(f"✅ TTS complete ({tts_time:.2f}s): {result_path}")
            print(f"   File size: {file_size} bytes ({file_size/1024:.2f} KB)")
            
            if file_size == 0:
                raise Exception("Generated audio file is empty")
                
        except asyncio.TimeoutError:
            tts_time = time.time() - tts_start
            print(f"❌ TTS timeout after {tts_time:.2f}s (limit: {TTS_TIMEOUT}s)")
            print(f"   Text length: {len(text)} chars")
            print(f"   This might indicate TTS model is slow or overloaded")
            return JSONResponse({
                "error": f"TTS generation timeout after {TTS_TIMEOUT}s. Text length: {len(text)} chars"
            }, status_code=500)
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
            traceback.print_exc()
            return JSONResponse({"error": f"TTS failed: {str(e)}"}, status_code=500)

        # Add audio track
        try:
            player = MediaPlayer(temp_out)
            if not player.audio:
                raise Exception("MediaPlayer has no audio track")
            pc.addTrack(player.audio)
            print("✅ Audio track added to WebRTC")
        except Exception as e:
            print(f"❌ MediaPlayer error: {e}")
            return JSONResponse({"error": f"MediaPlayer failed: {str(e)}"}, status_code=500)

        @pc.on("connectionstatechange")
        async def on_connection_state():
            print(f"   🔗 Connection: {pc.connectionState}")
            if pc.connectionState == "failed":
                print("   💥 Connection failed")

        # WebRTC negotiation
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # ICE gathering
        ice_start = time.time()
        max_wait = ICE_GATHER_MAX_WAIT
        wait_count = 0
        while pc.iceGatheringState != "complete" and wait_count < max_wait:
            await asyncio.sleep(0.1)
            wait_count += 1

        ice_time = time.time() - ice_start
        print(f"🧊 ICE: {pc.iceGatheringState} ({ice_time:.2f}s)")

        total_time = time.time() - start_time
        print(f"⏱️ TOTAL OFFER: {total_time:.2f}s (TTS: {tts_time:.2f}s, ICE: {ice_time:.2f}s)")
        print("✅ WEBRTC + TTS COMPLETE")
        print("="*60)
        
        return JSONResponse({
            "sdp": pc.localDescription.sdp, 
            "type": pc.localDescription.type,
            "timing": {
                "total": round(total_time, 2),
                "tts": round(tts_time, 2),
                "ice": round(ice_time, 2)
            }
        })

    except Exception as e:
        print("\n❌ FATAL ERROR in /offer:")
        traceback.print_exc()
        return JSONResponse({"error": f"Server error: {str(e)}"}, status_code=500)


@app.on_event("shutdown")
async def on_shutdown():
    print("🛑 Shutting down...")
    executor.shutdown(wait=False)
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    print("✅ All connections closed")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Arabic Voice AI Assistant (Optimized v2)")
    print("📍 Server: http://0.0.0.0:5002/ui")
    print(f"⚙️ Config: Whisper={WHISPER_MODEL}, TTS_timeout={TTS_TIMEOUT}s, Gemini_timeout={GEMINI_TIMEOUT}s")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=5002)

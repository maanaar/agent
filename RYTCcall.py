import asyncio
import tempfile
import os
import torch
import traceback
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer
from TTS.api import TTS
from faster_whisper import WhisperModel
import google.generativeai as genai

app = FastAPI()
pcs = set()

# =====================================================
# 🧠 MODEL LOADING
# =====================================================
print("🔊 Loading Arabic TTS model...")
tts = TTS(
    model_path="/root/.local/share/tts/tts_models--ar--custom--egtts_v0.1",
    config_path="/root/.local/share/tts/tts_models--ar--custom--egtts_v0.1/config.json"
)
print("✅ TTS model loaded.")

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel("medium", device=device)
print(f"✅ Whisper loaded on {device}")

genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyCVMkDlunj2qvsmP8gf3ExrqoqXmY3aYa0"))
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
print("✅ Gemini model ready")

# =====================================================
# 🔮 GEMINI RESPONSE
# =====================================================
async def get_gemini_response(text: str) -> str:
    try:
        instruction = (
            "تخيل إنك موظف خدمة عملاء مصري ودود بيتكلم باللهجة المصرية الطبيعية. "
          ",خلي إجابتك قصيرة، لطيفة، ومهذبة و حط تشكيل."
        )
        prompt = f"{instruction}\n\nالعميل قال: {text}\n\nرد الموظف:"
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini Error:", e)
        return "عذرًا، حصل خطأ بسيط. ممكن تعيد سؤالك؟"

# =====================================================
# 🧩 FRONTEND
# =====================================================
@app.get("/ui")
async def index():
    return HTMLResponse("""
<html>
<head>
<meta charset="UTF-8">
<style>
body { font-family: Arial; background:#f7f7f7; padding:20px; }
.container { background:white; max-width:800px; margin:auto; padding:25px; border-radius:12px;
 box-shadow:0 2px 10px rgba(0,0,0,0.1);}
button { padding:12px 20px; font-size:16px; border:none; border-radius:8px; cursor:pointer; margin:5px;}
#log { background:#f0f0f0; padding:10px; border-radius:6px; height:300px; overflow-y:auto; font-size:13px;}
.recording { background:#d9534f; color:white; }
</style>
</head>
<body>
<div class="container">
<h2>🎧 Arabic Voice AI Call</h2>
<button id="callBtn" onclick="toggleCall()">🎤 Start Call</button>
<button id="endBtn" onclick="endCall()" style="display:none;background:#999;color:white;">End Call</button>
<audio id="aiAudio" controls autoplay style="width:100%; margin-top:10px;"></audio>
<pre id="log"></pre>
</div>

<script>
let mediaRecorder, audioChunks = [];
let pc, stream;
let silenceTimeout, isRecording = false;
let callActive = false;
const logEl = document.getElementById("log");

function log(msg){ 
  const t = new Date().toLocaleTimeString();
  logEl.textContent += `[${t}] ${msg}\\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

async function toggleCall(){
  if(!callActive){ startCall(); }
  else{ endCall(); }
}

async function startCall(){
  log("📞 Starting call...");
  callActive = true;
  document.getElementById("callBtn").textContent = "⏹️ Stop Listening";
  document.getElementById("endBtn").style.display = "inline-block";

  // Setup WebRTC connection
  pc = new RTCPeerConnection({ iceServers: [{urls:'stun:stun.l.google.com:19302'}] });
  pc.addTransceiver("audio", {direction:"recvonly"});
  pc.ontrack = (e)=>{
    log("🎧 AI response received!");
    const aiAudio = document.getElementById("aiAudio");
    aiAudio.srcObject = e.streams[0];
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const res = await fetch("/offer", {
    method:"POST",
    body:JSON.stringify({sdp:offer.sdp, type:offer.type, text:"أهلاً"}),
    headers:{'Content-Type':'application/json'}
  });
  const ans = await res.json();
  await pc.setRemoteDescription(ans);
  log("✅ WebRTC connection ready.");

  // Start recording
  startRecording();
}

async function startRecording(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    const audioCtx = new AudioContext();
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    source.connect(analyser);
    const dataArray = new Float32Array(analyser.fftSize);

    function detectSilence(){
      analyser.getFloatTimeDomainData(dataArray);
      const rms = Math.sqrt(dataArray.reduce((s,a)=>s+a*a,0)/dataArray.length);
      if(rms < 0.015 && isRecording){ // silence
        clearTimeout(silenceTimeout);
        silenceTimeout = setTimeout(stopRecording, 1500);
      } else {
        clearTimeout(silenceTimeout);
      }
      if(isRecording) requestAnimationFrame(detectSilence);
    }

    mediaRecorder.ondataavailable = e=>audioChunks.push(e.data);
    mediaRecorder.onstop = sendAudio;
    mediaRecorder.start();
    isRecording = true;
    log("🎤 Recording started...");
    detectSilence();

  } catch(e){
    log("❌ Mic error: "+e.message);
  }
}

async function stopRecording(){
  if(!isRecording) return;
  isRecording = false;
  mediaRecorder.stop();
  stream.getTracks().forEach(t=>t.stop());
  log("⏸️ Detected silence — processing...");
}

async function sendAudio(){
  const blob = new Blob(audioChunks, {type:"audio/wav"});
  const form = new FormData();
  form.append("audio", blob, "speech.wav");
  const res = await fetch("/voice-conversation", {method:"POST", body:form});
  const data = await res.json();
  log("🗣️ You: "+data.transcription);
  log("🤖 AI: "+data.llm_response);

  // Play AI voice via WebRTC
  await playResponse(data.llm_response);

  // OPTION 1️⃣ Manual restart after AI finishes (default)
  // User clicks again to record new message

  // OPTION 2️⃣ Auto restart after AI finishes (uncomment to enable)
  // setTimeout(()=>{ if(callActive) startRecording(); }, 2000);
}

async function playResponse(text){
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const res = await fetch("/offer", {
    method:"POST",
    body:JSON.stringify({sdp:offer.sdp, type:offer.type, text:text}),
    headers:{'Content-Type':'application/json'}
  });
  const ans = await res.json();
  await pc.setRemoteDescription(ans);
}

function endCall(){
  log("🛑 Call ended.");
  callActive = false;
  if(isRecording) stopRecording();
  if(stream) stream.getTracks().forEach(t=>t.stop());
  if(pc) pc.close();
  document.getElementById("callBtn").textContent = "🎤 Start Call";
  document.getElementById("endBtn").style.display = "none";
}
</script>
</body>
</html>
""")

# =====================================================
# 🎤 VOICE CONVERSATION PIPELINE
# =====================================================
@app.post("/voice-conversation")
async def voice_conversation(audio: UploadFile = File(...)):
    """Handle voice input: STT → LLM → return text for TTS and stream TTS onto the open pc."""
    try:
        print("\n" + "="*60)
        print("🎤 VOICE CONVERSATION PIPELINE STARTED")
        print("="*60)

        # ---------------------- STEP 1: Save Audio ----------------------
        print("\n📥 STEP 1: Receiving audio file...")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        content = await audio.read()
        temp_audio.write(content)
        temp_audio.close()

        print(f"✅ Audio received and saved -> {temp_audio.name} ({len(content)} bytes)")

        # ---------------------- STEP 2: STT (Whisper) ----------------------
        print("\n🎤 STEP 2: Starting Speech-to-Text (Whisper)...")
        segments, info = whisper_model.transcribe(temp_audio.name, language="ar")
        text_input = " ".join([segment.text for segment in segments]).strip()

        print(f"   ⏱️ Audio duration: {getattr(info, 'duration', 'unknown')}")
        print(f"   🗣️ Transcription: '{text_input}'")

        # remove temp recording early (we don't need to keep it)
        try:
            os.unlink(temp_audio.name)
        except Exception:
            pass

        if not text_input:
            print("❌ No speech detected in audio")
            return JSONResponse({"error": "No speech detected"}, status_code=400)

        # ---------------------- STEP 3: LLM (Gemini) ----------------------
        print("\n🤖 STEP 3: Sending to LLM (Gemini)...")
        response_text = await get_gemini_response(text_input)
        print(f"✅ LLM response: '{response_text}'")

        # ---------------------- STEP 4: Generate TTS and stream to existing pc ----------------------
        # Find an active pc with attached session (the persistent one)
        pc_with_session = None
        for pc in pcs:
            if hasattr(pc, "session"):
                pc_with_session = pc
                break

        if pc_with_session is None:
            # No active peer connection — still return the text so client can handle it.
            print("⚠️ No active PeerConnection found to play TTS.")
            return JSONResponse({
                "transcription": text_input,
                "llm_response": response_text
            })

        # Generate TTS to a temp file (uses your existing TTS model)
        try:
            print("🔊 Generating TTS audio...")
            ref_wav = "/app/ref.wav"
            tts_out = os.path.join(tempfile.gettempdir(), f"tts_reply_{int(asyncio.get_event_loop().time()*1000)}.wav")
            tts.tts_to_file(
                text=response_text,
                file_path=tts_out,
                speaker_wav=ref_wav,
                language="ar"
            )
            file_size = os.path.getsize(tts_out)
            print(f"✅ TTS generated: {tts_out} ({file_size} bytes)")
        except Exception as e:
            print("❌ TTS generation failed:", e)
            traceback.print_exc()
            return JSONResponse({
                "transcription": text_input,
                "llm_response": response_text,
                "warning": "TTS generation failed"
            }, status_code=500)

        # Create MediaPlayer for the generated file and add its audio track to the existing pc.
        try:
            print("▶️ Streaming TTS to peer connection...")
            player = MediaPlayer(tts_out)
            if not player.audio:
                raise Exception("MediaPlayer has no audio track")

            # Add track to the existing PC to stream to client
            pc_with_session.addTrack(player.audio)

            # Give some time for the track to be negotiated & start playing.
            # This is intentionally short — you can increase if you see cutoff.
            await asyncio.sleep(0.5)

            # NOTE: keep a reference to player until play finishes; we rely on the player object
            # staying in scope for the duration of playback. Sleep based on file size to avoid early cleanup.
            try:
                duration_est = max(0.6, (file_size / (16000 * 2)) * 0.95)
            except Exception:
                duration_est = 1.2
            await asyncio.sleep(duration_est + 0.25)

            # Stop and cleanup the player
            try:
                await player.stop()
            except Exception:
                pass

            # remove the temp tts file
            try:
                os.unlink(tts_out)
            except Exception:
                pass

            print("✅ TTS streamed and cleaned up.")
        except Exception as e:
            print("❌ Error streaming TTS to pc:", e)
            traceback.print_exc()

        # ---------------------- STEP 5: Return JSON (transcription + text response) ----------------------
        return JSONResponse({
            "transcription": text_input,
            "llm_response": response_text
        })

    except Exception as e:
        print("❌ Error in voice_conversation handler:", e)
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# =====================================================
# 🔊 OFFER HANDLER (TTS STREAM)
# =====================================================
@app.post("/offer")
async def offer(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "مرحباً")
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

        config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
        pc = RTCPeerConnection(configuration=config)
        pcs.add(pc)

        ref_wav = "/app/ref.wav"
        temp_audio = os.path.join(tempfile.gettempdir(), f"tts_{id(pc)}.wav")
        tts.tts_to_file(text=text, file_path=temp_audio, speaker_wav=ref_wav, language="ar")

        player = MediaPlayer(temp_audio)
        pc.addTrack(player.audio)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        await asyncio.sleep(1)
        return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

@app.on_event("shutdown")
async def on_shutdown():
    for pc in pcs:
        await pc.close()
    pcs.clear()
    print("🛑 All connections closed.")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Running Arabic Voice AI Call on http://0.0.0.0:5002/ui")
    uvicorn.run(app, host="0.0.0.0", port=5002)

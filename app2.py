import asyncio
import tempfile
import os
import torch
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer
from av import open as av_open
from TTS.api import TTS
from faster_whisper import WhisperModel
import google.generativeai as genai
import traceback

app = FastAPI()
pcs = set()

# 🧠 Load models at startup
print("🔊 Loading Arabic TTS model...")
tts = TTS(
    model_path="/root/.local/share/tts/tts_models--ar--custom--egtts_v0.1",
    config_path="/root/.local/share/tts/tts_models--ar--custom--egtts_v0.1/config.json"
)
print("✅ TTS model loaded.")

# 🎤 Load Whisper STT model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel("medium", device=device)
print(f"✅ Loaded faster-whisper model on {device}")

# 🤖 Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCVMkDlunj2qvsmP8gf3ExrqoqXmY3aYa0")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
print("✅ Gemini LLM configured")


async def get_gemini_response(text: str) -> str:
    """
    Generate an Egyptian Arabic-style response using Gemini.
    """
    try:
        instruction = (
            "تخيل إنك موظف خدمة عملاء مصري ودود بيتكلم باللهجة المصرية الطبيعية، "
            "بترد باحترام وبلُطف على العميل، ومبتستخدمش فصحى رسمية. "
            "خلي إجابتك بسيطة، قريبة من كلام الناس العادي، وماتطولش."
        )

        prompt = f"{instruction}\n\nالعميل قال: {text}\n\nرد الموظف:"

        response = await asyncio.to_thread(
            gemini_model.generate_content,
            prompt
        )

        return response.text.strip()

    except Exception as e:
        print("Gemini Error:", e)
        traceback.print_exc()
        return "عذرًا يا فندم، حصل خطأ بسيط في النظام. ممكن تعيد سؤالك؟"


@app.get("/ui")
async def index():
    return HTMLResponse("""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial; padding: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h3 { color: #333; }
            .mode-selector { margin: 20px 0; padding: 15px; background: #f0f0f0; border-radius: 5px; }
            .mode-selector label { margin-right: 20px; font-size: 16px; }
            textarea { font-size: 16px; padding: 10px; width: 100%; box-sizing: border-box; border: 2px solid #ddd; border-radius: 5px; }
            button { padding: 12px 24px; font-size: 16px; cursor: pointer; margin: 5px; border: none; border-radius: 5px; background: #4CAF50; color: white; }
            button:hover { background: #45a049; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            #log { background: #f0f0f0; padding: 15px; max-height: 400px; overflow-y: auto; border-radius: 5px; font-family: monospace; font-size: 12px; }
            .recording { background: #f44336 !important; }
            #audioContainer { margin: 20px 0; }
            .hidden { display: none; }
        </style>
    </head>
    <body>
      <div class="container">
        <h3>🎙️ Arabic Voice AI Assistant</h3>
        
        <div class="mode-selector">
            <strong>Mode:</strong>
            <label><input type="radio" name="mode" value="text" checked onchange="switchMode()"> 📝 Text Mode</label>
            <label><input type="radio" name="mode" value="voice" onchange="switchMode()"> 🎤 Voice Mode</label>
        </div>

        <!-- Text Mode -->
        <div id="textMode">
            <textarea id="text" placeholder="اكتب النص هنا..." rows="4">مرحباً، كيف حالك؟</textarea><br><br>
            <button onclick="startText()">🔊 Generate & Play</button>
        </div>

        <!-- Voice Mode -->
        <div id="voiceMode" class="hidden">
            <p>Click the button and speak in Arabic. The AI will listen, process your speech, and respond with voice.</p>
            <button id="recordBtn" onclick="toggleRecording()">🎤 Start Recording</button>
            <button onclick="startVoiceConversation()" id="sendVoiceBtn" disabled>📤 Send to AI</button>
            <audio id="recordingPlayback" controls style="display:none; width: 100%; margin-top: 10px;"></audio>
        </div>

        <button onclick="clearLog()">Clear Log</button>
        
        <div id="audioContainer"></div>
        <h4>Logs:</h4>
        <pre id="log"></pre>
      </div>

      <script>
        let mediaRecorder;
        let audioChunks = [];
        let recordedBlob = null;

        function log(msg) {
          const pre = document.getElementById("log");
          const timestamp = new Date().toLocaleTimeString();
          pre.textContent += `[${timestamp}] ${msg}\\n`;
          pre.scrollTop = pre.scrollHeight;
          console.log(msg);
        }

        function clearLog() {
          document.getElementById("log").textContent = "";
        }

        function switchMode() {
          const mode = document.querySelector('input[name="mode"]:checked').value;
          const textMode = document.getElementById('textMode');
          const voiceMode = document.getElementById('voiceMode');
          
          if (mode === 'text') {
            textMode.classList.remove('hidden');
            voiceMode.classList.add('hidden');
          } else {
            textMode.classList.add('hidden');
            voiceMode.classList.remove('hidden');
          }
        }

        async function toggleRecording() {
            const btn = document.getElementById('recordBtn');
            const sendBtn = document.getElementById('sendVoiceBtn');
            const playback = document.getElementById('recordingPlayback');

            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                // Start recording
                try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioChunks = [];
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    recordedBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const url = URL.createObjectURL(recordedBlob);
                    playback.src = url;
                    playback.style.display = 'block';
                    sendBtn.disabled = false;
                    log("✅ Recording saved.");

                    // 🚀 Automatically send to AI after stop
                    log("🤖 Automatically sending to AI...");
                    await startVoiceConversation();
                };
                
                mediaRecorder.start();
                btn.textContent = '⏹️ Stop Recording';
                btn.classList.add('recording');
                sendBtn.disabled = true;
                playback.style.display = 'none';
                log("🎤 Recording started... Speak now!");
                
                } catch (err) {
                log("❌ Microphone error: " + err.message);
                }
            } else {
                // Stop recording
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                btn.textContent = '🎤 Start Recording';
                btn.classList.remove('recording');
                log("⏸️ Recording stopped.");
            }
        }


        async function startVoiceConversation() {
          if (!recordedBlob) {
            log("❌ No recording found!");
            return;
          }

          log("🚀 Starting voice conversation...");
          
          // Create FormData with audio
          const formData = new FormData();
          formData.append('audio', recordedBlob, 'recording.wav');

          // Send to server
          log("📤 Uploading audio to server...");
          const response = await fetch('/voice-conversation', {
            method: 'POST',
            body: formData
          });

          if (!response.ok) {
            const error = await response.text();
            log("❌ Server error: " + error);
            return;
          }

          const data = await response.json();
          log("🗣️ You said: " + data.transcription);
          log("🤖 AI responded: " + data.llm_response);
          log("📥 Preparing to play AI response...");

          // Now play the TTS response via WebRTC
          await playWebRTC(data.llm_response);
        }

        async function startText() {
          const text = document.getElementById('text').value;
          if (!text.trim()) {
            log("❌ Please enter some text!");
            return;
          }
          await playWebRTC(text);
        }

        async function playWebRTC(text) {
          try {
            log("🚀 Starting WebRTC connection...");
            
            const pc = new RTCPeerConnection({
              iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
              ]
            });

            pc.addTransceiver('audio', { direction: 'recvonly' });

            pc.ontrack = (event) => {
              log("🎧 Audio track received!");
              const audio = document.createElement('audio');
              audio.srcObject = event.streams[0];
              audio.autoplay = true;
              audio.controls = true;
              
              const container = document.getElementById('audioContainer');
              container.innerHTML = '<p><strong>AI Response Audio:</strong></p>';
              container.appendChild(audio);
              
              audio.onplay = () => log("▶️ AI is speaking!");
              audio.onended = () => log("✅ AI finished speaking");
            };

            pc.oniceconnectionstatechange = () => log("🧊 ICE: " + pc.iceConnectionState);
            pc.onconnectionstatechange = () => log("🔗 Connection: " + pc.connectionState);

            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            log("⏳ Gathering ICE candidates...");
            await new Promise(resolve => {
              if (pc.iceGatheringState === 'complete') resolve();
              else pc.onicegatheringstatechange = () => {
                if (pc.iceGatheringState === 'complete') resolve();
              };
            });

            log("📤 Sending to server...");
            const response = await fetch('/offer', {
              method: 'POST',
              body: JSON.stringify({
                sdp: pc.localDescription.sdp,
                type: pc.localDescription.type,
                text: text
              }),
              headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
              const errorText = await response.text();
              log("❌ Server error: " + errorText);
              return;
            }

            const answer = await response.json();
            log("📥 Received answer from server");
            await pc.setRemoteDescription(new RTCSessionDescription(answer));
            log("✅ Connection established!");
            
          } catch (error) {
            log("❌ Error: " + error.message);
            console.error(error);
          }
        }
      </script>
    </body>
    </html>
    """)


@app.post("/voice-conversation")
async def voice_conversation(audio: UploadFile = File(...)):
    """Handle voice input: STT → LLM → return text for TTS"""
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
        
        print(f"✅ Audio received and saved")
        print(f"   📍 Location: {temp_audio.name}")
        print(f"   📊 Size: {len(content)} bytes ({len(content)/1024:.2f} KB)")

        # ---------------------- STEP 2: STT (Whisper) ----------------------
        print("\n🎤 STEP 2: Starting Speech-to-Text (Whisper)...")
        print(f"   🔧 Model: faster-whisper (Large)")
        print(f"   🌍 Language: Arabic")
        print(f"   💻 Device: {device}")
        
        segments, info = whisper_model.transcribe(temp_audio.name, language="ar")
        
        print(f"   ⏱️ Audio duration: {info.duration:.2f} seconds")
        print(f"   🔍 Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        text_input = " ".join([segment.text for segment in segments])
        
        print(f"✅ Transcription complete")
        print(f"   🗣️ User said: '{text_input}'")
        print(f"   📏 Text length: {len(text_input)} characters")

        if not text_input.strip():
            print("❌ ERROR: No speech detected in audio")
            return JSONResponse(
                {"error": "No speech detected"},
                status_code=400
            )

        # ---------------------- STEP 3: LLM (Gemini) ----------------------
        print("\n🤖 STEP 3: Sending to LLM (Gemini)...")
        print(f"   📤 Input: '{text_input}'")
        
        response_text = await get_gemini_response(text_input)
        
        print(f"✅ LLM response received")
        print(f"   💬 AI response: '{response_text}'")
        print(f"   📏 Response length: {len(response_text)} characters")
        
        # Clean up temp file
        print(f"\n🗑️ Cleaning up temporary audio file: {temp_audio.name}")
        os.unlink(temp_audio.name)
        print("✅ Cleanup complete")

        print("\n" + "="*60)
        print("✅ VOICE CONVERSATION PIPELINE COMPLETE")
        print("   Next: TTS will generate speech from AI response")
        print("="*60 + "\n")
        
        return JSONResponse({
            "transcription": text_input,
            "llm_response": response_text
        })

    except Exception as e:
        print(f"❌ Error in voice conversation: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.post("/offer")
async def offer(request: Request):
    """Handle WebRTC offer and stream TTS audio"""
    try:
        data = await request.json()
        text = data.get("text", "مرحبا بكم")
        
        print(f"\n{'='*60}")
        print(f"🛰️ WEBRTC + TTS PIPELINE STARTED")
        print(f"{'='*60}")
        print(f"📝 Text to synthesize: '{text}'")
        print(f"📏 Text length: {len(text)} characters")

        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

        # ---------------------- STEP 1: Create WebRTC Connection ----------------------
        print(f"\n🔌 STEP 1: Creating WebRTC connection...")
        
        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
            ]
        )

        pc = RTCPeerConnection(configuration=config)
        pcs.add(pc)
        
        print(f"✅ RTCPeerConnection created")
        print(f"   🆔 Connection ID: {id(pc)}")
        print(f"   🌐 ICE servers: 2 STUN servers configured")

        # ---------------------- STEP 2: Generate TTS Audio ----------------------
        print(f"\n🗣️ STEP 2: Generating speech with TTS...")
        
        ref_wav = "/app/ref.wav"
        temp_audio = os.path.join(tempfile.gettempdir(), f"output_{id(pc)}.wav")
        
        print(f"   🔧 TTS Model: Arabic custom model")
        print(f"   🎤 Reference voice: {ref_wav}")
        print(f"   💾 Output file: {temp_audio}")

        if not os.path.exists(ref_wav):
            print(f"❌ ERROR: Reference file not found: {ref_wav}")
            return JSONResponse(
                {"error": f"Reference file not found: {ref_wav}"}, 
                status_code=500
            )

        print(f"   ⚙️ Starting TTS synthesis...")
        try:
            tts.tts_to_file(
                text=text,
                file_path=temp_audio,
                speaker_wav=ref_wav,
                language="ar"
            )
            
            file_size = os.path.getsize(temp_audio)
            
            print(f"✅ TTS synthesis complete")
            print(f"   📊 Generated audio file size: {file_size} bytes ({file_size/1024:.2f} KB)")
            
            if file_size == 0:
                raise Exception("Generated audio file is empty")
            
            # Get audio duration estimate
            duration_estimate = file_size / (16000 * 2)  # Assuming 16kHz, 16-bit
            print(f"   ⏱️ Estimated duration: {duration_estimate:.2f} seconds")
                
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
            print(traceback.format_exc())
            return JSONResponse({"error": f"TTS failed: {str(e)}"}, status_code=500)

        # ---------------------- STEP 3: Setup Media Player ----------------------
        print(f"\n🎵 STEP 3: Setting up audio stream...")
        
        try:
            print(f"   🔧 Creating MediaPlayer for: {temp_audio}")
            player = MediaPlayer(temp_audio)
            
            if not player.audio:
                raise Exception("MediaPlayer has no audio track")
            
            print(f"✅ MediaPlayer created successfully")
            
            audio_track = pc.addTrack(player.audio)
            
            print(f"✅ Audio track added to WebRTC connection")
            print(f"   🎼 Track type: {audio_track.kind}")
            
        except Exception as e:
            print(f"❌ MediaPlayer error: {e}")
            print(traceback.format_exc())
            return JSONResponse({"error": f"MediaPlayer failed: {str(e)}"}, status_code=500)

        # ---------------------- STEP 4: WebRTC Negotiation ----------------------
        print(f"\n🤝 STEP 4: WebRTC negotiation...")
        
        @pc.on("iceconnectionstatechange")
        async def on_ice_state():
            print(f"   🧊 ICE connection state: {pc.iceConnectionState}")

        @pc.on("connectionstatechange")
        async def on_connection_state():
            print(f"   🔗 Connection state: {pc.connectionState}")
            if pc.connectionState == "failed":
                print("   💥 Connection failed, closing peer connection")
                await pc.close()
                pcs.discard(pc)
            elif pc.connectionState == "connected":
                print("   ✅ WebRTC connection established successfully!")

        print(f"   📥 Setting remote description...")
        await pc.setRemoteDescription(offer)
        print(f"   ✅ Remote description set")

        print(f"   📝 Creating answer...")
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        print(f"   ✅ Answer created and set as local description")

        # ---------------------- STEP 5: ICE Gathering ----------------------
        print(f"\n🧊 STEP 5: Gathering ICE candidates...")
        print(f"   ⏳ Waiting for ICE gathering to complete...")
        
        max_wait = 50
        wait_count = 0
        while pc.iceGatheringState != "complete" and wait_count < max_wait:
            await asyncio.sleep(0.1)
            wait_count += 1
        
        if pc.iceGatheringState == "complete":
            print(f"   ✅ ICE gathering complete")
        else:
            print(f"   ⚠️ ICE gathering timeout (state: {pc.iceGatheringState})")

        # ---------------------- Complete ----------------------
        print(f"\n📤 Sending SDP answer to client...")
        print(f"{'='*60}")
        print(f"✅ WEBRTC + TTS PIPELINE COMPLETE")
        print(f"   Audio will now stream to browser")
        print(f"{'='*60}\n")

        return JSONResponse({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR in /offer:")
        print(traceback.format_exc())
        return JSONResponse(
            {"error": f"Server error: {str(e)}"}, 
            status_code=500
        )


@app.on_event("shutdown")
async def on_shutdown():
    print("🛑 Shutting down...")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    print("✅ All connections closed")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Arabic Voice AI Assistant")
    print("📍 Server: http://0.0.0.0:5002/ui")
    print("="*60)
    uvicorn.run(app, host="11.11.11.1", port=5002)

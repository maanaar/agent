import asyncio
import tempfile
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from av import open as av_open
from TTS.api import TTS
import traceback

app = FastAPI()
pcs = set()

# 🧠 Load Arabic TTS model once at startup
print("🔊 Loading Arabic TTS model...")
tts = TTS(
    model_path="/root/.local/share/tts/tts_models--ar--custom--egtts_v0.1",
    config_path="/root/.local/share/tts/tts_models--ar--custom--egtts_v0.1/config.json"
)
print("✅ TTS model loaded.")


@app.get("/ui")
async def index():
    return HTMLResponse("""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial; padding: 20px; }
            #log { background: #f0f0f0; padding: 10px; max-height: 400px; overflow-y: auto; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
            textarea { font-size: 16px; padding: 10px; }
        </style>
    </head>
    <body>
      <h3>🎙️ WebRTC Arabic TTS</h3>
      <textarea id="text" placeholder="اكتب النص هنا..." rows="4" cols="50">مرحباً، هذا اختبار.</textarea><br><br>
      <button onclick="start()">🔊 Generate & Play</button>
      <button onclick="clearLog()">Clear Log</button>
      <br><br>
      <div id="audioContainer"></div>
      <h4>Logs:</h4>
      <pre id="log"></pre>

      <script>
        function log(msg) {
          const pre = document.getElementById("log");
          const timestamp = new Date().toLocaleTimeString();
          pre.textContent += `[${timestamp}] ${msg}\n`;
          console.log(msg);
        }

        function clearLog() {
          document.getElementById("log").textContent = "";
        }

        async function start() {
          try {
            log("🚀 Starting WebRTC connection...");
            const text = document.getElementById('text').value;
            
            if (!text.trim()) {
              log("❌ Please enter some text!");
              return;
            }

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
              container.innerHTML = '';
              container.appendChild(audio);
              
              audio.onloadedmetadata = () => {
                log("✅ Audio metadata loaded");
              };
              audio.onplay = () => {
                log("▶️ Audio is playing!");
              };
              audio.onended = () => {
                log("✅ Audio finished playing");
              };
              audio.onerror = (e) => {
                log("❌ Audio error: " + JSON.stringify(e));
              };
            };

            pc.oniceconnectionstatechange = () => {
              log("🧊 ICE state: " + pc.iceConnectionState);
            };

            pc.onconnectionstatechange = () => {
              log("🔗 Connection state: " + pc.connectionState);
            };

            pc.onicegatheringstatechange = () => {
              log("📊 ICE gathering: " + pc.iceGatheringState);
            };

            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            // Wait for ICE gathering
            log("⏳ Waiting for ICE candidates...");
            await new Promise(resolve => {
              if (pc.iceGatheringState === 'complete') {
                resolve();
              } else {
                pc.onicegatheringstatechange = () => {
                  if (pc.iceGatheringState === 'complete') {
                    resolve();
                  }
                };
              }
            });

            log("📤 Sending offer to server...");
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


@app.post("/offer")
async def offer(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "مرحبا بكم")
        print(f"\n{'='*60}")
        print(f"🛰️ Received WebRTC offer")
        print(f"📝 Text to speak: {text}")
        print(f"{'='*60}")

        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

        # ICE configuration
        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
            ]
        )

        pc = RTCPeerConnection(configuration=config)
        pcs.add(pc)
        print("✅ RTCPeerConnection created")

        # 🗣️ Generate TTS
        ref_wav = "/app/ref.wav"
        temp_audio = os.path.join(tempfile.gettempdir(), f"output_{id(pc)}.wav")

        # Check reference file
        if not os.path.exists(ref_wav):
            print(f"❌ Reference file not found: {ref_wav}")
            return JSONResponse(
                {"error": f"Reference file not found: {ref_wav}"}, 
                status_code=500
            )

        print(f"🎤 Generating TTS...")
        try:
            tts.tts_to_file(
                text=text,
                file_path=temp_audio,
                speaker_wav=ref_wav,
                language="ar"
            )
            print(f"✅ Audio generated: {temp_audio}")
            
            # Verify file exists and has content
            if not os.path.exists(temp_audio):
                raise Exception("Generated audio file not found")
            
            file_size = os.path.getsize(temp_audio)
            print(f"📊 Audio file size: {file_size} bytes")
            
            if file_size == 0:
                raise Exception("Generated audio file is empty")
                
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
            print(traceback.format_exc())
            return JSONResponse({"error": f"TTS failed: {str(e)}"}, status_code=500)

        # 🎧 Create MediaPlayer and add track
        try:
            print(f"🎵 Creating MediaPlayer for: {temp_audio}")
            
            # Verify audio file is valid
            try:
                container = av_open(temp_audio)
                audio_stream = next(s for s in container.streams if s.type == 'audio')
                print(f"📊 Audio info: {audio_stream.codec_long_name}, "
                      f"{audio_stream.sample_rate}Hz, {audio_stream.channels} channels")
                container.close()
            except Exception as e:
                print(f"⚠️ Could not read audio info: {e}")
            
            # Create player - no options needed for WAV files
            player = MediaPlayer(temp_audio)
            
            if not player.audio:
                raise Exception("MediaPlayer has no audio track")
            
            print("✅ MediaPlayer created")
            
            # Add audio track to peer connection
            audio_track = pc.addTrack(player.audio)
            print(f"✅ Audio track added: {audio_track.kind}")
            
        except Exception as e:
            print(f"❌ MediaPlayer error: {e}")
            print(traceback.format_exc())
            return JSONResponse({"error": f"MediaPlayer failed: {str(e)}"}, status_code=500)

        # Event handlers
        @pc.on("iceconnectionstatechange")
        async def on_ice_state():
            print(f"🧊 ICE state: {pc.iceConnectionState}")

        @pc.on("connectionstatechange")
        async def on_connection_state():
            print(f"🔗 Connection state: {pc.connectionState}")
            if pc.connectionState == "failed":
                print("💥 Connection failed")
                await pc.close()
                pcs.discard(pc)
            elif pc.connectionState == "connected":
                print("✅ Connected successfully!")

        # Set remote description
        await pc.setRemoteDescription(offer)
        print("📥 Remote description set")

        # Create and set answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        print("📤 Answer created")

        # Wait for ICE gathering
        max_wait = 50  # 5 seconds max
        wait_count = 0
        while pc.iceGatheringState != "complete" and wait_count < max_wait:
            await asyncio.sleep(0.1)
            wait_count += 1
        
        if pc.iceGatheringState == "complete":
            print("✅ ICE gathering complete")
        else:
            print(f"⚠️ ICE gathering timeout (state: {pc.iceGatheringState})")

        print(f"📤 Sending answer to client")
        print(f"{'='*60}\n")

        return JSONResponse({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR in /offer endpoint:")
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
    print("🚀 Starting server on http://0.0.0.0:5002/ui")
    uvicorn.run(app, host="0.0.0.0", port=5002)

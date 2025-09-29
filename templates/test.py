async def process_audio(track, pc):
    """
    Receives user audio -> detects silence -> sends to Whisper -> Gemini -> prints response
    """
    print("🎙 Listening...")

    frames = []
    silence_threshold = 500  # you can tune this
    while True:
        frame = await track.recv()
        pcm = frame.to_ndarray()

        # Compute volume (RMS)
        volume = np.sqrt(np.mean(pcm**2))

        frames.append(pcm)

        # If silence detected & we have enough frames = process chunk
        if volume < silence_threshold and len(frames) > 20:
            print("⏸ Detected pause, processing speech chunk...")

            # Save audio chunk as WAV
            audio_data = np.concatenate(frames, axis=0)
            frames = []  # reset for next utterance
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with wave.open(temp_wav.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(48000)
                wf.writeframes(audio_data.tobytes())
            temp_wav.flush()

            # Transcribe with Whisper
            import openai
            with open(temp_wav.name, "rb") as f:
                result = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            text_input = result.text
            print("📝 Transcript:", text_input)

            # Send transcript to Gemini
            response_text = await get_gemini_response(text_input)
            print("🤖 Gemini Response:", response_text)

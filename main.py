import whisper
import pyperclip
import pyaudio
import wave

# Initialize the Whisper model
model = whisper.load_model("base")

# Variables for recording
audio_buffer = []
sample_rate = 16000
device_index = 1

# Function to record audio continuously
def record_audio(filename):
  CHUNK = 512
  FORMAT = pyaudio.paInt16
  CHANNELS = 1
  RATE = 44100

  p = pyaudio.PyAudio()

  stream = p.open(format=FORMAT,
          channels=CHANNELS,
          rate=RATE,
          input=True,
          input_device_index=1,
          frames_per_buffer=CHUNK)

  frames = []

  print("Recording audio...")
  while True:
    try:
      data = stream.read(CHUNK)
      frames.append(data)
    except KeyboardInterrupt:
      break

  print("Finished recording.")

  stream.stop_stream()
  stream.close()
  p.terminate()

  wf = wave.open(filename, 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(RATE)
  wf.writeframes(b''.join(frames))
  wf.close()


# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    # Use Whisper to transcribe the audio
    result = model.transcribe(audio_file)
    transcription = result['text']
    print("Transcription:", transcription)

    # Copy transcription to clipboard
    pyperclip.copy(transcription)
    print("Transcription copied to clipboard!")


AUDIO_FILE = "/Users/rohan/0_Inbox/test_llm_engineer/workspaces/voicenotes/temp_audio.wav"
# Start recording
record_audio(AUDIO_FILE)

# After CTRL-C is pressed, transcribe the recorded audio
transcribe_audio(AUDIO_FILE)

print("Exiting program.")

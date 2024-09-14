import whisper
import pyperclip
import pyaudio
import wave
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from typing import Type

# Initialize the Whisper model
model = whisper.load_model("base")

# Variables for recording
audio_buffer = []
sample_rate = 16000
device_index = 1

# Function to record audio continuously
def record_audio(filename: str) -> None:
    """
    Record audio from the microphone and save it to a file.
    
    Parameters:
    filename (str): The name of the file where the audio will be saved.
    """
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
def transcribe_audio(audio_file: str) -> None:
    """
    Transcribe audio using Whisper and copy the transcription to clipboard.
    
    Parameters:
    audio_file (str): The path to the audio file to be transcribed.
    """
    # Use Whisper to transcribe the audio
    result = model.transcribe(audio_file)
    transcription = result['text']
    print("Transcription:", transcription)

    # Copy transcription to clipboard
    pyperclip.copy(transcription)
    print("Transcription copied to clipboard!")

def transcribe_and_cleanup_chunks(chunks_folder: str, model: Type[whisper]) -> str:
    """
    Transcribe audio chunks from a directory and remove the chunk files.
    
    Parameters:
    chunks_folder (str): The directory containing audio chunks.
    model (whisper): The Whisper model to use for transcription.
    
    Returns:
    str: The combined transcription from all audio chunks.
    """
    transcription = ""
    for chunk_file in sorted(os.listdir(chunks_folder)):
        chunk_path = os.path.join(chunks_folder, chunk_file)
        if chunk_file.endswith('.wav'):
            result = model.transcribe(chunk_path)
            transcription += result['text'] + " "
            os.remove(chunk_path)  # Remove chunk file after transcription
    return transcription.strip()

def transcribe_audio_with_silence_handling(audio_file: str) -> None:
    """
    Transcribe audio while handling silence by splitting the audio into chunks.
    
    Parameters:
    audio_file (str): The path to the audio file to be processed.
    """
    # Load the audio file
    audio = AudioSegment.from_wav(audio_file)
    # Split audio by silence
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
    
    # Create a temporary directory for chunks
    chunks_folder = "/tmp/audio_chunks"
    if not os.path.exists(chunks_folder):
        os.makedirs(chunks_folder)
    
    # Export chunks as temporary WAV files
    for i, chunk in enumerate(chunks):
        chunk.export(os.path.join(chunks_folder, f"chunk_{i}.wav"), format="wav")
    
    # Transcribe chunks and clean up
    transcription = transcribe_and_cleanup_chunks(chunks_folder, model)
    print("Transcription:", transcription)
    pyperclip.copy(transcription)
    print("Transcription copied to clipboard!")


# Check for audio file argument
import sys
if len(sys.argv) > 1:
    transcribe_audio(sys.argv[1])
    #transcribe_audio_with_silence_handling(sys.argv[1])
else:
    # No audio file argument, proceed with recording
    AUDIO_FILE = "/tmp/temp_audio.wav"
    record_audio(AUDIO_FILE)
    transcribe_audio(AUDIO_FILE)
    #transcribe_audio_with_silence_handling(AUDIO_FILE)

print("Exiting program.")

"""
Module B: Speech-to-Text Transcription
=======================================
This module provides speech-to-text transcription with multiple backends:

Backends:
- groq: Groq Whisper API (fast, recommended for Singlish)
- gemini: Gemini Live API (experimental)

Features:
- File-based transcription
- Live microphone transcription
- Backend switching
"""

import os
import io
import wave
import asyncio
from typing import Optional, Literal

import pyaudio
from groq import Groq
from google.genai import types

from src.config import get_client as get_gemini_client, load_env_file


# ==============================================================================
# Configuration
# ==============================================================================

# Backend options
Backend = Literal["groq", "gemini"]
DEFAULT_BACKEND: Backend = "groq"

# Groq Whisper models
GROQ_MODEL_TURBO = "whisper-large-v3-turbo"  # Faster
GROQ_MODEL_LARGE = "whisper-large-v3"  # More accurate

# Gemini Live model
GEMINI_LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


# ==============================================================================
# Groq Client
# ==============================================================================

def get_groq_client() -> Groq:
    """Create and return a Groq client."""
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        raise TranscriptionError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Get your free API key at: https://console.groq.com/keys\n"
            "Then add to your .env file:\n"
            "  GROQ_API_KEY=your-api-key-here"
        )
    
    return Groq(api_key=api_key)


# ==============================================================================
# Audio Recording Helpers
# ==============================================================================

def record_chunk(
    duration_seconds: float = 5.0,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Record a chunk of audio from the microphone."""
    pya = pyaudio.PyAudio()
    
    try:
        mic_info = pya.get_default_input_device_info()
        
        stream = pya.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        
        frames = []
        total_chunks = int(sample_rate / CHUNK_SIZE * duration_seconds)
        
        for _ in range(total_chunks):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
    finally:
        pya.terminate()
    
    # Convert to WAV format in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    
    wav_buffer.seek(0)
    return wav_buffer.read()


# ==============================================================================
# Groq Whisper Transcription
# ==============================================================================

def transcribe_file(
    file_path: str,
    model: str = GROQ_MODEL_TURBO,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """Transcribe an audio file using Groq Whisper API."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    client = get_groq_client()
    
    print(f"🎵 Transcribing: {file_path}")
    print(f"   Model: {model}")
    
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            response_format="text",
            temperature=0.0,
        )
    
    return transcription


def transcribe_audio_bytes(
    audio_data: bytes,
    model: str = GROQ_MODEL_TURBO,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """Transcribe audio bytes using Groq Whisper API."""
    client = get_groq_client()
    
    audio_file = io.BytesIO(audio_data)
    audio_file.name = "audio.wav"
    
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model=model,
        language=language,
        prompt=prompt,
        response_format="text",
        temperature=0.0,
    )
    
    return transcription


def run_groq_transcription(
    chunk_duration: float = 5.0,
    model: str = GROQ_MODEL_TURBO,
    prompt: str = "Transcribe Singlish/Singaporean English accurately.",
) -> None:
    """Run live transcription using Groq Whisper."""
    print("\n" + "=" * 60)
    print("📝 Live Transcription (Groq Whisper)")
    print("=" * 60)
    print(f"   Model: {model}")
    print(f"   Chunk duration: {chunk_duration}s")
    print(f"   Prompt: {prompt}")
    print("\n   Speak into your microphone...")
    print("   Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    pya = pyaudio.PyAudio()
    mic_info = pya.get_default_input_device_info()
    print(f"🎤 Using: {mic_info['name']}\n")
    pya.terminate()
    
    # Status line length for proper clearing
    max_status_len = 50
    
    try:
        while True:
            # Show recording status
            status = f"🔴 Recording ({chunk_duration}s)..."
            print(status, end="", flush=True)
            
            audio_data = record_chunk(chunk_duration)
            
            # Update status to transcribing
            print(f"\r{'🔄 Transcribing...':<{max_status_len}}", end="", flush=True)
            
            try:
                text = transcribe_audio_bytes(audio_data, model=model, prompt=prompt)
                
                # Clear the status line completely and print result
                if text.strip():
                    print(f"\r{' ' * max_status_len}\r📝 {text.strip()}")
                else:
                    print(f"\r{' ' * max_status_len}\r   (silence)")
            except Exception as e:
                print(f"\r{' ' * max_status_len}\r❌ Error: {e}")
    
    except KeyboardInterrupt:
        print(f"\r{' ' * max_status_len}\r")  # Clear any remaining status
        print("🛑 Transcription stopped.")


# ==============================================================================
# Gemini Live Transcription
# ==============================================================================

async def run_gemini_transcription_async() -> None:
    """Run live transcription using Gemini Live API with buffered output."""
    pya = pyaudio.PyAudio()
    audio_queue = asyncio.Queue(maxsize=5)
    audio_stream = None
    
    # Text buffer for accumulating transcription
    text_buffer = []
    last_line_length = 0
    
    async def listen_audio():
        nonlocal audio_stream
        mic_info = pya.get_default_input_device_info()
        print(f"🎤 Using: {mic_info['name']}\n")
        
        audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        
        while True:
            data = await asyncio.to_thread(
                audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
            )
            await audio_queue.put({"data": data, "mime_type": "audio/pcm"})
    
    async def send_audio(session):
        while True:
            msg = await audio_queue.get()
            await session.send_realtime_input(audio=msg)
    
    async def receive_transcription(session):
        nonlocal text_buffer, last_line_length
        
        while True:
            turn = session.receive()
            async for response in turn:
                # Check for input transcription (what user said)
                if (
                    response.server_content
                    and response.server_content.input_transcription
                ):
                    transcript = response.server_content.input_transcription
                    if transcript.text:
                        # Add to buffer
                        text_buffer.append(transcript.text)
                        
                        # Build current line
                        current_text = "".join(text_buffer).strip()
                        
                        # Clear previous line and print updated text
                        clear_str = "\r" + " " * last_line_length + "\r"
                        display_str = f"📝 {current_text}"
                        print(f"{clear_str}{display_str}", end="", flush=True)
                        last_line_length = len(display_str)
                
                # Check for turn complete (sentence finished)
                if (
                    response.server_content
                    and response.server_content.turn_complete
                ):
                    if text_buffer:
                        # Print final sentence and move to new line
                        final_text = "".join(text_buffer).strip()
                        clear_str = "\r" + " " * last_line_length + "\r"
                        print(f"{clear_str}📝 {final_text}")
                        text_buffer = []
                        last_line_length = 0
    
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction="You are a silent assistant. Just listen quietly.",
    )
    
    client = get_gemini_client()
    
    try:
        async with client.aio.live.connect(
            model=GEMINI_LIVE_MODEL,
            config=config,
        ) as session:
            print("\n" + "=" * 60)
            print("📝 Live Transcription (Gemini Live API)")
            print("=" * 60)
            print(f"   Model: {GEMINI_LIVE_MODEL}")
            print("\n   Speak into your microphone...")
            print("   Transcription updates in real-time")
            print("   Press Ctrl+C to stop")
            print("=" * 60 + "\n")
            
            async with asyncio.TaskGroup() as tg:
                tg.create_task(listen_audio())
                tg.create_task(send_audio(session))
                tg.create_task(receive_transcription(session))
    
    except asyncio.CancelledError:
        pass
    finally:
        if audio_stream:
            audio_stream.close()
        pya.terminate()
        print("\n\n🛑 Transcription stopped.")


def run_gemini_transcription() -> None:
    """Run Gemini Live transcription (sync wrapper)."""
    try:
        asyncio.run(run_gemini_transcription_async())
    except KeyboardInterrupt:
        pass


# ==============================================================================
# Main Entry Point with Backend Selection
# ==============================================================================

def run_live_transcription(
    backend: Backend = DEFAULT_BACKEND,
    chunk_duration: float = 5.0,
) -> None:
    """
    Run live speech-to-text transcription.
    
    Args:
        backend: Which backend to use ("groq" or "gemini")
        chunk_duration: For Groq, duration of each recording chunk
    
    Usage:
        >>> run_live_transcription(backend="groq")
        >>> run_live_transcription(backend="gemini")
    """
    if backend == "groq":
        run_groq_transcription(chunk_duration=chunk_duration)
    elif backend == "gemini":
        run_gemini_transcription()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'groq' or 'gemini'.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    load_env_file()
    
    print("=" * 60)
    print("Speech-to-Text Transcription")
    print("=" * 60)
    
    # Parse backend from args
    backend = "groq"
    file_path = None
    
    for arg in sys.argv[1:]:
        if arg in ("--groq", "-g"):
            backend = "groq"
        elif arg in ("--gemini", "-m"):
            backend = "gemini"
        elif not arg.startswith("-"):
            file_path = arg
    
    if file_path:
        # File transcription (Groq only)
        try:
            result = transcribe_file(file_path)
            print("\n" + "=" * 60)
            print("📜 Transcription:")
            print("=" * 60)
            print(result)
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        # Live transcription
        try:
            run_live_transcription(backend=backend)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except TranscriptionError as e:
            print(f"❌ {e}")

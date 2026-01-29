"""
Module C: Audio Recording Utility
==================================
This module provides utility functions for recording audio locally.
Useful for testing the audio understanding module or capturing
audio samples for later analysis.

Features:
- Record raw WAV audio from microphone
- Configurable duration and sample rate
- Simple command-line interface
"""

import os
import wave
import struct
from typing import Optional

import pyaudio


# ==============================================================================
# Recording Configuration
# ==============================================================================

DEFAULT_CHANNELS = 1          # Mono
DEFAULT_SAMPLE_RATE = 16000   # 16kHz (matches Live API input)
DEFAULT_CHUNK_SIZE = 1024     # Frames per buffer
DEFAULT_FORMAT = pyaudio.paInt16  # 16-bit audio


# ==============================================================================
# Recording Functions
# ==============================================================================

def record_audio(
    output_file: str,
    duration_seconds: float = 5.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    show_progress: bool = True,
) -> str:
    """
    Record audio from the default microphone to a WAV file.
    
    Args:
        output_file: Path for the output WAV file.
        duration_seconds: How long to record (in seconds).
        sample_rate: Sample rate in Hz (default: 16000).
        channels: Number of audio channels (default: 1 for mono).
        show_progress: Whether to show a progress indicator.
    
    Returns:
        str: Path to the recorded file.
    
    Raises:
        RuntimeError: If no microphone is available.
    
    Example:
        >>> record_audio("my_recording.wav", duration_seconds=10)
        'Recording for 10.0 seconds...'
        'Saved to: my_recording.wav'
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize PyAudio
    pya = pyaudio.PyAudio()
    
    try:
        # Get default input device
        try:
            mic_info = pya.get_default_input_device_info()
        except IOError:
            raise RuntimeError(
                "No microphone found. Please connect a microphone."
            )
        
        print(f"Using: {mic_info['name']}")
        print(f"Recording for {duration_seconds:.1f} seconds...")
        
        # Open audio stream
        stream = pya.open(
            format=DEFAULT_FORMAT,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=DEFAULT_CHUNK_SIZE,
        )
        
        # Calculate number of chunks to record
        total_chunks = int(sample_rate / DEFAULT_CHUNK_SIZE * duration_seconds)
        frames = []
        
        # Record audio
        for i in range(total_chunks):
            data = stream.read(DEFAULT_CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
            
            # Show progress
            if show_progress and i % 10 == 0:
                progress = (i / total_chunks) * 100
                bar_len = 30
                filled = int(bar_len * i / total_chunks)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r   [{bar}] {progress:.0f}%", end="", flush=True)
        
        if show_progress:
            print(f"\r   [{'█' * 30}] 100%")
        
        # Close stream
        stream.stop_stream()
        stream.close()
        
    finally:
        pya.terminate()
    
    # Save as WAV file
    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pya.get_sample_size(DEFAULT_FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    
    # Get file size
    file_size = os.path.getsize(output_file)
    file_size_kb = file_size / 1024
    
    print(f"Saved to: {output_file}")
    print(f"   Size: {file_size_kb:.1f} KB")
    print(f"   Format: WAV ({sample_rate}Hz, {channels}ch, 16-bit)")
    
    return output_file


def record_with_countdown(
    output_file: str,
    duration_seconds: float = 5.0,
    countdown: int = 3,
    **kwargs,
) -> str:
    """
    Record audio with a countdown before starting.
    
    Args:
        output_file: Path for the output WAV file.
        duration_seconds: How long to record (in seconds).
        countdown: Seconds to count down before recording.
        **kwargs: Additional arguments passed to record_audio.
    
    Returns:
        str: Path to the recorded file.
    """
    import time
    
    print(f"\nRecording will start in {countdown} seconds...")
    
    for i in range(countdown, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("   🔴 GO!")
    return record_audio(output_file, duration_seconds, **kwargs)


def list_audio_devices() -> None:
    """
    List all available audio input devices.
    
    Useful for troubleshooting microphone issues.
    """
    pya = pyaudio.PyAudio()
    
    print("\nAvailable Audio Input Devices:")
    print("-" * 50)
    
    default_idx = None
    try:
        default_info = pya.get_default_input_device_info()
        default_idx = default_info["index"]
    except IOError:
        pass
    
    for i in range(pya.get_device_count()):
        info = pya.get_device_info_by_index(i)
        
        # Only show input devices
        if info["maxInputChannels"] > 0:
            is_default = " (DEFAULT)" if i == default_idx else ""
            print(
                f"  [{i}] {info['name']}{is_default}\n"
                f"      Channels: {info['maxInputChannels']}, "
                f"Rate: {info['defaultSampleRate']:.0f}Hz"
            )
    
    pya.terminate()
    print()


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Module C: Audio Recording Utility")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list" or sys.argv[1] == "-l":
            list_audio_devices()
        else:
            output_file = sys.argv[1]
            duration = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
            
            try:
                record_with_countdown(output_file, duration)
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("\nUsage:")
        print("  python -m src.audio_recorder <output.wav> [duration_seconds]")
        print("  python -m src.audio_recorder --list  # List audio devices")
        print("\nExamples:")
        print("  python -m src.audio_recorder recording.wav")
        print("  python -m src.audio_recorder test.wav 10")
        print("\nOr use programmatically:")
        print("  from src.audio_recorder import record_audio")
        print('  record_audio("output.wav", duration_seconds=5)')

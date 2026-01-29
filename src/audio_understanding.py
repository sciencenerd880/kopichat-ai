"""
Module A: Audio Understanding (File & Inline)
==============================================
This module provides functionality to analyze audio files using
the Google Gemini API. It automatically selects the best input method
based on file size:

- Small Files (<20MB): Inline bytes using types.Part.from_bytes
- Large Files (>=20MB): Files API upload using client.files.upload

Supported audio formats:
- MP3, WAV, AIFF, AAC, OGG, FLAC
"""

import os
import time
import mimetypes
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from src.config import (
    get_client,
    load_env_file,
    AUDIO_MODEL,
    INLINE_SIZE_THRESHOLD,
)


# ==============================================================================
# MIME Type Detection
# ==============================================================================

# Extended MIME type mapping for audio files
AUDIO_MIME_TYPES = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/m4a",
    ".opus": "audio/opus",
    ".weba": "audio/webm",
    ".webm": "audio/webm",
}


def get_audio_mime_type(file_path: str) -> str:
    """
    Detect the MIME type of an audio file.
    
    Args:
        file_path: Path to the audio file.
    
    Returns:
        str: The MIME type string (e.g., "audio/mp3").
    
    Raises:
        ValueError: If the file extension is not recognized.
    """
    ext = Path(file_path).suffix.lower()
    
    # Check our extended mapping first
    if ext in AUDIO_MIME_TYPES:
        return AUDIO_MIME_TYPES[ext]
    
    # Fall back to mimetypes library
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith("audio/"):
        return mime_type
    
    raise ValueError(
        f"Unsupported audio format: '{ext}'. "
        f"Supported formats: {', '.join(AUDIO_MIME_TYPES.keys())}"
    )


# ==============================================================================
# File Analysis Functions
# ==============================================================================

def analyze_audio_inline(
    client: genai.Client,
    file_path: str,
    prompt: str = "Describe this audio clip",
) -> str:
    """
    Analyze audio by passing data inline (for small files).
    
    This method is suitable for files smaller than 20MB.
    The audio data is embedded directly in the API request.
    
    Args:
        client: Authenticated Gemini client.
        file_path: Path to the audio file.
        prompt: The analysis prompt.
    
    Returns:
        str: The model's response text.
    """
    print(f"📤 Sending audio inline: {file_path}")
    
    mime_type = get_audio_mime_type(file_path)
    
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    
    response = client.models.generate_content(
        model=AUDIO_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type,
            ),
        ],
    )
    
    return response.text


def analyze_audio_upload(
    client: genai.Client,
    file_path: str,
    prompt: str = "Describe this audio clip",
    wait_for_processing: bool = True,
    max_wait_seconds: int = 120,
) -> str:
    """
    Analyze audio by uploading via Files API (for large files).
    
    This method uploads the file to Google's servers first,
    then references it in the API request. Required for files >= 20MB.
    
    Args:
        client: Authenticated Gemini client.
        file_path: Path to the audio file.
        prompt: The analysis prompt.
        wait_for_processing: Whether to wait for file processing.
        max_wait_seconds: Maximum time to wait for processing.
    
    Returns:
        str: The model's response text.
    """
    print(f"📤 Uploading audio file: {file_path}")
    
    # Upload the file
    uploaded_file = client.files.upload(file=file_path)
    print(f"✅ Upload complete: {uploaded_file.name}")
    
    # Wait for file processing if needed
    if wait_for_processing:
        start_time = time.time()
        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > max_wait_seconds:
                raise TimeoutError(
                    f"File processing timed out after {max_wait_seconds}s"
                )
            print("⏳ Waiting for file processing...")
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"File processing failed: {uploaded_file.name}")
        
        print(f"✅ File processed: {uploaded_file.state.name}")
    
    # Generate content with the uploaded file
    response = client.models.generate_content(
        model=AUDIO_MODEL,
        contents=[prompt, uploaded_file],
    )
    
    return response.text


def analyze_audio(
    file_path: str,
    prompt: str = "Describe this audio clip",
    client: Optional[genai.Client] = None,
) -> str:
    """
    Main function to analyze an audio file.
    
    Automatically selects the appropriate input method based on file size:
    - Files < 20MB: Inline bytes method (faster, simpler)
    - Files >= 20MB: Files API upload (required for large files)
    
    Args:
        file_path: Path to the audio file to analyze.
        prompt: The prompt describing what analysis to perform.
                Examples:
                - "Describe this audio clip"
                - "Transcribe this audio"
                - "What language is spoken in this audio?"
                - "Summarize the main points discussed"
        client: Optional pre-configured client. If None, creates one.
    
    Returns:
        str: The model's analysis/transcription response.
    
    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        ValueError: If the audio format is not supported.
        ConfigurationError: If the API key is not configured.
    
    Example:
        >>> result = analyze_audio("podcast.mp3", "Summarize this podcast")
        >>> print(result)
    """
    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Get file size
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"🎵 Analyzing audio: {file_path}")
    print(f"   File size: {file_size_mb:.2f} MB")
    
    # Create client if not provided
    if client is None:
        client = get_client()
    
    # Choose method based on file size
    if file_size < INLINE_SIZE_THRESHOLD:
        print(f"   Method: Inline (file < {INLINE_SIZE_THRESHOLD // (1024 * 1024)}MB)")
        return analyze_audio_inline(client, file_path, prompt)
    else:
        print(f"   Method: Upload (file >= {INLINE_SIZE_THRESHOLD // (1024 * 1024)}MB)")
        return analyze_audio_upload(client, file_path, prompt)


def transcribe_audio(
    file_path: str,
    language: Optional[str] = None,
    client: Optional[genai.Client] = None,
) -> str:
    """
    Convenience function to transcribe audio to text.
    
    Args:
        file_path: Path to the audio file.
        language: Optional language hint (e.g., "English", "Spanish").
        client: Optional pre-configured client.
    
    Returns:
        str: The transcribed text.
    """
    if language:
        prompt = f"Transcribe this audio. The language is {language}. Only output the transcription, no other text."
    else:
        prompt = "Transcribe this audio. Only output the transcription, no other text."
    
    return analyze_audio(file_path, prompt, client)


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    # Load environment variables
    load_env_file()
    
    print("=" * 60)
    print("Module A: Audio Understanding")
    print("=" * 60)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this audio clip"
        
        try:
            result = analyze_audio(audio_file, prompt)
            print("\n" + "=" * 60)
            print("📝 Analysis Result:")
            print("=" * 60)
            print(result)
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("\nUsage:")
        print("  python -m src.audio_understanding <audio_file> [prompt]")
        print("\nExamples:")
        print('  python -m src.audio_understanding sample.mp3')
        print('  python -m src.audio_understanding interview.wav "Transcribe this audio"')
        print('  python -m src.audio_understanding podcast.mp3 "Summarize the main points"')

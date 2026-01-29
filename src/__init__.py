# kopichat-ai: Modular Gemini API Audio Application
# ==================================================
# This package contains modules for interacting with Google Gemini API
# for audio understanding and real-time voice interaction.

from src.config import get_client, get_api_key
from src.audio_understanding import analyze_audio
from src.live_interaction import run_live_transcription
from src.audio_recorder import record_audio

__all__ = [
    "get_client",
    "get_api_key",
    "analyze_audio",
    "run_live_transcription",
    "record_audio",
]

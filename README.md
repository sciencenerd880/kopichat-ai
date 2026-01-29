# ☕ kopichat-ai

A modular Python application for interacting with Google Gemini API for **audio understanding** and **real-time voice interaction**.

## 🚀 Features

### Module A: Audio Understanding
- **Automatic input method selection** based on file size (inline vs. upload)
- **Transcription** - Convert audio to text
- **Analysis** - Describe, summarize, or analyze audio content
- Supports: MP3, WAV, AAC, OGG, FLAC, and more

### Module B: Real-Time Voice Interaction
- **Bidirectional audio streaming** with Gemini Live API
- **Low-latency conversation** with natural voice activity detection
- **Native audio output** for human-like responses

### Module C: Audio Recording Utility
- **Record WAV audio** from microphone
- **List audio devices** for troubleshooting
- Simple CLI for quick recordings

## 📦 Installation

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- PortAudio library (for pyaudio)

```bash
# Install PortAudio (macOS)
brew install portaudio

# Install PortAudio (Ubuntu/Debian)
sudo apt-get install portaudio19-dev

# Install PortAudio (Windows)
# PyAudio wheels include portaudio, no extra install needed
```

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/kopichat-ai.git
cd kopichat-ai

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### Configuration

1. Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Edit `.env` and add your key:

```bash
GEMINI_API_KEY=your_actual_api_key_here
```

## 🎯 Usage

### Interactive Mode
```bash
python main.py
```

This launches an interactive prompt with all available commands.

### Command Line

```bash
# Analyze an audio file
python main.py analyze recording.mp3 "Describe this audio"

# Transcribe audio to text
python main.py transcribe interview.wav

# Start real-time voice session
python main.py live

# Record audio from microphone
python main.py record output.wav 10  # Record 10 seconds

# List audio input devices
python main.py devices
```

### Programmatic Usage

```python
from src.config import load_env_file
from src.audio_understanding import analyze_audio, transcribe_audio
from src.live_interaction import run_live_session
from src.audio_recorder import record_audio

# Load API key from .env
load_env_file()

# Analyze audio
result = analyze_audio("podcast.mp3", "Summarize the main topics")
print(result)

# Transcribe audio
transcript = transcribe_audio("meeting.wav")
print(transcript)

# Record audio
record_audio("recorded.wav", duration_seconds=10)

# Start live session (async)
import asyncio
asyncio.run(run_live_session())
```

## 🔐 Security

This application follows security best practices:

- ✅ **No hardcoded API keys** - Keys loaded from environment
- ✅ **Environment-based configuration** - Uses `.env` file
- ✅ **Clear error messages** - Helpful feedback for missing config
- ✅ **`.env` in `.gitignore`** - Secrets never committed

## 📁 Project Structure

```
kopichat-ai/
├── .env                    # API keys (not committed)
├── .gitignore              # Git ignore rules
├── main.py                 # Main entry point & CLI
├── pyproject.toml          # Project configuration
├── requirements.txt        # Pip dependencies
├── src/
│   ├── __init__.py         # Package exports
│   ├── config.py           # Configuration & security
│   ├── audio_understanding.py  # Module A: File analysis
│   ├── live_interaction.py     # Module B: Real-time voice
│   └── audio_recorder.py       # Module C: Recording utility
└── notebooks/              # Jupyter notebooks
```

## 🔊 Audio Specifications

### Input (Microphone)
- Format: 16-bit PCM
- Sample Rate: 16kHz
- Channels: Mono

### Output (Speakers)
- Format: 16-bit PCM
- Sample Rate: 24kHz
- Channels: Mono

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `google-genai` | Google Gemini API SDK |
| `pyaudio` | Audio I/O for real-time streaming |

## 📝 Models Used

| Model | Use Case |
|-------|----------|
| `gemini-3-flash-preview` | Audio understanding & analysis |
| `gemini-2.5-flash-native-audio-preview` | Real-time voice interaction |

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

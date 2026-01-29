# ☕ kopichat-ai

A modular Python application for **audio understanding**, **transcription**, and **real-time AI interaction**. Powered by Google Gemini, Groq Whisper, and Apple's MLX.

## 🚀 Features

### 🎧 Audio Understanding (Gemini)
- **Transcription & Analysis** - Convert audio to text and analyze content.
- **Automatic Upload** - Handles small files (inline) and large files (File API) automatically.
- Supports: MP3, WAV, AAC, OGG, FLAC, and more.

### 🎙️ Live Transcription (Multi-Backend)
- **MLX Whisper (Local)** - **Recommended for Mac.** Ultra-fast on-device transcription optimized for Apple Silicon (M1/M2/M3).
- **Groq Whisper (Cloud)** - Lightning fast cloud-based transcription with excellent Singlish support.
- **Gemini Live (Streaming)** - Real-time streaming transcription directly from Gemini.
- **Anti-Hallucination** - Built-in filters to reduce "phantom speech" during silence.

### 🎤 Recording Utility
- **High-Quality Recording** - Save microphone input to WAV files.
- **Device Management** - List and select audio input devices.

## 📦 Installation

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- **System Libraries (macOS):**
  ```bash
  # Required for Audio I/O and Local Transcription
  brew install portaudio ffmpeg
  ```

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/kopichat-ai.git
cd kopichat-ai

# Create virtual environment and install dependencies
uv sync
```

### Configuration

Create a `.env` file in the root directory:

```bash
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

## 🎯 Usage

### Interactive Mode (Recommended)
```bash
python main.py
```
This launches a premium CLI interface where you can run all commands.

### Live Transcription Commands

```bash
# Start local transcription (Apple Silicon optimized)
python main.py live

# Choose MLX model (turbo, medium, small)
python main.py live --medium

# Start Groq cloud transcription (Best for Singlish)
python main.py live --groq

# Start Gemini Live transcription
python main.py live --gemini

# Enable verbose mode (shows more details)
python main.py live -v
```

### Other Commands

```bash
# Analyze an audio file
python main.py analyze interview.mp3 "Summarize this conversation"

# Transcribe a file using MLX (Local, Default)
python main.py live_file recording.wav

# Transcribe a file using Groq (Cloud API)
python main.py live_file recording.wav --groq

# Record 10 seconds of audio
python main.py record output.wav 10

# List audio devices
python main.py devices
```

## 📁 Project Structure

```
kopichat-ai/
├── .env                    # API keys (Secrets)
├── main.py                 # CLI & Application Entry
├── src/
│   ├── config.py           # Global Config & Security
│   ├── audio_understanding.py  # Gemini File Analysis
│   ├── live_interaction.py     # MLX / Groq / Gemini STT
│   └── audio_recorder.py       # Recording Utilities
└── notebooks/              # Research & Experiments
```

## 🛠️ Technology Stack

| Component | Technology | Use Case |
|-----------|------------|----------|
| **Core** | Python 3.11 | Logic & glue code |
| **STT (Local)** | `mlx-whisper` | Fast local transcription on M-series Mac |
| **STT (Cloud)** | `groq` | Ultra-low latency Whisper API |
| **Analysis** | `google-genai` | Audio reasoning & understanding |
| **Audio I/O** | `pyaudio` | Microphone & speaker handling |

## 📝 Models Used

| Backend | Model | Default |
|---------|-------|---------|
| **Gemini** | `gemini-2.0-flash` | Analysis & Transcription |
| **MLX** | `whisper-large-v3-turbo` | Local Real-time STT |
| **Groq** | `whisper-large-v3-turbo` | Cloud Real-time STT |

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

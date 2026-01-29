#!/usr/bin/env python3
"""
kopichat-ai: Modular Gemini API Audio Application
===================================================
A secure, modular Python application for interacting with Google Gemini API
for audio understanding and real-time voice interaction.

Modules:
- Module A: Audio Understanding (analyze_audio, transcribe_audio)
- Module B: Live Transcription (run_live_transcription)
- Module C: Audio Recording Utility (record_audio)

Security:
- API keys loaded from environment variables
- No hardcoded secrets
- Secure configuration management

Usage:
    # Interactive menu
    python main.py

    # Direct commands
    python main.py analyze <audio_file> [prompt]
    python main.py transcribe <audio_file>
    python main.py live
    python main.py record <output_file> [duration]
"""

import sys
import asyncio

# Import our modules
from src.config import load_env_file, get_client, ConfigurationError
from src.audio_understanding import analyze_audio, transcribe_audio
from src.live_interaction import run_live_transcription
from src.audio_recorder import record_audio, record_with_countdown, list_audio_devices


def print_banner():
    """Print the application banner."""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║   kopichat-ai - Audio AI Application                      ║
    ║                                                            ║
    ║   Audio understanding (Gemini) + Speech-to-text (MLX)     ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """Print the interactive menu."""
    print("""
    Available Commands:
    ─────────────────────────────────────────────────────────────
    
    AUDIO ANALYSIS (Gemini)
       analyze <file> [prompt]  - Analyze an audio file
       transcribe <file>        - Transcribe audio to text
    
     LIVE TRANSCRIPTION
        live                     - Live STT (MLX local - default)
        live --groq              - Live STT (Groq API)
        live --gemini            - Live STT (Gemini)
        
        Live Options:
        --turbo, --medium, --small - Select MLX model
        --verbose, -v              - Show debug info
        
        live_file <file>           - Transcribe file (MLX local)
        live_file <file> --groq    - Transcribe file (Groq API)
    
    RECORDING
       record <file> [duration] - Record audio from microphone
       devices                  - List audio input devices
    
    OTHER
       help                     - Show this menu
       quit / exit              - Exit the application
    
    ─────────────────────────────────────────────────────────────
    """)


def handle_analyze(args: list) -> None:
    """Handle the analyze command."""
    if not args:
        print("Usage: analyze <audio_file> [prompt]")
        return
    
    file_path = args[0]
    prompt = " ".join(args[1:]) if len(args) > 1 else "Describe this audio clip"
    
    try:
        result = analyze_audio(file_path, prompt)
        print("\n" + "=" * 60)
        print("Analysis Result:")
        print("=" * 60)
        print(result)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")


def handle_transcribe(args: list) -> None:
    """Handle the transcribe command."""
    if not args:
        print("Usage: transcribe <audio_file>")
        return
    
    file_path = args[0]
    
    try:
        result = transcribe_audio(file_path)
        print("\n" + "=" * 60)
        print("Transcription:")
        print("=" * 60)
        print(result)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")


def handle_live(args: list) -> None:
    """Handle the live transcription command."""
    from src.live_interaction import (
        MLX_MODEL_TURBO, 
        MLX_MODEL_MEDIUM, 
        MLX_MODEL_SMALL
    )
    
    # Defaults
    backend = "mlx"
    model = MLX_MODEL_TURBO
    verbose = False
    
    # Parse options
    for arg in args:
        if arg in ("--gemini", "-g", "gemini"):
            backend = "gemini"
        elif arg in ("--groq", "-q", "groq"):
            backend = "groq"
        elif arg in ("--mlx", "-m", "mlx"):
            backend = "mlx"
        elif arg in ("--verbose", "-v"):
            verbose = True
        elif arg == "--medium":
            model = MLX_MODEL_MEDIUM
        elif arg == "--small":
            model = MLX_MODEL_SMALL
        elif arg == "--turbo":
            model = MLX_MODEL_TURBO
    
    try:
        run_live_transcription(backend=backend, model=model, verbose=verbose)
    except KeyboardInterrupt:
        print("\nTranscription ended.")
    except Exception as e:
        print(f"Error: {e}")


def handle_live_file(args: list) -> None:
    """Handle the live_file transcription command."""
    from src.live_interaction import (
        transcribe_file, 
        MLX_MODEL_TURBO,
        MLX_MODEL_MEDIUM,
        MLX_MODEL_SMALL
    )
    
    if not args:
        print("Usage: live_file <audio_file> [--groq|--mlx] [--turbo|--medium|--small]")
        return
    
    file_path = args[0]
    backend = "mlx"
    model = MLX_MODEL_TURBO
    
    # Parse options
    for arg in args[1:]:
        if arg in ("--groq", "-q"):
            backend = "groq"
        elif arg in ("--mlx", "-m"):
            backend = "mlx"
        elif arg == "--medium":
            model = MLX_MODEL_MEDIUM
        elif arg == "--small":
            model = MLX_MODEL_SMALL
        elif arg == "--turbo":
            model = MLX_MODEL_TURBO
    
    try:
        result = transcribe_file(file_path, backend=backend, model=model)
        backend_name = "Groq Whisper" if backend == "groq" else "MLX Whisper"
        print("\n" + "=" * 60)
        print(f"Transcription ({backend_name}):")
        print("=" * 60)
        print(result)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")


def handle_record(args: list) -> None:
    """Handle the record command."""
    if not args:
        print("Usage: record <output_file> [duration_seconds]")
        return
    
    output_file = args[0]
    duration = float(args[1]) if len(args) > 1 else 5.0
    
    try:
        record_with_countdown(output_file, duration)
    except Exception as e:
        print(f"Error: {e}")


def handle_devices(args: list) -> None:
    """Handle the devices command."""
    list_audio_devices()


def run_interactive():
    """Run the interactive command loop."""
    print_banner()
    
    # Check configuration
    try:
        get_client()
        print("    API configuration verified\n")
    except ConfigurationError as e:
        print(f"    Configuration warning: {e}\n")
    
    print_menu()
    
    while True:
        try:
            user_input = input("kopichat> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split()
            command = parts[0].lower()
            args = parts[1:]
            
            if command in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            elif command == "help":
                print_menu()
            elif command == "analyze":
                handle_analyze(args)
            elif command == "transcribe":
                handle_transcribe(args)
            elif command == "live":
                handle_live(args)
            elif command == "live_file":
                handle_live_file(args)
            elif command == "record":
                handle_record(args)
            elif command == "devices":
                handle_devices(args)
            else:
                print(f"Unknown command: {command}")
                print("   Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def run_command_line():
    """Run with command-line arguments."""
    if len(sys.argv) < 2:
        run_interactive()
        return
    
    command = sys.argv[1].lower()
    args = sys.argv[2:]
    
    if command == "analyze":
        handle_analyze(args)
    elif command == "transcribe":
        handle_transcribe(args)
    elif command == "live":
        handle_live(args)
    elif command == "live_file":
        handle_live_file(args)
    elif command == "record":
        handle_record(args)
    elif command == "devices":
        handle_devices(args)
    elif command in ("help", "--help", "-h"):
        print_banner()
        print_menu()
    else:
        print(f"Unknown command: {command}")
        print("   Run 'python main.py help' for usage.")


if __name__ == "__main__":
    # Load environment variables from .env file
    load_env_file()
    
    # Run the application
    run_command_line()

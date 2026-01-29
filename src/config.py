"""
Global Configuration & Security Module
=======================================
This module handles secure API key management and client initialization
for the Google Gemini API using the google-genai SDK.

Security Best Practices:
- API keys are NEVER hardcoded
- Keys are retrieved from environment variables
- Clear error messages for missing configuration
"""

import os
from google import genai

# ==============================================================================
# Constants
# ==============================================================================

# Model for audio understanding (file analysis, transcription)
AUDIO_MODEL = "gemini-3-flash-preview"

# Model for real-time live audio interaction
LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

# File size threshold for inline vs. upload (20MB in bytes)
INLINE_SIZE_THRESHOLD = 20 * 1024 * 1024  # 20MB


class ConfigurationError(Exception):
    """
    Raised when there is a configuration error,
    such as a missing API key.
    """

    pass


def get_api_key() -> str:
    """
    Retrieve the Gemini API key from environment variables.
    
    Returns:
        str: The API key for authentication.
    
    Raises:
        ConfigurationError: If GEMINI_API_KEY is not set in environment.
    
    Security Note:
        - Never log or print the API key
        - Never commit .env files to version control
        - Use environment variables or secret managers in production
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ConfigurationError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Please set it in your .env file or environment:\n"
            "  export GEMINI_API_KEY='your-api-key-here'\n"
            "Or add it to your .env file:\n"
            "  GEMINI_API_KEY=your-api-key-here"
        )
    
    return api_key


def get_client() -> genai.Client:
    """
    Create and return a configured Gemini API client.
    
    Returns:
        genai.Client: An authenticated client instance.
    
    Raises:
        ConfigurationError: If authentication fails.
    """
    api_key = get_api_key()
    return genai.Client(api_key=api_key)


def load_env_file(env_path: str = ".env") -> None:
    """
    Load environment variables from a .env file.
    
    This is a simple implementation that doesn't require python-dotenv.
    For production, consider using python-dotenv or similar.
    
    Args:
        env_path: Path to the .env file (default: ".env")
    """
    if not os.path.exists(env_path):
        return
    
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse key=value pairs
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")
    try:
        load_env_file()
        client = get_client()
        print("✅ Configuration successful!")
        print(f"   Audio Model: {AUDIO_MODEL}")
        print(f"   Live Model: {LIVE_MODEL}")
        print(f"   Inline Size Threshold: {INLINE_SIZE_THRESHOLD / (1024 * 1024):.0f}MB")
    except ConfigurationError as e:
        print(f"❌ Configuration Error:\n{e}")

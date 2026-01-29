---
name: Python Coding Style
description: Coding preferences and standards for Python development in this project
---

# Python Coding Style Guidelines

This skill defines the coding preferences and standards to follow when writing or modifying Python code in this project.

## 1. Package Management: Use `uv` Instead of `pip`

Always use `uv` as the package manager. Never use `pip` directly.

### Installing Packages
```bash
# Correct
uv add <package-name>

# Incorrect - DO NOT USE
pip install <package-name>
```

### Running Scripts
```bash
# Correct
uv run <script.py>

# Incorrect - DO NOT USE
python <script.py>
```

### Creating Virtual Environments
```bash
# Correct
uv venv

# Incorrect - DO NOT USE
python -m venv .venv
```

### Syncing Dependencies
```bash
# Sync from pyproject.toml
uv sync

# Lock dependencies
uv lock
```

---

## 2. Logging: Use `logger` Instead of `print`

Always use Python's `logging` module instead of `print()` statements for any output.

### Standard Logger Setup
Add this at the top of each module (after imports):

```python
import logging

logger = logging.getLogger(__name__)
```

### Usage Examples
```python
# Correct
logger.debug("Processing started for item: %s", item_id)
logger.info("Successfully loaded %d records", count)
logger.warning("Configuration file not found, using defaults")
logger.error("Failed to connect to database: %s", error)
logger.exception("Unexpected error occurred")  # Includes stack trace

# Incorrect - DO NOT USE
print("Processing started")
print(f"Loaded {count} records")
```

### Logging Levels Guide
- `DEBUG`: Detailed information for diagnosing problems
- `INFO`: Confirmation that things are working as expected
- `WARNING`: Something unexpected happened, but the software is still working
- `ERROR`: A more serious problem, some function failed
- `CRITICAL`: A serious error, the program may be unable to continue

### Main Script Logger Configuration
In the main entry point (e.g., `main.py`), configure the root logger:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
```

---

## 3. No Emojis

Do not use emojis anywhere in the code, including:
- Comments
- Docstrings
- Log messages
- String literals
- Variable names

```python
# Correct
logger.info("Operation completed successfully")

# Incorrect - DO NOT USE
logger.info("Operation completed successfully! 🎉")
print("Starting process... 🚀")
```

---

## 4. File Header: High-Level Description

Every Python script must begin with a high-level description block immediately after the shebang (if present) and before any imports.

### Template
```python
#!/usr/bin/env python3
"""
Module: <module_name>
Description: <Brief one-line description of what this module does>

Purpose:
    <2-3 sentences explaining the main purpose and responsibility of this module>

Key Features:
    - <Feature 1>
    - <Feature 2>
    - <Feature 3>

Usage:
    <Brief example of how to use this module, if applicable>

Dependencies:
    - <Key external dependency 1>
    - <Key external dependency 2>

Author: <optional>
Created: <optional date>
"""

import logging
# ... rest of imports
```

### Example
```python
#!/usr/bin/env python3
"""
Module: audio_processor
Description: Handles real-time audio capture and processing for speech recognition.

Purpose:
    This module provides functionality to capture audio from the microphone,
    process it in real-time, and prepare it for speech-to-text conversion.
    It manages audio streams and handles buffering efficiently.

Key Features:
    - Real-time audio capture from system microphone
    - Noise reduction and audio preprocessing
    - Configurable sample rates and buffer sizes

Usage:
    from audio_processor import AudioProcessor
    processor = AudioProcessor(sample_rate=16000)
    processor.start_capture()

Dependencies:
    - pyaudio: For audio capture
    - numpy: For audio data processing
"""

import logging

logger = logging.getLogger(__name__)

# ... rest of the code
```

---

## Quick Reference Checklist

When creating or modifying Python files, verify:

- [ ] Using `uv` commands (not `pip` or raw `python`)
- [ ] Using `logger` (not `print`)
- [ ] No emojis anywhere in the code
- [ ] High-level description header is present at the top of the file

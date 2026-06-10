# Webpage Summariser & Study Assistant

A modern, AI-powered study companion Chrome extension and dashboard that extracts text, generates summaries, flashcards, multiple-choice quizzes, and enables interactive chat options.

## System Prerequisites: FFmpeg (Required for Audio Transcription)

OpenAI Whisper relies on the system-level `ffmpeg` library to handle audio transcoding. Please ensure `ffmpeg` is installed on your operating system and added to your system path:

### 1. Windows (Choose one option)
*   **Via winget (Recommended)**:
    Open PowerShell as Administrator and run:
    ```powershell
    winget install Gypsey.FFmpeg
    ```
*   **Via Chocolatey**:
    ```powershell
    choco install ffmpeg
    ```
*   **Manual Installation**:
    1. Download the static build binaries from [FFmpeg Windows Builds](https://www.gyan.dev/ffmpeg/builds/).
    2. Extract the files and rename the folder to `ffmpeg`.
    3. Move the folder to `C:\ffmpeg`.
    4. Search for "Edit the system environment variables" in your Windows search bar, click "Environment Variables", edit the `Path` variable under System variables, and append `C:\ffmpeg\bin`.
    5. Open a new Command Prompt or PowerShell and verify with: `ffmpeg -version`.

### 2. macOS
*   Install via [Homebrew](https://brew.sh/):
    ```bash
    brew install ffmpeg
    ```

### 3. Linux
*   Install via apt (Debian/Ubuntu):
    ```bash
    sudo apt update
    sudo apt install ffmpeg
    ```
*   Install via dnf (Fedora):
    ```bash
    sudo dnf install ffmpeg
    ```

---

## Getting Started

1.  Create a `.env` file in the root directory:
    ```text
    GEMINI_API_KEY=your_gemini_api_key
    GROQ_API_KEY=your_groq_api_key_optional
    ```
2.  Start the backend server using the virtual environment configuration script:
    ```powershell
    .\run_backend.bat
    ```

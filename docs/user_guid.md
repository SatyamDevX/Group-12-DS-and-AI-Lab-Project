# User Guide: Regional Dialect Synthesis Pipeline

## 1. Starting the Server

The application must be run from an active virtual environment. Environment variables must be set explicitly before launching the server process to ensure the models load correctly in the background.

**Step 1: Activate the virtual environment**
```bash
source ~/hindi-haryanvi-tts-vm/venv/bin/activate
```

**Step 2: Export environment variables**
Run these commands before opening your screen session so the background processes inherit them:
```bash
export LLM_BASE_MODEL_ID=/model_weights/llm/base
export TTS_CHECKPOINT=/model_weights/tts/best_model_16731.pth
export TTS_CONFIG=model_weights/tts/config.json
export DEVICE=cpu
export LLM_LOAD_IN_4BIT=false
export HF_HUB_OFFLINE=1
```

**Step 3: Launch a background session**
Start a multiplexer session to keep the server alive after you disconnect:
```bash
screen -S server
```

**Step 4: Start the application**
Navigate to the project root and spin up the Uvicorn server:
```bash
cd ~/hindi-haryanvi-tts-vm
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

> **Note:** To detach from the screen session while leaving the server running, press `Ctrl+A` followed by `D`. To reattach later, run `screen -r server`.

---

## 2. Using the Web Interface

Once the server is running, open your web browser and navigate to the server's IP address and port (e.g., `http://34.123.144.143:8080`). The web interface defaults to a dark theme, which can be toggled via the icon in the top right.

### 2.1 Navigation Tabs
The interface is divided into three distinct operation modes via the top tab bar:

* **▶ Full Pipeline:** Enter standard Hindi text. This mode translates the text to Haryanvi (powered by Gemma 4) and immediately synthesizes the audio. Visual indicators will track the progress of each step. 
    * *Quick actions:* Under the main "Run Pipeline" button, you can also trigger "Translate Only" or "TTS Only" for modular testing.
* **🔤 Translate Only:** Enter standard Hindi text to receive only the Haryanvi translation. No audio is generated.
* **🔊 TTS Only:** This tab changes the input box to accept **Haryanvi** text directly. It skips the translation model entirely and generates speech from the provided dialect text.

### 2.2 Input Features
* **Example Chips:** Below the text areas, there are quick-fill example sentences. Clicking any of these chips will automatically populate the input box with the selected text.
* **Character Limit:** Inputs are capped at 1000 characters to prevent memory overflow during synthesis.
* **Keyboard Shortcuts:** You can press `Ctrl + Enter` (or `Cmd + Enter` on Mac) while typing in the text area to immediately submit your request.

### 2.3 Handling Outputs
* **Translation Output:** The translated Haryanvi text will appear in the output card. Click the **Copy** button in the corner to quickly copy the text to your clipboard.
* **Audio Output:** Once synthesis is complete, an HTML5 audio player will appear, allowing you to play the generated speech directly in your browser. Click the **Download WAV** button to save the file to your local device. 
* **Error Handling:** If an input is invalid or a model fails, an error banner will drop down below the input section detailing the issue.

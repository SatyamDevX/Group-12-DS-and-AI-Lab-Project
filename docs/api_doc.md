# API Documentation: Hindi → Haryanvi TTS Pipeline

This API provides endpoints for translating Hindi text to Haryanvi and synthesizing that text into speech. 

**Important Note on Server Startup:** Models are loaded in the background when the server starts. The server will accept requests immediately, but endpoints requiring the models will return a `503 Service Unavailable` status until the loading process is complete. Check the `/health` endpoint to verify model readiness.

---

## 1. Health & Status

### `GET /health`
Checks the health of the server and the loading status of the machine learning models.

* **Response (200 OK):**
    ```json
    {
      "status": "ok", 
      "models_ready": true,
      "error": null
    }
    ```
    *(Note: `status` will be "loading" if models are still initializing in the background. `error` will contain a string message if model loading failed).*

---

## 2. Translation

### `POST /api/translate`
Translates Hindi text to Haryanvi without generating audio.

* **Request Body (`application/json`):**
    ```json
    {
      "text": "Your Hindi text here"
    }
    ```
* **Success Response (200 OK):**
    ```json
    {
      "hindi": "Your Hindi text here",
      "haryanvi": "Translated Haryanvi text here"
    }
    ```

---

## 3. Text-to-Speech (TTS)

### `POST /api/tts`
Synthesizes provided Haryanvi text into an audio file.

* **Request Body (`application/json`):**
    ```json
    {
      "text": "Your Haryanvi text here"
    }
    ```
* **Success Response (200 OK):** Returns a downloadable `.wav` file (`audio/wav`) named `haryanvi_tts.wav`.

### `POST /api/tts/base64`
Synthesizes provided Haryanvi text and returns the audio encoded as a base64 string. Ideal for web frontends avoiding binary file handling.

* **Request Body (`application/json`):**
    ```json
    {
      "text": "म्हने बेरा कोन्या के वो कड़े गया"
    }
    ```
* **Success Response (200 OK):**
    ```json
    {
      "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
      "sample_rate": 22050
    }
    ```

---

## 4. Full Pipeline (Translation + TTS)

### `POST /api/pipeline`
Executes the full pipeline: translates Hindi input to Haryanvi, generates the audio file, and returns a URL to fetch the generated audio.

* **Request Body (`application/json`):**
    ```json
    {
      "text": "आज बहुत गर्मी है"
    }
    ```
* **Success Response (200 OK):**
    ```json
    {
      "hindi": "आज बहुत गर्मी है",
      "haryanvi": "Translated Haryanvi text",
      "audio_id": "123e4567-e89b-12d3-a456-426614174000",
      "audio_url": "/api/audio/123e4567-e89b-12d3-a456-426614174000"
    }
    ```

### `POST /api/pipeline/base64`
Executes the full pipeline and returns everything (translation + base64 encoded audio) in a single JSON response. Avoids the need for a secondary fetch request.

* **Request Body (`application/json`):**
    ```json
    {
      "text": "आज बहुत गर्मी है"
    }
    ```
* **Success Response (200 OK):**
    ```json
    {
      "hindi": "आज बहुत गर्मी है",
      "haryanvi": "Translated Haryanvi text",
      "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
      "sample_rate": 22050
    }
    ```

---

## 5. Audio Retrieval

### `GET /api/audio/{audio_id}`
Serves a previously generated audio file by its unique UUID. Used in conjunction with the standard `/api/pipeline` endpoint.

* **Path Parameters:**
    * `audio_id` (string): The UUID of the audio file returned by the pipeline.
* **Success Response (200 OK):** Returns the `.wav` file (`audio/wav`).

---

## Common Error Codes

Across all endpoints, you may encounter the following HTTP error statuses:

* `400 Bad Request`: The input `text` is empty, or an invalid UUID format was provided for audio retrieval.
* `404 Not Found`: The requested audio file does not exist or has expired.
* `500 Internal Server Error`: Background model loading failed completely (check `/health` for the exact error trace).
* `503 Service Unavailable`: The models are still loading in the background thread. Wait a moment and retry.

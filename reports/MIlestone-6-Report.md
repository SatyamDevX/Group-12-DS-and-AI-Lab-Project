# 📘 Milestone 6 Report

## Deployment & Documentation

### Regional Dialect Synthesis Pipeline

---

## 1. Overview

This milestone converts the trained models into a **deployable, reproducible AI system**.

### Objective

* Input: Hindi text
* Output: Haryanvi speech

---

## 2. System Evolution (IMPORTANT)

### Model Transition (Research → Deployment)

Initially, a LLaMA 3.1 model was fine-tuned using QLoRA on 5,594 Hindi–Haryanvi pairs.  
While the model achieved strong translation quality, it suffered from:

- High inference latency  
- Large memory footprint  

To enable real-time deployment, the system was redesigned:

- Replaced LLaMA inference with Gemma GGUF model  
- Applied prompt engineering with few-shot examples  
- Achieved faster inference while maintaining acceptable dialect quality  

This transition highlights a key engineering tradeoff between model accuracy and deployment feasibility.

### Phase 1 — Training

* Model: LLaMA 3.1 (QLoRA)
* Dataset: 5,594 parallel pairs
* Outcome: High-quality translation


### Problem

* High latency
* Large model size

---

### Phase 2 — Deployment Optimization

* Switched to **Gemma GGUF**
* Used **prompt engineering**
* Added **few-shot examples**

### Result

* Faster inference
* Production-ready system

---

## 3. Final Architecture

```id="arch2"
Hindi Input
   ↓
Gemma (Prompt-based Translation)
   ↓
Haryanvi Text
   ↓
VITS Model
   ↓
Audio Output
```

---

## 4. Deployment Details

### 4.1 Platform

* GCP Virtual Machine
* FastAPI backend
* Uvicorn server

---

### 4.2 System Components

| Component | Description       |
| --------- | ----------------- |
| Frontend  | Static HTML       |
| Backend   | FastAPI           |
| Models    | Local GGUF + VITS |
| Storage   | Local filesystem  |

---

### 4.3 Running the System

```bash id="run2"
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 
```

---

## 5. API Design

### Endpoints

| Endpoint               | Function             |
| ---------------------- | -------------------- |
| `/health`              | Status               |
| `/api/translate`       | Hindi → Haryanvi     |
| `/api/tts`             | Text → Speech        |
| `/api/pipeline`        | Full pipeline        |
| `/api/pipeline/base64` | Full pipeline (JSON) |

---

## 6. Inference Pipeline

```python id="pipe1"
def pipeline(text):
    haryanvi = translate(text)
    audio = synthesize(haryanvi)
    return haryanvi, audio
```

---

## 7. Implementation Highlights

### 7.1 Background Model Loading

* Models load in separate thread
* Server starts instantly

### 7.2 Async Execution

* Uses `asyncio` + thread executor
* Prevents blocking

### 7.3 Error Handling

* 503 → models loading
* 400 → invalid input
* 500 → model failure

(Implemented in FastAPI backend )

---

## 8. Technical Details

## Dataset Summary

- Text Dataset: 5,594 Hindi–Haryanvi sentence pairs  
- Audio Dataset: ~5,500 Haryanvi speech samples  

These datasets were used for training the translation and TTS models respectively.

### Translation

* Initial model: LLaMA 3.1 (8B) using QLoRA 


* Delployed Model: gemma-4-E4B-it-Q4_K_S
* Method: Prompt engineering

### TTS

* Model: Coqui VITS (as base model)
* Checkpoint: best_model_16731.pth (fully trained)

---

## 9. Reproducibility

* requirements.txt
* local model checkpoints
* consistent API interface

---

## 10. Example

Input:

```id="ex3"
तुम कहाँ जा रहे हो?
```

Output:

```id="ex4"
तू कड़े जा रया सै?
(Audio generated)
```

---

## 11. Results

### Translation

* Captures dialect variations
* Some Hindi bias remains

### TTS

* Generates valid waveform
* Naturalness improving

---

## Performance Considerations

- LLaMA inference was slow (seconds per request)
- Gemma GGUF reduced latency significantly
- Background model loading prevents server blocking
- Async API ensures non-blocking request handling

---

## Frontend–Backend Interaction

- User enters Hindi text in HTML interface
- Request sent to FastAPI backend
- Backend processes:
  1. Translation
  2. TTS synthesis
- Response returned as audio or JSON

---

## 12. Limitations

- Translation still shows Hindi influence in some cases  
- TTS prosody not fully natural  
- Limited dataset size affects generalization  

## 13. Challenges

* Low-resource dataset
* Dialect ambiguity
* TTS prosody issues

---

## 14. Future Work

* Larger dataset
* Fine-tune Gemma
* Improve TTS quality
* 

---

## 15. Conclusion

This milestone successfully delivers:

* Deployable AI system
* End-to-end pipeline
* Real-time inference

The project demonstrates a practical solution for **low-resource dialect synthesis**.

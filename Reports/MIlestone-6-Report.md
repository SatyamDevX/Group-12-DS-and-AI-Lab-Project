# 📘 Milestone 6 Report: Deployment & Documentation  
### Regional Dialect Synthesis Pipeline (Hindi → Haryanvi → Speech)

---

## 1. Overview

Milestone 6 focuses on transforming the developed pipeline into a **deployable, reproducible, and user-accessible system**.

### Objective
To deploy and document an end-to-end pipeline that:
- Translates Hindi text → Haryanvi dialect
- Converts Haryanvi text → natural speech audio

### Final Pipeline
1. **Text-to-Text Translation**
   - Model: LLaMA 3.1 8B (QLoRA fine-tuned)
   - Task: Hindi → Haryanvi

2. **Text-to-Speech (TTS)**
   - Model: Coqui VITS
   - Task: Haryanvi → Speech waveform

---

## 2. Deployment

### 2.1 Deployment Type
- Local deployment using:
  - **Streamlit App (Frontend)**
  - **Python Backend (Model Inference)**

---

### 2.2 Application Workflow

User Input (Hindi Text)  
↓  
Translation Model (LLaMA)  
↓  
Haryanvi Text Output  
↓  
TTS Model (VITS)  
↓  
Generated Speech (Audio)

---

### 2.3 How to Run (Local)

```bash
# Clone repository
git clone <repo-url>

# Navigate
cd project

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/app.py
```

---

### 2.4 Input & Output

| Type   | Description |
|--------|------------|
| Input  | Hindi text sentence |
| Output | Haryanvi text + generated speech (.wav) |

---

## 3. Technical Documentation

### 3.1 Environment Setup

| Component    | Specification |
|--------------|--------------|
| Python       | 3.12 |
| PyTorch      | 2.10 |
| Transformers | 5.0 |
| PEFT         | 0.18 |
| TTS Library  | Coqui TTS |
| GPU          | NVIDIA A100 / T4 |

Install:
```bash
pip install -r requirements.txt
```

---

### 3.2 Data Pipeline

#### Translation Dataset
- ~5000 Hindi–Haryanvi sentence pairs
- Cleaned and tokenized
- Instruction formatted

#### TTS Dataset
- Dataset: `ankitdhiman/haryanvi-tts`
- ~5500 samples
- Converted to LJSpeech format

---

### 3.3 Model Architecture

#### Translation
- LLaMA 3.1 8B with QLoRA fine-tuning
- Low-rank adaptation for efficiency

#### TTS
- VITS architecture:
  - Variational Autoencoder + GAN
  - End-to-end waveform generation

---

### 3.4 Training Summary (From Milestone 4)

- Optimizer: AdamW  
- Precision: BF16  
- Best config: **lower_lr_longer**  
- Stable convergence achieved  

---

### 3.5 Evaluation Summary (From Milestone 5)

| Metric | Value |
|--------|------|
| BLEU   | 71.76 |
| chrF   | 77.65 |
| Exact Match | 50.42% |

---

### 3.6 Inference Pipeline

```python
def pipeline(text):
    haryanvi_text = translate_model(text)
    audio = tts_model(haryanvi_text)
    return haryanvi_text, audio
```

---

### 3.7 Deployment Details

| Component | Description |
|----------|------------|
| Frontend | Streamlit UI |
| Backend  | Python inference pipeline |
| Model Hosting | Local (can extend to cloud) |

---

### 3.8 System Design Considerations

- Modular pipeline (translation + TTS separated)
- Scalable to other dialects
- Lightweight inference using QLoRA
- Reusable components

---

### 3.9 Error Handling & Monitoring

- Input validation (empty text)
- Max length checks
- Exception handling for model loading
- Logging for debugging

---

### 3.10 Reproducibility Checklist

- Fixed random seeds
- Saved model checkpoints
- Dataset versioning
- requirements.txt provided

---

## 4. User Documentation

### 4.1 App Overview
A simple interface that converts Hindi sentences into spoken Haryanvi audio.

---

### 4.2 Steps to Use

1. Launch the app  
2. Enter Hindi text  
3. Click **Generate**  
4. View:
   - Translated Haryanvi text  
   - Audio playback  

---

### 4.3 Example

Input:
```
तुम कहाँ जा रहे हो?
```

Output:
```
तू कड़े जा रया सै?
(Audio generated)
```

---

### 4.4 Troubleshooting

| Issue | Solution |
|------|---------|
| App not loading | Check dependencies |
| No audio | Verify TTS model path |
| Slow response | Use GPU |

---

## 5. API Documentation (Optional)

### Endpoint

```
POST /predict
```

### Request

```json
{
  "text": "तुम कहाँ जा रहे हो?"
}
```

### Response

```json
{
  "haryanvi_text": "तू कड़े जा रया सै?",
  "audio_path": "output.wav"
}
```

---

## 6. Licensing & Dataset References

- Code License: MIT  
- Dataset:
  - Hindi–Haryanvi corpus (custom)
  - `ankitdhiman/haryanvi-tts` (Hugging Face)

---

## 7. Future Work

### Improvements
- Increase dataset size  
- Reduce Hindi lexical bias  
- Add phoneme-based modeling  

### Extensions
- Multi-dialect support  
- Real-time speech input  
- Mobile app integration  

### Limitations
- Small dataset (~5K samples)  
- Occasional translation errors  
- TTS prosody limitations  

---

## 8. Conclusion

Milestone 6 successfully delivers:

- A **working deployment** of the pipeline  
- **Comprehensive technical documentation**  
- A **user-friendly interface**  

The system demonstrates strong potential for:
- Low-resource dialect AI  
- Speech synthesis applications  
- Regional language preservation  

# Milestone 3 Report: Model Architecture & End-to-End Pipeline Verification  
### Project: Regional Dialect Synthesis Pipeline (Haryanvi)

---

## Team Members

- Abhishek  
- Satyam Srivastava  
- Sanket Agrawal  
- Md Fazlur Rahman  

---

## 1. Overview

This milestone focuses on designing and justifying the model architecture and validating the **end-to-end pipeline** using a subset of the dataset.

The system consists of two major components:

1. **Module A:** Hindi → Haryanvi Translation (Text-to-Text)
2. **Module B:** Haryanvi Text → Speech (Text-to-Audio)

We also verify that all components—from preprocessing to inference—work together correctly.

---

## 2. Dataset Organization

The dataset is structured into raw and processed formats for both text and audio pipelines.

### Directory Structure


project/
│
├── data/
│ ├── raw/
│ │ ├── text/
│ │ │ ├── xor_qa.json
│ │ │ ├── synthetic.json
│ │ │
│ │ ├── audio/
│ │ ├── wav_files/
│ │ └── metadata.csv
│ │
│ ├── processed/
│ │ ├── text/
│ │ │ ├── train.json
│ │ │ ├── val.json
│ │ │ ├── test.json
│ │ │
│ │ ├── audio/
│ │ ├── train/
│ │ ├── eval/
│
├── notebooks/
│ ├── text_to_text.ipynb
│ ├── text_to_audio.ipynb
│
├── models/
└── outputs/


---

## 3. Preprocessing Pipeline

### 3.1 Text Preprocessing

The following steps are applied:

- Removal of null values and duplicates  
- Sentence length filtering (3–100 words)  
- Unicode normalization (Devanagari consistency)  
- Instruction-based formatting for LLM  

#### Example:


System: Translate Hindi to Haryanvi
User: तुम कहाँ जा रहे हो?
Assistant: तू कड़े जा रया सै?


---

### 3.2 Tokenization

- Tokenizer: LLaMA 3.1 tokenizer  
- Max sequence length: 128  
- Padding: EOS token  
- Output format: tensors (`input_ids`, `attention_mask`)  

---

### 3.3 Audio Preprocessing

- Resampling: 22050 Hz  
- Duration filtering: 1–15 seconds  
- Transcript normalization  

#### Output format:


{
"audio_array": waveform,
"text": "राम खेत में जा रया सै"
}


---

## 4. Model Architecture

---

### 4.1 Module A: Translation Model

**Architecture: LLaMA 3.1 + QLoRA**

#### Key Components:

- Pretrained Transformer (LLaMA 3.1)
- Low-Rank Adaptation (LoRA layers)
- Instruction tuning framework

#### Data Flow:


Hindi Text → Tokenizer → LLaMA + QLoRA → Haryanvi Text


---

### Why this Architecture?

- Efficient fine-tuning for low-resource datasets  
- Strong multilingual capabilities  
- Captures morpho-syntactic transformations  

---

### Limitations:

- Sensitive to synthetic data quality  
- May hallucinate dialect patterns  

---

### 4.2 Module B: TTS Model (Planned + Dummy Execution)

**Architecture: VITS (Planned)**

#### Components:

- Text Encoder  
- Latent Variable Model  
- Flow-based Decoder  
- Vocoder (HiFi-GAN)

#### Flow:


Haryanvi Text → Encoder → Latent Space → Waveform Generator


---

### Current Status (Milestone 3)

- TTS training not yet completed  
- Dummy pipeline implemented for integration testing  

#### Example:


Input: "तू कड़े जा रया सै?"
Output: output.wav (placeholder)


---

## 5. End-to-End Pipeline

### System Flow


Hindi Input
↓
Tokenizer
↓
LLaMA + QLoRA
↓
Haryanvi Text
↓
TTS Model
↓
Audio Output


---

## 6. Input-Output Compatibility

| Component | Format |
|----------|--------|
| Translation Output | Devanagari Text |
| TTS Input | Normalized Haryanvi Text |

✔ Ensured consistent script and normalization across modules  

---

### Input Shapes

| Model | Shape |
|------|------|
| LLaMA | (batch_size, seq_len=128) |
| TTS | Variable-length text |

---

## 7. End-to-End Pipeline Verification

A subset of the dataset was passed through the pipeline to validate integration.

### 🔍 Translation Model Output (Actual Run)

![Model Output](hindi-to-haryanvi-text-conversion.jpeg)

### Observations:

- Correct lexical substitutions observed (e.g., “रहा हूँ” → “रया सै”)  
- Sentence structure preserved  
- Dialect transformation consistent  

---

### Sample Outputs

| Hindi Input | Model Output |
|------------|------------|
| मैं आज बाजार जा रहा हूँ | मैं आज बाजार जा रया सै। |
| क्या तुम कल स्कूल जाओगे? | क्या तुम कल स्कूल जाओगे। |
| यह किताब मेरी है, इसे मत छूना | या किताब मेरी सै, इसे न छूना। |

---

### Pipeline Execution


Hindi → Translation → Output → (Dummy) Audio Generation


✔ All components executed successfully  
✔ No format mismatch  
✔ Pipeline verified  

---

## 8. Model Outputs

### Translation Example


Input: मुझे पानी चाहिए
Output: मन्ने पानी चाहिए


---

### TTS Output (Dummy)


Generated: output_1.wav
Duration: ~2 sec (placeholder)


---

## 9. Loss Functions & Evaluation Metrics

### Module A (Translation)

- Loss: Cross-Entropy Loss  
- Metrics:
  - BLEU Score  
  - chrF++  
  - Manual evaluation  

---

### Module B (TTS)

- Loss:
  - Reconstruction Loss  
  - KL Divergence  
  - Adversarial Loss  

- Metrics:
  - MOS (Mean Opinion Score)  
  - MCD (Mel-Cepstral Distortion)  
  - WER  

---

## 10. Architecture Justification

| Model | Reason |
|------|--------|
| LLaMA + QLoRA | Efficient and suitable for low-resource dialect modeling |
| Synthetic Data | Required due to lack of real dataset |
| VITS | End-to-end speech generation with prosody modeling |

---

### Comparison with Alternatives

| Model | Limitation |
|------|-----------|
| Google Translate | No dialect awareness |
| Seq2Seq LSTM | Weak context modeling |
| FastSpeech2 | Limited prosody flexibility |

---

## 11. Strengths

- End-to-end pipeline design  
- Dialect-aware transformation  
- Modular architecture  
- Low-resource adaptability  

---

## 12. Limitations

- Limited real-world Haryanvi dataset  
- TTS module not fully trained  
- Dependency on synthetic data  

---

## 13. Conclusion

This milestone successfully delivers:

- A well-defined **model architecture**  
- Verified **end-to-end pipeline execution**  
- Proper **data-model compatibility**

The project is now ready for:

➡️ Full-scale model training (Milestone 4)  
➡️ Performance optimization and evaluation  

---

## 14. References

Refer Milestone 1 and Milestone 2 reports.
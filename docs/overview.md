# Regional Dialect Synthesis Pipeline (Haryanvi)

## Project Overview

This project provides a complete, end-to-end machine learning pipeline that converts standard Hindi text into spoken Haryanvi (Bangru). It bridges the low-resource dialect gap by combining a Large Language Model (LLM) for translation with a GAN-based Text-to-Speech (TTS) engine for audio synthesis.

## Architecture

```text
[Hindi Text Input]
      │
      ▼
[Translation Model]      ← Translation Stage (LLaMA 3.1 8B or Gemma 4B)
      │
      ▼
[Haryanvi Text (Devanagari)]
      │
      ▼
[Coqui VITS (GAN TTS)]   ← Synthesis Stage
      │
      ▼
[Audio Output — 22050 Hz WAV]
```

The system operates in a two-stage pipeline, processing standard Hindi input into an eventual Haryanvi audio waveform. 

### 1. Text-to-Text Translation (Hindi → Haryanvi)
This stage translates standard Devanagari Hindi into Devanagari Haryanvi. The architecture supports two separate modeling approaches:
* **LLaMA 3.1 8B Instruct:** Fine-tuned using QLoRA adapters on a bilingual Hindi-Haryanvi parallel corpus. 
* **Gemma 4B:** Utilizes prompt engineering and few-shot learning to achieve translation natively.

### 2. Text-to-Speech Synthesis (Haryanvi Text → Audio)
This stage converts the translated Haryanvi text into a natural-sounding 22050 Hz WAV audio file.
* **Coqui VITS:** A single-stage end-to-end TTS model (Conditional Variational Autoencoder with Adversarial Learning). It is warm-started from a Hindi base model and fine-tuned on a dedicated Haryanvi audio corpus using character-level tokenization to preserve dialect-specific pronunciation.

---

## Live Deployment

A live testing environment is currently deployed and accessible via a FastAPI backend. You can test the end-to-end pipeline at the following address:
* **Live Server:** [http://34.123.144.143:8080/](http://34.123.144.143:8080/)

---

## Quick Setup

System dependencies are managed via pip. To prepare your environment, install the requirements:

```bash
pip install -r requirements.txt
```

---

## Documentation Directory

This document serves as a high-level summary. For specific details regarding usage, API integration, or model training, please refer to the dedicated documentation files:

* **`README.md`**: Repository structure, prerequisites, and startup commands.
* **`user_guide.md`**: Instructions for end-users on how to format inputs and interact with the system.
* **`api_doc.md`**: Detailed HTTP methods, request payloads, and response schemas for the FastAPI server.
* **`technical_doc.md`**: In-depth training logs, dataset preprocessing, hyperparameter configurations, ablation studies, and evaluation metrics (chrF, BLEU, Loss curves) for the LLaMA and VITS models.

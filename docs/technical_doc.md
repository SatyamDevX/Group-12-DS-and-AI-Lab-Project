# Technical Documentation: Regional Dialect Synthesis Pipeline

## 1. Architecture Overview

This pipeline translates standard Hindi text into Haryanvi (Bangru) speech via two decoupled components: an LLM-based Text-to-Text translation module and a VITS-based Text-to-Speech (TTS) synthesis module. 

**Compute Environment:** NVIDIA A100/T4 (16GB-40GB VRAM), BF16/FP16 Precision, Python 3.12, PyTorch 2.10.

---

## 2. Module A: Text-to-Text Translation

The translation module accepts standard Hindi (Devanagari) and outputs Haryanvi (Devanagari). It implements two distinct approaches: Supervised Fine-Tuning (SFT) for high-accuracy cloud deployment and a Quantized GGUF fallback for edge inference.

### 2.1 Dataset & Preprocessing
* **Corpus:** ~5,000 parallel Hindi-Haryanvi sentence pairs.
* **Filtering:** Strict sentence-length bounds (3–100 words) to eliminate padding inefficiencies.
* **Formatting:** Mapped to a causal language modeling task using a strict system/user/assistant chat template.

### 2.2 Primary SFT Model: LLaMA 3.1 8B Instruct
* **Quantization:** Base weights loaded in 4-bit NormalFloat (NF4) with double quantization. Compute dtype is BF16.
* **QLoRA Configuration:** Trainable adapters injected into attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Rank $r=16$, $\alpha=32$.
* **Assistant-Only Loss Masking:** Causal cross-entropy loss is computed exclusively on the assistant's response. All system prompt tokens and padding are masked with `-100` to prevent gradient saturation on boilerplate text.
* **Hyperparameters:** The optimal configuration utilized a cosine annealing schedule with a peak learning rate of $1e-4$ over 4 epochs. 
* **Evaluation Anomaly:** During validation, generative metrics (BLEU, Exact Match) collapsed to identical static values (e.g., Exact Match exactly 50.42%) across divergent hyperparameter runs, indicating a critical inference script bug where dynamic LoRA adapters failed to load into memory or predictions were cached improperly.

### 2.3 Edge Inference Fallback: Gemma 4 E4B GGUF
Implemented in `translator.py` using `llama-cpp-python` for deterministic, low-resource deployment.
* **Prompt Engineering:** Enforces strict linguistic rules via few-shot prompting, specifically phonetic shifts (न → ण), honorific collapse (आप → थम/तू), and dialect-specific postpositions.
* **Decoding Stategy:** Low entropy (`temperature=0.1`, `top_p=0.95`, `repeat_penalty=1.05`).
* **Devanagari Validation Loop:** Output is intercepted by a regex Unicode property matcher (`\p{Devanagari}`). If the Devanagari-to-Latin character ratio drops below 0.5, the script injects a synthetic failure message (`[अमान्य आउटपुट]`) into the context window and triggers a retry with an entropy bump (`temperature=0.3`) to break the hallucination loop.

---

## 3. Module B: Text-to-Speech Synthesis

The TTS engine uses a single-stage Coqui VITS (Conditional Variational Autoencoder with Adversarial Learning) architecture, transforming Haryanvi text directly into a 22050 Hz waveform without an intermediate vocoder stage.

### 3.1 Audio Preprocessing & Tokenization
* **Corpus & Filtering:** 5,500 utterances filtered down to 2,714 valid training samples by dropping nulls, duplicates, and restricting word counts to 3–25 words to ensure stable attention alignment.
* **Multi-script Transliteration:** Source dataset contained mixed scripts. Bengali was transliterated via `indic-transliteration`; Urdu via a hardcoded mapping.
* **NFC Normalization:** Unicode NFC normalization was enforced globally. Un-normalized decomposed Devanagari characters silently break tokenizer attention maps.
* **Tokenization Strategy:** Standard phonemizers (e.g., `espeak-ng`) were explicitly bypassed because the `hi` (Hindi) locale mispronounces Bangru-specific retroflexes. Input is processed via `basic_cleaners` into a 173-character vocabulary.

### 3.2 Training Configuration
* **Transfer Learning:** Initialized from `SYSPIN/vits_Hindi_Female`. Because Hindi and Haryanvi share significant phonological overlap, 100% of the `waveform_decoder`, `flow`, and `duration_predictor` weights were successfully transferred. Only the `text_encoder` embedding layer required partial re-initialization due to the expanded 173-character vocabulary.
* **GAN Parameters:** Equal learning rates of $1e-5$ for both Generator and Discriminator to ensure stable adversarial equilibrium. 
* **Precision:** Mixed precision (FP16) enabled a batch size of 16 without OOM errors on 16GB VRAM.

### 3.3 Convergence Profile
Trained for 300 epochs (~50,700 steps).
* **Mel Loss:** Dropped sharply from ~32.0 to ~13.36, continually converging without plateauing.
* **KL Divergence:** Collapsed from 3.70 to 1.50 within 25 epochs, indicating rapid posterior encoder alignment.
* **Discriminator Loss:** Exhibited initial volatility before stabilizing strictly at ~2.75, confirming zero mode collapse across the 300 epochs.
* **Feature Loss:** Saturated early (~epoch 165 at 6.31), providing no further optimization signal.

---

## 4. Pipeline Integration & Systemic Constraints

### 4.1 Handoff Integrity
Module A outputs pure UTF-8 Devanagari text. Module B's `basic_cleaners` accepts this string directly without an intermediate phonemization layer. This direct character-to-spectrogram mapping preserves the translated morphology without interference from external language libraries.

### 4.2 Known Technical Limitations
1. **Hindi Lexical Bias (TTS):** Due to the heavy reliance on the `SYSPIN` Hindi warm-start weights and a limited Haryanvi dataset (~1.5 hours of audio), the decoder exhibits a latent acoustic prior that occasionally forces standard Hindi pronunciations (e.g., *जा रहा है* overriding *जा रया सै*) on rare or complex tokens.
2. **Vocabulary Deficits:** Nukta characters (`ख़`, `ड़`) and specific punctuation (`—`, `|`) are absent from the TTS character configuration. During inference, these are silently discarded, which can subtly alter loanword pronunciation.
3. **Generative Evaluation Tooling:** The inability to dynamically load LoRA weights during validation batching (Module A) invalidates the quantitative generative metrics (BLEU, Exact Match) for cross-experiment comparison. Future iterations require an updated HF `PeftModel` inference script.

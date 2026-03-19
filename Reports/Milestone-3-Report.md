# Milestone 3 Report: Model Architecture & End-to-End Pipeline Verification

**Project:** Regional Dialect Synthesis Pipeline (Standard Hindi to Haryanvi/Bangru)  
**Team Members:** Abhishek, Satyam Srivastava, Sanket Agrawal, Md Fazlur Rahman  

## 1. Overview
This milestone outlines the design, justification, and verification of our dual-architecture machine learning pipeline. The project aims to cross the low-resource dialect barrier by breaking the task into two sequential domains:
* **Module A (Text-to-Text):** Translating Standard Hindi to the Haryanvi (Bangru) dialect.
* **Module B (Text-to-Speech):** Synthesizing the translated Haryanvi text into highly natural, dialect-accurate speech.

This report verifies the complete end-to-end pipeline by running a subset of the data ("tracer bullet" execution) to confirm that preprocessing, tokenization, forward passes, and output generations function cohesively without dimension mismatches or architectural failures.

---

## 2. Dataset Organization
The datasets are strictly isolated into distinct environments to prevent data leakage and memory overhead, utilizing dedicated Hugging Face repositories for both domains.

**Text Dataset:** `Satyam-Srivastava/RDS` (Parallel Hindi-Haryanvi corpus).
**Audio Dataset:** `ankitdhiman/haryanvi-tts` (~4,000 single-speaker Haryanvi text-audio pairs).

### Directory Structure
```text
project_root/
│
├── data_text/
│   ├── raw/
│   │   ├── train_5000.json
│   │   └── supplementary_pairs.txt
│   ├── processed/
│   │   ├── train.json (80%)
│   │   ├── val.json   (10%)
│   │   └── test.json  (10%)
│
├── data_audio/
│   ├── raw/
│   │   └── hf_cache_streaming/ (Bypassing torchcodec limits)
│   ├── haryanvi_vits_data/ (LJSpeech Format)
│   │   ├── wavs/
│   │   │   ├── audio_0.wav
│   │   │   └── ...
│   │   └── metadata.csv (Matched text-to-audio index)
│
├── notebooks/
│   ├── 01_llama3_text_translation.ipynb
│   └── 02_vits_acoustic_modeling.ipynb
│
└── models/
    ├── rds_lora_adapter/
    └── tts_output/vits_haryanvi_test/
```

---

## 3. Preprocessing Pipeline

### 3.1 Module A: Text Preprocessing & Tokenization
Before passing text to the LLM, the data undergoes strict formatting to map to a causal language modeling task:
1.  **Length Filtering:** Sentences outside the 3–25 word boundary are dropped to prevent padding inefficiencies.
2.  **Chat Template Formatting:** Data is mapped to the LLaMA 3.1 instruction format.
    * *System:* "You are a regional dialect translator..."
    * *User:* "तुम कहाँ जा रहे हो?" (Hindi)
    * *Assistant:* "तू कड़े जा रया सै?" (Haryanvi)
3.  **Tokenization:** We use the `Meta-Llama-3.1-8B-Instruct` tokenizer. 
    * **Padding/Truncation:** Sequences are padded using the End-Of-Sequence (EOS) token to a `max_seq_length` of 256.

### 3.2 Module B: Audio Preprocessing & Phonemization
Audio processing requires mapping continuous waveforms to discrete phonemes:
1.  **Format Extraction:** We bypass the Hugging Face `torchcodec` bug by streaming raw bytes (`decode=False`) and using `soundfile` to extract explicit NumPy arrays.
2.  **Resampling:** All audio is strictly standardized to **22050 Hz** (required for VITS).
3.  **Metadata Alignment:** Exported to the standard `LJSpeech` format (`filename|transcript|normalized_transcript`).
4.  **Phonemizer:** We apply the `espeak-ng` engine configured for Hindi (`hi`) to map Devanagari script into the International Phonetic Alphabet (IPA). This handles schwa-deletion specific to North Indian dialects.

---

## 4. Model Architecture & Data Flow Diagram

### 4.1 Module A: LLaMA 3.1 + QLoRA (Text Translation)
* **Base Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
* **Quantization:** 4-bit NormalFloat (NF4) via `bitsandbytes` (enables A100 VRAM compliance).
* **Adaptation:** Low-Rank Adaptation (LoRA) targeting all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) to learn morpho-syntactic dialect shifts without catastrophic forgetting.

### 4.2 Module B: VITS via Transfer Learning (Text-to-Speech)
* **Architecture:** Variational Inference with adversarial learning for end-to-end Text-to-Speech.
* **Initialization Strategy:** To overcome the 4,000-sample limitation, we initialize using pre-trained Hindi checkpoints (e.g., AI4Bharat `vits_rasa_13`).
* **Components:**
    * *Text Encoder:* Maps phonemes to hidden representations.
    * *Stochastic Duration Predictor:* Captures the variable, highly compressed syllabic rhythm of Bangru.
    * *Posterior Encoder (Normalizing Flows):* Maps linear spectrograms to a complex latent space.
    * *Vocoder:* Integrated HiFi-GAN for direct waveform generation.

### 4.3 Data Flow Architecture

```text
[RAW HINDI INPUT] ("मैं आज बाजार जा रहा हूँ")
       │
       ▼
(LLaMA Tokenizer) ───────> [Input_IDs: Shape (Batch, Seq_Len)]
       │
       ▼
[Module A: LLaMA 3.1 + QLoRA] ──> (Next Token Prediction / Cross-Entropy)
       │
       ▼
[HARYANVI TEXT OUTPUT] ("मैं आज बाजार जा रया सै।")
       │
       ▼
(espeak-ng Phonemizer) ──> [Phoneme Sequence: Shape (Batch, Phoneme_Len)]
       │
       ▼
[Module B: VITS Text Encoder & Duration Predictor]
       │
       ▼
[Latent Acoustic Space] ──> (Flow-based Decoder + HiFi-GAN)
       │
       ▼
[RAW AUDIO WAVEFORM] (.wav file @ 22050Hz)
```

---

## 5. Input-Output Compatibility & Tensor Shapes
The pipeline ensures flawless handoffs between components:

| Component / Layer | Expected Input Format | Tensor Shape / Dimensions | Output Format |
| :--- | :--- | :--- | :--- |
| **LLaMA Base** | Tokenized IDs + Attention Masks | `(batch_size=4, seq_len=≤256)` | Causal Logits `(batch, seq, vocab_size)` |
| **LoRA Adapters** | Hidden States | `rank=16, alpha=32` | Delta Hidden States |
| **VITS Encoder** | Phoneme IDs (from `espeak-ng`) | `(batch_size=8, text_length)` | Encoded latent `(batch, hidden_dim, text_length)` |
| **VITS Vocoder** | Mel-spectrograms | `(batch, 80, mel_length)` | Audio Waveform `(batch, 1, audio_length)` |

*Integration Note:* Module A outputs pure Devanagari UTF-8 strings. Module B accepts these strings natively, passing them to the Coqui TTS audio processor for phonemization.

---

## 6. Architecture Justification & Alternatives

### Text Module: Why LLaMA 3.1 QLoRA?
* **Suitability:** Haryanvi lacks the millions of parallel sentences required to train a Seq2Seq model (like MarianMT or standard T5) from scratch.
* **Comparison:** Standard translation models fail on dialects because they treat them as entirely new languages. LLaMA 3.1 already possesses deep semantic understanding of standard Hindi. QLoRA simply "steers" that existing knowledge toward Bangru syntax. 

### Audio Module: Why VITS (over XTTS / FastPitch)?
* **Suitability:** VITS features a *stochastic duration predictor*. Haryanvi (Bangru) exhibits aggressive syllabic compression and a flatter melodic variance compared to standard Hindi.
* **Comparison (vs. FastSpeech2):** FastSpeech2 relies on deterministic (MSE) duration prediction, resulting in a robotic, metronomic cadence that ruins dialect authenticity. 
* **Comparison (vs. XTTS):** Autoregressive models (XTTS/VALL-E) suffer from high inference latency and hallucinations. VITS provides real-time, end-to-end generation in a single forward pass without the compound errors of two-stage (FastPitch + Vocoder) pipelines.

---

## 7. End-to-End Pipeline Verification (Tracer Bullet Execution)
To verify component integration, we executed a "tracer bullet" run on a hyper-restricted dataset. 

**Module A Verification:**
* Executed `SFTTrainer` on the dataset using a batch size of 4.
* Loss decreased successfully from `0.094` to `0.090` over 3 epochs.
* LoRA adapters merged and saved successfully.

**Module B Verification:**
* Streamed exactly 50 samples from the audio dataset to bypass API limits.
* Executed the VITS Coqui `Trainer` loop for 1 epoch.
* **Result:** The Generative Adversarial Network (GAN) loops stabilized, the phonemizer successfully parsed the Devanagari text, and an output checkpoint was successfully compiled without dimension mismatch errors.

---

## 8. Model Outputs & Examples

**Translation Execution (Actual Model Output):**
* *Input:* `क्या तुम कल स्कूल जाओगे?`
* *Model Output:* `क्या तुम कल स्कूल जाओगे।` *(Note: Requires further tuning on interrogatives).*
* *Input:* `यह किताब मेरी है, इसे मत छूना`
* *Model Output:* `या किताब मेरी सै, इसे न छूना।` *(Excellent dialect substitution).*

**Pipeline Execution (Text-to-Audio Integration):**
* Fed the generated string: `"मैं आज बाजार जा रया सै।"` into the newly initialized VITS checkpoint.
* **Output Generated:** `test_haryanvi_output.wav`
* *Status:* Waveform successfully generated. As expected from a 1-epoch/50-sample run, the fidelity is low, but it absolutely verifies the architectural math and pipeline bridging.

---

## 9. Loss Functions & Evaluation Metrics

### Module A (Text Translation)
* **Loss Function:** Causal Language Modeling Loss (Standard Cross-Entropy).
* **Metrics:** * **chrF++:** (Character n-gram F-score) Optimal for morphologically rich dialects like Haryanvi.
    * **BLEU Score:** For exact lexical overlap.

### Module B (Text-to-Speech)
* **Loss Functions:** * *Generator:* Mel-spectrogram Reconstruction Loss (L1), KL-Divergence Loss (for duration alignment), Adversarial Loss, and Feature Matching Loss.
    * *Discriminator:* Adversarial (Real vs. Fake) Loss.
* **Metrics:**
    * **MCD (Mel-Cepstral Distortion):** Measures the objective acoustic distance between synthesized and ground-truth audio.
    * **MOS (Mean Opinion Score):** Human evaluation metric for naturalness and dialect accuracy.

---

## 10. Conclusion & Next Steps
This milestone completely validates the structural integrity of the project. We have successfully bypassed Hugging Face dataset bottlenecks, resolved phonemizer incompatibilities, and verified that data flows perfectly from raw standard Hindi text all the way to a Haryanvi `.wav` file. 

**Next Steps for Milestone 4:**
1.  Initiate full-scale training on Module B utilizing all 4,000 samples.
2.  Implement strategic layer freezing for the VITS Text Encoder to preserve the Hindi base while fine-tuning the duration predictor.
3.  Calculate quantitative metrics (BLEU/chrF++) for the text module.

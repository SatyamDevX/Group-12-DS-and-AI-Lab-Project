# Milestone 3 Report: Model Architecture & End-to-End Pipeline Verification

**Project:** Regional Dialect Synthesis Pipeline (Standard Hindi to Haryanvi/Bangru)  
**Team Members:** Abhishek, Satyam, Sanket, Fazlur  
**Notebooks:** `text-to-text-pipe.ipynb`, `tts-pipe.ipynb`  
**Deadline:** 19 March 2026  

---

## Team Contributions

| Team Member | Contributions |
| :--- | :--- |
| Abhishek | • Designed Hindi → Haryanvi translation pipeline using LLaMA 3.1 + QLoRA <br> • Implemented instruction-based chat template formatting and tokenizer integration <br> • Ran inference on subset data and validated dialect outputs <br>  |
| Satyam | • Created Milestone 3 report (model architecture + pipeline explanation) <br> • Integrated outputs from text and audio modules <br> • Added model output examples for verification <br> • Updated worklog and managed final submission |
| Sanket | • Verified dataset consistency with model input requirements <br> • Uploaded processed datasets to Hugging Face <br> • Assisted in pipeline testing and validation <br> • Prepared initial PPT structure for milestone presentation |
| Fazlur | • Designed TTS architecture (VITS-based) <br> • Defined audio preprocessing pipeline (resampling, filtering, LJSpeech formatting) <br> • Structured dataset into TTS-ready format <br>  |

---

## 1. Overview


This milestone outlines the design, justification, and verification of our dual-architecture machine learning pipeline. The project aims to cross the low-resource dialect barrier by breaking the task into two sequential domains:

- **Module A (Text-to-Text):** Translating Standard Hindi to the Haryanvi (Bangru) dialect using a fine-tuned LLaMA 3.1 model.
- **Module B (Text-to-Speech):** Synthesizing the translated Haryanvi text into natural, dialect-accurate speech using VITS.

This report verifies the complete end-to-end pipeline by running a subset of the data to confirm that preprocessing, tokenization, forward passes, and output generations function cohesively without dimension mismatches or architectural failures.

---

## 2. Dataset Organization

The datasets are strictly isolated into distinct environments to prevent data leakage, utilizing dedicated Hugging Face repositories for both domains.

- **Text Dataset:** `Satyam-Srivastava/RDS` — Parallel Hindi-Haryanvi corpus (~5,000 pairs).
- **Audio Dataset:** `ankitdhiman/haryanvi-tts` — ~4,000 single-speaker Haryanvi text-audio pairs.

The HuggingFace token is securely stored in Colab Secrets and accessed via `userdata.get('HF_TOKEN')` to avoid hardcoding credentials.

### Directory Structure

```text
project_root/
│
├── hf_dataset_cache/               ← Full dataset downloaded via snapshot_download()
│   ├── metadata.csv
│   └── train/
│       └── *.wav
│
├── haryanvi_vits_data/             ← Processed TTS data (LJSpeech format)
│   ├── wavs/
│   │   ├── audio_0.wav
│   │   ├── audio_1.wav
│   │   └── ...
│   ├── metadata.csv                ← LJSpeech format: file|text|text
│   └── phoneme_cache/
│
├── tts_output/
│   └── vits_haryanvi-<timestamp>/
│       ├── best_model.pth
│       └── config.json
│
└── notebooks/
    ├── text-to-text-pipe.ipynb
    └── tts-pipe.ipynb
```

> **Note:** Text module train/val/test splits are held in-memory as a HuggingFace `DatasetDict` object and are not persisted to disk as `.json` files.

### Data Splits

| Module | Split | Size | Proportion |
| :--- | :--- | :--- | :--- |
| Text (Module A) | Train | ~4,011 | 80% |
| Text (Module A) | Validation | ~501 | 10% |
| Text (Module A) | Test | ~502 | 10% |
| Audio (Module B) | Train | ~90% of filtered set | 90% |
| Audio (Module B) | Eval | capped at 50 samples | 10% |

---

## 3. Preprocessing Pipeline

### 3.1 Module A: Text Preprocessing & Tokenization

Before passing text to the LLM, the data undergoes strict formatting to map to a causal language modeling task:

1. **Data Ingestion:** Two sources are merged — a JSON file (`hindiharyanvidataset5000.json`) and parallel `.txt` files (`hindi1.txt`, `haryanvi1.txt`) — yielding 5,025 initial pairs.
2. **Deduplication:** `drop_duplicates()` removes 8 repeated sentence pairs, reducing the dataset to 5,017 pairs.
3. **Length Filtering:** Sentences outside the 3–25 word boundary are dropped to prevent padding inefficiencies and out-of-memory errors.
4. **Train/Val/Test Split:** Stratified 80/10/10 split using `random_state=42` for reproducibility.
5. **Chat Template Formatting:** Each pair is mapped to the LLaMA 3.1 instruction format:
   - *System:* `"You are a precise Hindi to Haryanvi (Bangru) translator..."`
   - *User:* `"तुम कहाँ जा रहे हो?"` (Hindi)
   - *Assistant:* `"तू कड़े जा रया सै?"` (Haryanvi)
6. **Tokenization:** The `Meta-Llama-3.1-8B-Instruct` tokenizer is applied with:
   - `max_length = 128`
   - `padding = 'max_length'` using the EOS token as the pad token
   - `truncation = True`
   - Output columns: `input_ids`, `attention_mask` (raw text columns dropped)

### 3.2 Module B: Audio Preprocessing

Audio processing prepares raw `.wav` files for the VITS acoustic model:

1. **Deduplication:** `drop_duplicates(subset="text")` removes repeated transcripts to prevent the model from overfitting to repeated utterances on a small dataset.
2. **Word-count Filtering:** Transcripts outside the 3–25 word range are dropped. Sentences that are too short produce training artifacts; sentences that are too long cause VRAM spikes during mel-spectrogram computation. Both steps are run before any audio I/O to avoid wasting compute.
3. **Full Dataset Download:** The complete repository is downloaded locally via `snapshot_download()`. Each audio file is then loaded and resampled using `librosa.load(audio_abs_path, sr=22050, mono=True)`.
4. **Resampling:** All audio is standardized to **22050 Hz** (required sample rate for VITS).
5. **Metadata Alignment:** Processed files are exported to the standard `LJSpeech` format: `filename|transcript|normalized_transcript`.
6. **Character-Level Text Input:** VITS is configured with `use_phonemes=False` and `text_cleaner="basic_cleaners"`. This is intentional — `espeak-ng` has no native Haryanvi support and its Hindi (`hi`) phoneme mapping mispronounces dialect-specific retroflex sounds unique to Bangru. Character-level input preserves the original Devanagari script directly, allowing the model to learn acoustic mappings from raw characters.
7. **Spectrogram Configuration:** Applied by the `AudioProcessor` at training time — FFT size: 1024, hop length: 256, 80 mel bins, silence trimming at 45 dB threshold.

---

## 4. Model Architecture & Data Flow

### 4.1 Module A: LLaMA 3.1 + QLoRA (Text Translation)

- **Base Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Quantization:** 4-bit NormalFloat (NF4) via `bitsandbytes`, enabling A100 VRAM compliance.
- **Adaptation:** Low-Rank Adaptation (LoRA) targeting all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) to learn morpho-syntactic dialect shifts without catastrophic forgetting of base Hindi knowledge.

### 4.2 Module B: VITS (Text-to-Speech)

- **Architecture:** Variational Inference with adversarial learning for end-to-end Text-to-Speech synthesis.
- **Current Status:** VITS is initialized from scratch for pipeline verification in this milestone.
- **Planned (Milestone 4):** To overcome the 4,000-sample limitation, the final pipeline will initialize from pre-trained Hindi checkpoints (e.g., AI4Bharat `vits_rasa_13`) via transfer learning, freezing the Text Encoder and fine-tuning only the Duration Predictor and Decoder on Haryanvi data.
- **Components:**
  - *Text Encoder:* Maps character IDs to hidden representations.
  - *Stochastic Duration Predictor:* Captures the variable syllabic rhythm of Bangru.
  - *Posterior Encoder + Normalizing Flows:* Maps linear spectrograms to a complex latent space.
  - *Vocoder:* Integrated HiFi-GAN for direct waveform synthesis.

### 4.3 Data Flow Diagram

```text
╔══════════════════════════════════════════════════════════════╗
║              RAW INPUT: Standard Hindi Text                  ║
║           "मैं आज बाजार जा रहा हूँ।"                        ║
╚══════════════════════════════════════════════════════════════╝
                           │
                           ▼
              ┌────────────────────────┐
              │   Length Filtering     │  (3–25 words)
              │   + Deduplication      │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Chat Template Format  │  system / user / assistant
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  LLaMA 3.1 Tokenizer   │  BPE Subword Tokenization
              │  max_length = 128      │  → input_ids, attention_mask
              └────────────────────────┘
                           │
                           ▼
       ╔═══════════════════════════════════════╗
       ║  MODULE A: LLaMA 3.1 8B + QLoRA      ║
       ║  (4-bit NF4 Quantization)             ║
       ║  LoRA: rank=16, alpha=32              ║
       ║  Loss: Causal Cross-Entropy           ║
       ╚═══════════════════════════════════════╝
                           │
                           ▼
              ┌────────────────────────┐
              │  Greedy / Beam Decode  │  token IDs → UTF-8 string
              └────────────────────────┘
                           │
                           ▼
╔══════════════════════════════════════════════════════════════╗
║            HARYANVI TEXT OUTPUT                              ║
║           "मैं आज बाजार जा रया सै।"                         ║
╚══════════════════════════════════════════════════════════════╝
                           │
                           ▼
              ┌────────────────────────┐
              │  basic_cleaners        │  Devanagari normalization
              │  + Character Tokenizer │  char → integer IDs
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  VITS Text Encoder     │  char_ids → hidden states
              │                        │  Shape: (B, hidden, T_text)
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Stochastic Duration   │  predicts frame alignment
              │  Predictor             │  Shape: (B, 1, T_text)
              └────────────────────────┘
                           │
                           ▼
       ╔═══════════════════════════════════════╗
       ║  MODULE B: VITS Decoder               ║
       ║  Normalizing Flows + HiFi-GAN         ║
       ║  Mel bins: 80 | FFT: 1024             ║
       ║  Loss: L1 + KL + Adversarial          ║
       ╚═══════════════════════════════════════╝
                           │
                           ▼
╔══════════════════════════════════════════════════════════════╗
║         RAW AUDIO WAVEFORM OUTPUT                            ║
║         test_haryanvi_output.wav  @ 22050 Hz                 ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 5. Input-Output Compatibility & Tensor Shapes

| Component | Expected Input Format | Shape / Dimensions | Output |
| :--- | :--- | :--- | :--- |
| LLaMA Tokenizer | Raw UTF-8 Hindi string | — | `input_ids (batch, ≤128)` |
| LLaMA Base Model | `input_ids` + `attention_mask` | `(batch=4, seq≤128)` | Logits `(batch, seq, vocab_size)` |
| LoRA Adapters | Hidden states | `rank=16, alpha=32` | Delta hidden states |
| VITS Text Encoder | Character IDs (`basic_cleaners`) | `(batch=8, text_len)` | `(batch, hidden_dim, text_len)` |
| VITS Duration Predictor | Encoded hidden states | `(batch, hidden_dim, text_len)` | Duration per character |
| VITS Posterior Encoder | Linear spectrogram | `(batch, n_fft//2+1, mel_len)` | Latent `z` |
| HiFi-GAN Vocoder | Mel-spectrogram | `(batch, 80, mel_len)` | Waveform `(batch, 1, audio_len)` |

**Integration Note:** Module A outputs a pure Devanagari UTF-8 string. Module B's `basic_cleaners` accepts this string directly without any intermediate phonemization step, ensuring a clean handoff between modules.

---

## 6. Architecture Justification & Alternatives

### Text Module: Why LLaMA 3.1 QLoRA?

Haryanvi lacks the millions of parallel sentences required to train a Seq2Seq model (like MarianMT or T5) from scratch. Standard translation models fail on dialects because they treat them as entirely new languages. LLaMA 3.1 already possesses deep semantic understanding of standard Hindi — QLoRA simply steers that existing knowledge toward Bangru syntax and morphology without requiring large compute or catastrophic forgetting.

### Audio Module: Why VITS?

| Model | Duration Prediction | Inference | Low-Resource Suitability |
| :--- | :--- | :--- | :--- |
| **VITS** | Stochastic (flow-based) | Single forward pass | ✅ High |
| FastSpeech2 | Deterministic (MSE) | Two-stage | ⚠️ Robotic cadence |
| XTTS / VALL-E | Autoregressive | Slow, hallucination-prone | ❌ Needs large data |

VITS's stochastic duration predictor captures the aggressive syllabic compression and flatter melodic variance characteristic of Bangru. FastSpeech2's deterministic MSE duration produces a metronomic cadence that breaks dialect authenticity. XTTS suffers from high inference latency and hallucinations without large-scale dialect data.

---

## 7. End-to-End Pipeline Verification

To verify component integration, we executed a tracer bullet run on a restricted dataset subset.

### Module A Verification (`text-to-text-pipe.ipynb`)

- Executed `SFTTrainer` on the filtered dataset with `batch_size=4`.
- Loss decreased from `0.094` → `0.090` over 3 epochs, confirming gradient flow through LoRA layers.
- LoRA adapters saved successfully to `./llama-3.1-haryanvi-adapter-final`.

### Module B Verification (`tts-pipe.ipynb`)

- The full audio dataset was downloaded via `snapshot_download()`. After deduplication and word-count filtering (3–25 words), all valid samples were exported to LJSpeech format.
- The eval split was capped at `eval_split_max_size=50`; the full filtered set was used for training.
- Executed the VITS `Trainer` loop for 1 epoch.
- **Result:** The GAN training loop stabilized, `basic_cleaners` correctly processed Devanagari input, and a checkpoint was saved without dimension mismatch errors.
- Checkpoint discovery uses dynamic `.pth` lookup (preferring `best_model.pth`, falling back to latest by modified time) to avoid hardcoded filename errors.

---

## 8. Model Outputs & Examples

### Translation Output (Actual Model Results)

| Input (Standard Hindi) | Model Output (Haryanvi) | Notes |
| :--- | :--- | :--- |
| `मैं आज बाजार जा रहा हूँ।` | `मैं आज बाजार जा रया सै।` | Correct verb morphology |
| `यह किताब मेरी है, इसे मत छूना।` | `या किताब मेरी सै, इसे न छूणा।` | Excellent dialect substitution |
| `क्या तुम कल स्कूल जाओगे?` | `क्या तुम कल स्कूल जाओगे।` | Interrogative marker lost — needs tuning |

### TTS Output

- Input fed: `"मैं आज बाजार जा रया सै।"`
- **Output:** `test_haryanvi_output.wav` @ 22050 Hz
- *Status:* Waveform successfully generated. As expected from a 1-epoch run, perceptual fidelity is low, but the result confirms correct architectural math and pipeline bridging between Module A and Module B.

---

## 9. Loss Functions & Evaluation Metrics

### Module A (Text Translation)

- **Loss Function:** Causal Language Modeling Loss (Cross-Entropy over next-token prediction).
- **Evaluation Metrics:**
  - **chrF++:** Character n-gram F-score — optimal for morphologically rich dialects like Haryanvi where lexical substitutions occur at the sub-word level.
  - **BLEU Score:** Measures exact lexical overlap for benchmarking against baseline translation models.

### Module B (Text-to-Speech)

- **Loss Functions:**
  - *Generator:* Mel-spectrogram Reconstruction Loss (L1), KL-Divergence Loss (latent alignment), Adversarial Loss, Feature Matching Loss.
  - *Discriminator:* Adversarial (Real vs. Fake) Loss.
- **Evaluation Metrics:**
  - **UTMOS (Unified TTS MOS Predictor):** A reference-free neural MOS estimator that scores naturalness without requiring ground-truth recordings — appropriate here since the pipeline generates novel Haryanvi speech with no reference audio available.
  - **MOS (Mean Opinion Score):** Human evaluation for naturalness and dialect authenticity, planned for Milestone 4 after full-scale training.
  - *Note: MCD (Mel-Cepstral Distortion) will be computed only in Milestone 4 once ground-truth re-recordings of the test sentences are collected.*

---

## 10. Conclusion & Next Steps

This milestone completely validates the structural integrity of the dual-module pipeline. We have verified that data flows correctly from raw Standard Hindi text through Module A translation and into Module B speech synthesis, producing a Haryanvi `.wav` output without integration errors.

**Next Steps for Milestone 4:**

1. Initiate full-scale training on Module B using all ~4,000 filtered samples.
2. Implement transfer learning for VITS by initializing from the AI4Bharat `vits_rasa_13` Hindi checkpoint, freezing the Text Encoder, and fine-tuning the Duration Predictor and Decoder on Haryanvi data.
3. Calculate quantitative metrics — BLEU/chrF++ for Module A and UTMOS for Module B.
4. Collect human MOS evaluations for dialect authenticity scoring.
```

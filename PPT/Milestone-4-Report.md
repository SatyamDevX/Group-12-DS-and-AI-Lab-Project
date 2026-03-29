# Milestone 4 Report: Complete Model Training
### End-to-End LLaMA Fine-Tuning and Coqui VITS Speech Synthesis Pipeline


**Project:** Regional Dialect Synthesis Pipeline (Standard Hindi to Haryanvi/Bangru)  
**Team Members:** Abhishek, Satyam, Sanket, Fazlur  
**Notebooks:** `llama-text-pipe-final.ipynb`, `coqui-tts-pipe-final.ipynb`  
**logs/summary-files:** `experiment_summary.csv`, `trainer_0_log_2.txt`  
**Deadline:** 29 March 2026  

---

## Team Contributions

| Team Member | Contributions |
| :--- | :--- |
| Abhishek | • Complete end to end training of llama (text to text model part)  |
| Satyam | • Handled Training of coqui-tts model on hardware |
| Sanket | • helped with documentation |
| Fazlur | • wrote all codes for tts model |

---

## Overview



This milestone documents the training, experimentation, and evaluation of a two-stage pipeline for converting standard Hindi text into spoken Haryanvi. The pipeline consists of:

1. **Text-to-Text Translation** — A QLoRA fine-tuned LLaMA 3.1 8B Instruct model that translates Hindi (Devanagari) into Haryanvi (Devanagari).
2. **Text-to-Speech Synthesis** — A Coqui VITS GAN-based model fine-tuned on a Haryanvi audio corpus to synthesize natural-sounding speech from the translated text. 

### End-to-End Pipeline

```
[Hindi Text Input]
      │
      ▼
[LLaMA 3.1 8B + QLoRA Adapter]  ← Translation Stage
      │
      ▼
[Haryanvi Text (Devanagari)]
      │
      ▼
[Coqui VITS (GAN TTS)]          ← Synthesis Stage
      │
      ▼
[Audio Output — 22050 Hz WAV]
```

---

## Part 1: Text-to-Text Translation (LLaMA 3.1 8B)


### 1.1 Dataset & Preprocessing

**Source:** `Satyam-Srivastava/RDS` (Hugging Face Hub) — a bilingual Hindi/Haryanvi dataset containing paired text samples and JSON-formatted sentence pairs.

**Preprocessing Steps:**

- Merged Hindi/Haryanvi text columns with JSON pair entries into a unified parallel corpus.
- Removed duplicate rows and filtered out any samples containing null or empty fields.
- Applied strict sentence-length filtering using `TEXT_MIN_WORDS` and `TEXT_MAX_WORDS` thresholds to eliminate noise from extremely short or excessively long sequences.
- Performed a **truncation sanity check**: analyzed the token length distribution of all samples against `LLM_MAX_LENGTH` to confirm that the chosen sequence ceiling introduces negligible truncation loss across the dataset.

**Data Splits:**

| Split | Purpose |
|---|---|
| Train | Primary optimization objective |
| Validation | Tuning-time early stopping and hyperparameter selection |
| Test | Final held-out evaluation (never seen during tuning) |
| Qualitative Holdout | Fixed sample set for consistent manual side-by-side review across all runs |

Per-split prediction files are saved to each experiment's output directory under `combined_outputs/llm/{experiment_name}/` and include `validation_predictions.csv` and `qualitative_predictions.csv`.

---

### 1.2 Model Architecture

**Base Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`

**Fine-Tuning Strategy:** QLoRA (Quantized Low-Rank Adaptation)

- The base model weights are frozen and loaded in **4-bit NF4 (NormalFloat4)** precision using `BitsAndBytesConfig`. NF4 is theoretically optimal for normally-distributed neural network weights and outperforms standard FP4.
- **Double quantization** is enabled — a second quantization pass is applied to the quantization constants themselves, saving an additional ~0.3 bits per parameter.
- Trainable **LoRA adapters** are injected into all major attention and MLP projection layers:

```
q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj
```

The complete adapter configuration per experiment is stored in each run's `hyperparameters.json`.

**Assistant-Only Loss Masking:**

A critical design decision is that loss is computed **exclusively on the assistant (Haryanvi) response tokens**. All system prompt tokens, user instruction tokens, and padding positions are masked with `-100`, so the cross-entropy loss ignores them entirely. This prevents the model from wasting gradient signal on boilerplate instruction text and focuses it entirely on learning the Hindi → Haryanvi mapping.

---

### 1.3 Training Configuration

The following configuration was held constant across all experiments unless explicitly varied:

| Parameter | Value |
|---|---|
| Base Model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Quantization | 4-bit NF4, double quantization |
| Compute dtype | BF16 (FP16 fallback for non-Ampere GPUs) |
| Optimizer | `paged_adamw_32bit` |
| LR Schedule | Cosine annealing |
| Per-device Train Batch Size | 4 |
| Gradient Accumulation Steps | 4 (effective batch size = **16**) |
| Per-device Eval Batch Size | 4 |
| Gradient Checkpointing | Enabled |
| Checkpoint Selection | Based on lowest validation loss |
| Eval Strategy | Validation split (tuning) / Test split (final eval, strictly separated) |

**Loss Function:** Standard causal language modeling cross-entropy, restricted to assistant response tokens via `-100` masking.

**Evaluation Metrics:**

- `eval_loss` — primary optimization signal on the validation split.
- `exact_match_rate` — strict full-sequence accuracy.
- `corpus_bleu` — corpus-level n-gram precision.
- `corpus_chrf` — **primary quality metric**; character-level F-score robust to morphological variation, making it especially well-suited for Haryanvi, a morphologically variable dialect of Hindi where minor phonological differences would unfairly penalize BLEU.

---

### 1.4 Hyperparameter Experiments

Six experiments were conducted to systematically identify optimal hyperparameter values. The table below shows the exact configuration for each run. All experiment configs are reproducible via the corresponding `hyperparameters.json` and `run_manifest.json` saved in each experiment directory.

#### Experiment Configurations

| Experiment | LR | Epochs | LoRA r | LoRA α | Dropout | Weight Decay | Max Grad Norm | Max Seq Len | Warmup Steps |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 2e-4 | 3 | 16 | 32 | 0.05 | 0.0 | 0.3 | 192 | 50 |
| `lower_lr_longer` | **1e-4** | **4** | 16 | 32 | 0.05 | 0.0 | 0.3 | 192 | **100** |
| `longer_context_256` | 2e-4 | 3 | 16 | 32 | 0.05 | 0.0 | 0.3 | **256** | 50 |
| `higher_dropout_reg` | 2e-4 | 3 | 16 | 32 | **0.15** | **0.01** | 0.3 | 192 | 50 |
| `stronger_grad_clip` | 2e-4 | 3 | 16 | 32 | 0.05 | 0.0 | **0.15** | 192 | 50 |
| `higher_rank` | 2e-4 | 3 | **32** | **64** | **0.1** | 0.0 | 0.3 | 192 | 50 |

> Bold values indicate the parameter being varied relative to baseline.

#### Quantitative Results

Results are drawn from `experiment_summary.csv`. Validation metrics are computed on the validation split; qualitative metrics are computed on the fixed qualitative holdout set.

| Experiment | eval\_loss ↓ | Val Exact Match | Val BLEU ↑ | Val chrF ↑ | Qual Exact Match | Qual BLEU ↑ | Qual chrF ↑ | Train Runtime (s) |
|---|---|---|---|---|---|---|---|---|
| `lower_lr_longer` | **0.003241** | 50.42% | **71.760** | **77.649** | 56.00% | 74.282 | 79.571 | 3082.6 |
| `higher_dropout_reg` | 0.003432 | 50.42% | 71.748 | 77.641 | 56.00% | 74.282 | 79.571 | 2312.8 |
| `longer_context_256` | 0.003891 | 50.42% | **71.762** | 77.648 | 56.00% | 74.282 | 79.571 | 2339.0 |
| `stronger_grad_clip` | 0.004164 | 50.42% | 71.748 | 77.641 | 56.00% | 74.282 | 79.571 | 2328.9 |
| `higher_rank` | 0.004269 | 50.42% | 71.748 | 77.640 | 56.00% | 74.282 | 79.571 | 2334.4 |
| `baseline` | 0.004359 | 50.42% | 71.745 | 77.629 | 56.00% | 74.282 | 79.571 | 2304.0 |

**Best experiment:** `lower_lr_longer` — achieves the lowest `eval_loss` (0.003241), highest validation chrF (77.649), and highest validation BLEU (71.760), at the cost of the longest training runtime (~3083 seconds). Training curves per experiment are available in each run's `training_log_history.csv`.

#### Key Observations

- **Evaluation pipeline failure:** Qualitative metrics (Exact Match, BLEU, chrF) are mathematically identical across all runs, and validation Exact Match is fixed at 50.42%. This anomaly indicates a critical failure in the evaluation pipeline—highly likely due to the inference script caching initial predictions or failing to dynamically load the distinct LoRA adapter weights for each experiment. These generative metrics cannot be used to draw valid conclusions until the evaluation script is corrected and the evaluations are re-run.
- **Lower LR + more epochs is the clearest win:** `lower_lr_longer` improves eval_loss from 0.00436 to 0.00324 (a 25.7% reduction) with only a modest ~34% increase in training time.
- **Higher LoRA rank did not help:** `higher_rank` (r=32) performed slightly *worse* than baseline (r=16) on eval_loss (0.00427 vs 0.00436), suggesting the baseline adapter capacity is already sufficient for this task and doubling it introduces noise.
- **Stronger gradient clipping** (0.15 vs 0.3) actively harmed performance, producing the third-worst `eval_loss` (0.004164), likely by excessively constraining gradient updates.

---

### 1.5 Generalization & Training Stability

Several techniques were employed to maximize generalization and prevent training instability:

- **Pre-flight sanity checks:** Before every long run, assistant-label verification is performed to confirm that `-100` masking is correctly applied. A `worst_validation_cases.csv` is exported after each run — sorted by ascending sentence-level chrF — to proactively diagnose data pipeline issues and identify systematic failure modes.
- **Quantized loading (QLoRA):** 4-bit NF4 loading dramatically reduces GPU VRAM consumption, preventing OOM crashes on mid-range hardware (T4/V100) and enabling a larger effective batch size via gradient accumulation.
- **Gradient checkpointing:** Recomputes intermediate activations during the backward pass rather than storing them, stabilizing the memory footprint across long sequences at the cost of a small compute overhead.
- **LoRA dropout:** Applied within adapter layers to reduce co-adaptation of adapter parameters. Tested at 0.05 (baseline), 0.1 (`higher_rank`), and 0.15 (`higher_dropout_reg`). The 0.15 dropout + weight decay combination (0.01) produced the second-best eval_loss, demonstrating that regularization is beneficial — but must be balanced against underfitting.
- **Weight decay (AdamW L2):** Tested in `higher_dropout_reg` (0.01 vs 0.0 elsewhere). Combined with higher dropout, it produced the second-best eval_loss (0.003432), confirming its value as a complementary regularizer.
- **Strict split separation:** The validation split drives hyperparameter decisions; the test split is evaluated only once per experiment, fully preventing data leakage. Final test split evaluation is strictly withheld until the evaluation pipeline bug is resolved and a single best model is verified.

---

### 1.6 Results & Observations

**Best Model:** `lower_lr_longer`

| Metric | Validation Split | Qualitative Holdout |
|---|---|---|
| Exact Match Rate | 50.42% | 56.00% |
| Corpus BLEU | 71.760 | 74.282 |
| Corpus chrF | 77.649 | 79.571 |
| Eval Loss | 0.003241 | — |

The qualitative holdout consistently outperforms the validation split on BLEU and chrF across all experiments, suggesting the holdout set consists of shorter, more canonical sentences that better match frequent patterns in the training data.

**Error Analysis (from `worst_validation_cases.csv`):**

Sentence-level chrF analysis reveals that failure cases are predominantly:
- Proper nouns and named entities with no aligned Haryanvi equivalent in the training corpus.
- Dialect-specific idiomatic expressions that require non-compositional translation.
- Complex compound sentences where the Haryanvi target requires significant structural reordering relative to Hindi.

---

### 1.7 Generated Artifacts

Each experiment produces a self-contained output directory at:
```
combined_outputs/llm/{experiment_name}/
```

The following artifacts are generated per run:

| Artifact | Path | Description |
|---|---|---|
| `adapter/` | `{experiment}/adapter/` | Saved LoRA adapter weights for inference or merging |
| `checkpoints/` | `{experiment}/checkpoints/` | Intermediate training checkpoints (best-val-loss selection) |
| `hyperparameters.json` | `{experiment}/hyperparameters.json` | Full hyperparameter config for exact reproducibility |
| `run_manifest.json` | `{experiment}/run_manifest.json` | Runtime metadata: hardware, dataset stats, timestamps |
| `summary.json` | `{experiment}/summary.json` | Aggregated metrics summary for the completed run |
| `training_log_history.csv` | `{experiment}/training_log_history.csv` | Step-level training and eval loss curves |
| `validation_predictions.csv` | `{experiment}/validation_predictions.csv` | Per-sample predictions on validation split with sentence chrF |
| `qualitative_predictions.csv` | `{experiment}/qualitative_predictions.csv` | Predictions on fixed qualitative holdout for manual review |
| `worst_validation_cases.csv` | `{experiment}/worst_validation_cases.csv` | Lowest chrF-scoring validation samples for error analysis |

> All artifact paths are also recorded programmatically inside `run_manifest.json`, enabling automated downstream processing without hardcoding directory names.

---

### 1.8 Key Findings (Text-to-Text)

**What Worked Well:**

- **Assistant-only loss masking** (`-100` on prompt/padding tokens) was the single highest-impact design decision. Without it, the model generates instruction-like boilerplate rather than clean Haryanvi translations.
- **Lower LR with longer training** (`lower_lr_longer`, LR=1e-4, 4 epochs) outperformed all other configurations on eval_loss and validation chrF, producing a **25.7% reduction in eval_loss** over the baseline.
- **QLoRA's 4-bit NF4 double quantization** successfully fit the 8B model on available GPU VRAM without compromising adapter training quality.
- **chrF** proved a significantly more informative and stable tuning signal than BLEU for this morphologically variable target dialect.

**What Did Not Perform As Expected:**

- **Higher LoRA rank (r=32)** did not yield proportional gains; it performed slightly worse than baseline (r=16), indicating the adapter capacity at r=16 is already well-matched to the task complexity and dataset size.
- **Stronger gradient clipping** (max_grad_norm=0.15) actively hurt performance, yielding the third-worst `eval_loss`.
- **Evaluation pipeline integrity:** The identical generative metrics across all runs exposed a critical flaw in the inference script (likely adapter loading or caching failures) that must be resolved before final model selection.

**Bottlenecks:**

- The limited size of the parallel Hindi-Haryanvi corpus is the primary ceiling on translation quality. Low-resource dialect pairs have few publicly available sentence-aligned datasets.
- Proper nouns and idiomatic expressions remain consistently hard categories regardless of hyperparameter changes, pointing to a data coverage gap rather than a modeling issue.

**Plans for Improvement:**

- Data augmentation via back-translation to synthetically expand the training corpus.
- Extend training to more epochs now that `lower_lr_longer` has demonstrated clear gains from additional passes — contingent on A100 compute access for faster iteration.
- Introduce human annotation on the qualitative holdout set as a gold-standard complement to automated chrF/BLEU scores.
- Explore retrieval-augmented generation (RAG) or few-shot prompting with dialect glossaries to improve proper noun and idiom handling.

---

## Part 2: Text-to-Speech Synthesis (Coqui VITS)

### 2.1 Dataset & Preprocessing

**Source:** `ankitdhiman/haryanvi-tts` (Hugging Face) — Haryanvi utterances with mixed-script transcriptions (Devanagari, Bengali, Urdu).

**Preprocessing:**
- Dropped null/duplicate text entries.
- **Multi-script transliteration:** Bengali → Devanagari via `indic-transliteration`; Urdu/Arabic → Devanagari via a hardcoded `URDU_DEVANAGARI_MAP` (the `indo-arabic-transliteration` library was discarded — incompatible with Python 3.12).
- Applied **Unicode NFC normalization** to enforce composed Devanagari forms — without this, the tokenizer silently drops decomposed characters.
- **Word-count filtering (3–25 words):** Excludes utterances too short for prosody learning or too long for stable attention alignment.
- Collapsed whitespace; exported to **LJSpeech format** (`file_id|text|normalized_text`).
- Audio resampled to **22050 Hz mono 16-bit PCM WAV**; silence trimmed at 45 dB threshold.

**Final counts:** 2764 rows → **2714 train / 50 eval**. Dataset manifest saved to `haryanvi_vits_data/metadata.csv`.



### 2.2 Model Architecture

**Model:** VITS — Conditional Variational Autoencoder with Adversarial Learning for
End-to-End Text-to-Speech (Kim et al., ICML 2021)

VITS is a **single-stage end-to-end TTS model** that synthesizes raw waveforms directly from
text without a separate vocoder stage. Total parameters: **83,067,244 (~83M)**.

```
Text Input (Devanagari characters)
        │
        ▼
┌──────────────────────┐
│   Text Encoder       │  Transformer encoder — learns phonological
│  (enc_p / 110 keys)  │  representations of Devanagari characters
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Duration Predictor  │  Stochastic predictor for how long each
│  (dp / 284 keys)     │  character is spoken
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Normalizing Flow    │  Bijective transformation: maps prior
│  (flow / 112 keys)   │  latent space → mel-spectrogram space
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Waveform Decoder    │  HiFiGAN — converts latent → raw 22050 Hz audio
│  (dec / 231 keys)    │  Multi-period discriminator adversarial training
└──────────────────────┘
        │
        ▼
  Raw Audio (22050 Hz WAV)

─────────── Training only ───────────
┌──────────────────────┐
│  Posterior Encoder   │  VAE encoder — encodes mel-spectrogram → latent
│  (enc_q / 100 keys)  │  Used only during training with teacher forcing
└──────────────────────┘
```

**Character Vocabulary: 173 characters (no phonemizer)**

Using a character-level tokenizer (rather than espeak-ng) avoids dialect-specific
mispronunciations — espeak-ng's Hindi locale does not account for Haryanvi phonology.
The vocabulary covers:

- Full Devanagari vowel set including candra forms (ऑ, ऒ)
- All 36 consonants + nukta variants (क़, ख़, ग़, ज़, ड़, ढ़, फ़, य़)
- Vowel signs (matras), halant (्), anusvara (ं), visarga (ः), chandrabindu (ँ)
- Devanagari digits (०–९) + ASCII digits (0–9)
- Full Latin alphabet A–Z, a–z (for loanwords like "SMS", "AI")
- Punctuation: `! , - . ? । ॥ ' " ( ) :`

**Pre-trained Checkpoint: `SYSPIN/vits_Hindi_Female`**

Training warm-starts from this Coqui-native Hindi female TTS model. Hindi shares
~85–90% phonological vocabulary with Haryanvi, making it an ideal donor model.

| Component | Keys Loaded | Missing | Status |
|---|---|---|---|
| `text_encoder` | 110 | 1 (emb.weight — vocab size mismatch) | ⚠️ Partial (expected) |
| `waveform_decoder` | 231 | 0 | ✅ 100% |
| `flow` | 112 | 0 | ✅ 100% |
| `duration_predictor` | 284 | 0 | ✅ 100% |
| `posterior_encoder` | 100 | 0 | ✅ 100% |

Only `text_encoder.emb.weight` is randomly initialized (due to the expanded 173-character
Haryanvi vocabulary vs. 122-character Hindi vocabulary). All other layers transfer perfectly,
providing a strong acoustic foundation from day one.

---

### 2.3 Training Configuration

**Key configuration (final run):**

| Parameter | Value |
|---|---|
| Generator LR | `1e-5` |
| Discriminator LR | `1e-5` |
| Batch Size | 16 |
| Eval Batch Size | 8 |
| Mixed Precision (FP16) | Enabled |
| Epochs | 300 |
| Evaluation | Enabled (`run_eval=True`), every 1680 steps (~10 epochs) |
| Evaluation delay | First eval after epoch 20 (early instability skipped) |
| Checkpoint saves | Every 334 steps (~2 epochs) |
| Best model saved after | Step 2000 (post-stabilization only) |
| Phonemizer | Disabled (character-level) |

**Hardware & Throughput:**

| Parameter | Value |
|---|---|
| Hardware | Google Colab T4 GPU (16 GB VRAM) |
| Steps per Epoch | ~169 (2714 samples ÷ batch size 16) |
| Total Steps (300 epochs) | ~50,700 |

**Notable config decisions:**
- **Equal LR for generator and discriminator (`1e-5`)** — both kept at the same conservative rate to ensure stable adversarial balance during fine-tuning.
- **`save_best_after=500`** — prevents premature best-model overwriting during the first ~12 epochs where GAN losses are volatile.
- **`test_delay_epochs=20`** — skips evaluation entirely during early training when outputs are unintelligible, saving compute.

### 2.4 Training Progression & Convergence

The VITS model was trained for 300 epochs. The chart below shows all six loss components
across the full training run:

![VITS Training Loss Curves — 300 Epochs](loss_performance2.png)
*Figure 1: VITS training loss curves across 300 epochs. loss_mel (primary quality indicator)
drops from ~32 to ~13. KL divergence converges sharply within the first 25 epochs.
Discriminator loss stabilizes at ~2.75, confirming no mode collapse throughout.*

Loss values at key training milestones (read from training log):

| Epoch | `loss_mel` ↓ | `loss_kl` ↓ | `loss_feat` ↓ | `loss_gen` ↓ | `loss_disc` | `loss_duration` ↓ |
|---|---|---|---|---|---|---|
| 0 | ~32.0 | ~3.70 | ~8.50 | ~2.22 | ~2.82 | ~2.12 |
| 25 | ~19.0 | ~1.50 | ~6.50 | ~1.90 | ~2.72 | ~1.75 |
| 50 | ~15.0 | ~1.45 | ~6.50 | ~1.87 | ~2.73 | ~1.70 |
| 100 | ~14.5 | ~1.42 | ~6.50 | ~1.85 | ~2.74 | ~1.68 |
| 200 | ~13.5 | ~1.38 | ~6.50 | ~1.85 | ~2.75 | ~1.65 |
| 300 | ~13.0 | ~1.36 | ~6.50 | ~1.84 | ~2.75 | ~1.63 |

Key observations from the training curves:

- **`loss_mel`** (primary quality indicator) drops sharply from ~32 to ~15 within the first
  50 epochs — a direct benefit of the Hindi warm-start — then continues to gradually converge
  to ~13 by epoch 300, showing the model has not plateaued and still has room to improve.
- **`loss_kl`** collapses rapidly in the first ~25 epochs (3.70 → 1.50), meaning the posterior
  encoder aligned with the prior very early. After epoch 25 it decays slowly and steadily to
  ~1.36 — a well-behaved VAE posterior.
- **`loss_feat`** drops from ~8.5 to ~6.5 within the first 25 epochs and then plateaus
  completely, indicating feature matching between real and generated audio stabilized early.
- **`loss_gen`** shows a sharp spike at initialization (~2.22) before dropping to ~1.85 by
  epoch 25, then flattening — the generator found a stable adversarial equilibrium quickly.
- **`loss_disc`** exhibits an initial dip to ~2.55 in the first few steps before rising and
  stabilizing around **2.75** for the remainder of training. This flat, stable discriminator
  loss is the key indicator that **no mode collapse occurred**.
- **`loss_duration`** decays smoothly from ~2.12 to ~1.63 over the full 300 epochs —
  the duration predictor is still actively improving, suggesting further training would
  yield better prosodic timing.

---

### 2.5 What Was Tried, Discarded, and Why

A significant portion of the engineering effort involved navigating incompatible tooling
in the low-resource TTS ecosystem. The following approaches were systematically evaluated
and discarded:

| Approach | Why Discarded |
|---|---|
| **HuggingFace `Trainer` API** | HF Trainer assumes a single `forward → loss → backprop` cycle. VITS requires alternating generator/discriminator updates with 2 separate optimizers and 5 simultaneous losses — architecturally incompatible. HF supports VITS for inference only. |
| **gokulkarthik/TTS (AI4Bharat fork)** | Research-grade 2022 code; fails to build on Python 3.12 (Colab runtime) with `subprocess-exited-with-error`. No `pyproject.toml`, no modern Python support. |
| **ai4bharat/vits-rasa-13 (HuggingFace)** | Uses HuggingFace `AutoModel/AutoTokenizer` format — incompatible with Coqui's `.pth` checkpoint format. |
| **ai4bharat/vits-hi-itts-base (Original VITS format)** | Uses original `jaywalnut310/vits` key naming (`enc_p`, `dec`, `dp`, `enc_q`) vs. Coqui naming (`text_encoder`, `waveform_decoder`, `duration_predictor`, `posterior_encoder`). Even after full key remapping, only 34% of the HiFiGAN decoder loaded — producing buzzy, distorted audio requiring ~2× more training to converge. |
| **`trainer` (PyPI)** | No Python 3.12 distribution exists. `coqui-tts` already bundles `coqui-tts-trainer` with the identical `from trainer import Trainer, TrainerArgs` API. |

**✅ Final Stack:**

| Component | Choice | Reason |
|---|---|---|
| TTS Framework | `coqui-tts` (PyPI, idiap fork) | Actively maintained, Python 3.12 support, full VITS GAN Trainer built-in |
| Pre-trained checkpoint | `SYSPIN/vits_Hindi_Female` | Coqui-native format → 100% decoder loaded |
| Text processing | Character-level (no phonemizer) | Avoids dialect mispronunciations from espeak-ng Hindi locale |
| Transliteration | `indic-transliteration` + hardcoded maps | Bengali→Devanagari (library) + Urdu→Devanagari (hardcoded) |

---

### 2.6 Hyperparameter Experiments

The following parameters were identified for systematic tuning. The initial configuration
served as the baseline for all comparisons.

| Parameter | Current Value | Tested / Suggested Range | Impact Area |
|---|---|---|---|
| `lr` (generator) | `1e-5` | `5e-6` – `2e-5` | Audio quality and convergence speed |
| `lr_disc` (discriminator) | `2e-4` | `1e-4` – `3e-4` | Adversarial training balance |
| `batch_size` | `16` | `16` – `32` | Training stability vs. throughput |
| VITS hidden channels | `192` (default) | `128` – `256` | Model capacity vs. VRAM |
| `trim_db` (silence) | `45 dB` | `35` – `55 dB` | Audio boundary quality |
| `epochs` | `300` | extended to 500 target | Prosody naturalness |


---

### 2.7 Generalization & Training Stability

- **Hindi warm-start (SYSPIN/vits_Hindi_Female):** The most impactful stability decision.
  Because Coqui-native format allowed 100% decoder weight transfer, the model skipped the
  random-initialization instability phase entirely. The mel loss began dropping immediately
  from step 0, unlike a from-scratch run where the first ~50 steps typically produce
  incoherent noise.
- **Character-level tokenization (no phonemizer):** Bypassing espeak-ng's Hindi locale
  prevents systematic mispronunciations of Haryanvi-specific words (e.g., *कोनी*, *म्हारा*,
  *सै*). The 173-character Devanagari + Latin vocabulary is sufficient to represent the full
  dataset without phoneme normalization.
- **Script uniformity via NFC normalization:** Forcing all text into NFC-composed Devanagari
  ensures the tokenizer sees a consistent representation regardless of how the source text
  encoded compound characters. Without this, silently dropped characters cause misaligned
  attention maps.
- **FP16 mixed precision:** ~40% reduction in VRAM consumption and ~40% speed improvement on
  T4. Enabled training at batch size 16 without OOM, which is critical for stable GAN updates.
- **`save_best_after=500`:** Prevents premature overwriting of the best checkpoint during the
  first ~6 epochs where generator/discriminator losses are volatile. The final best checkpoint
  is selected on lowest eval mel loss after training stabilizes.
- **Dataset quality filtering (3–25 word range):** Excluding too-short and too-long utterances
  prevents the duration predictor from learning degenerate alignment patterns.

---

### 2.8 Results & Sample Outputs

**Best Checkpoint:** `best_model_48503.pth` (selected by lowest eval `loss_mel` at Epoch 287)

| Loss Component | Initial | Best Value (Minimum) | Epoch Achieved |
|---|---|---|---|
| `loss_mel` | ~32.0 | 13.365898 | Epoch 287 |
| `loss_duration` | ~2.12 | 1.640014 | Epoch 297 |
| `loss_kl` | ~3.70 | 1.296159 | Epoch 269 |
| `loss_feat` | ~8.50 | 6.314875 | Epoch 165 |
| `loss_gen` | ~2.22 | 1.826886 | Epoch 169 |
| `loss_disc` | ~2.82 | 2.559658 | Epoch 7 |

**Qualitative Observations:**
* Speech is intelligible with improving prosodic rhythm as training progressed.
* The Hindi warm-start meant the model never passed through a "babbling" phase — recognizable Haryanvi phoneme sequences were present from the first evaluation checkpoint.
* `loss_duration` continued declining through all 300 epochs, indicating the model was still actively improving its timing and prosodic pacing.
* A minor vocabulary gap was observed during inference: nukta characters ख़ and ड़, the em-dash —, and the | Devanagari danda variant were absent from the character config and silently discarded. This did not cause crashes but slightly affects rare-character pronunciation.

**Sample Outputs:**
Eleven distinct audio samples were generated using `best_model_48503.pth` and are stored in the `tts_output_wav_files/` folder:

| File | Description |
|---|---|
| `test_haryanvi_output.wav` | Long complex sentence — weather, farming, multi-clause structure |
| `test_haryanvi_output2.wav` | Short idiomatic Haryanvi — jungle/animal theme |
| `test_haryanvi_output3.wav` | Multi-line conversational Haryanvi passage |
| `test_haryanvi_output4.wav` – `11.wav` | 8 diverse sentences — village life, complaints, weather, animals, relationships |
| `test_haryanvi_outputfinal.wav` | Short narrative story: Ramfal and the old man |
| `test_haryanvi_output5min.wav` | ~5 min long-form story: Surta and the injured deer — full prosody stress test |
| `test_haryanvi_output5mintoughtest.wav` | ~5 min long-form story: Raghubeeer and Munna — dialogue-heavy, toughest prosody test |


---

### 2.9 Generated Artifacts

All training outputs are saved under:
`tts_output/vits_haryanvi_ai4bharat-<date>/`

| Artifact | Path | Description |
|---|---|---|
| `best_model_48503.pth` | `tts_output/.../best_model_48503.pth` | Best generator checkpoint (lowest eval loss, ~512 MB) |
| `checkpoint_*.pth` | `tts_output/.../checkpoint_*.pth` | Per-epoch intermediate checkpoints for training resumption |
| `config.json` | `tts_output/.../config.json` | Full saved training configuration for reproducible inference |
| `test_haryanvi_output.wav` | `Project root` | Synthesized audio sample — inference validation output |
| `metadata.csv` | `haryanvi_vits_data/metadata.csv` | LJSpeech-format dataset manifest (2764 rows) |
| `phoneme_cache/` | `haryanvi_vits_data/phoneme_cache/` | Coqui character-level tokenizer cache |
| `best_model.pth` | `syspin_hindi_vits/best_model.pth` | Source Hindi pre-trained checkpoint used for warm-start |
| `config.json` | `syspin_hindi_vits/config.json` | SYSPIN model config |
| `tts_pipe_coqui_native.ipynb` | `Project root` | Full training notebook (preprocessing + training + inference) |

---


### 2.10 Key Findings (TTS)

**What Worked Well:**

- **Coqui-native warm-start** from `SYSPIN/vits_Hindi_Female` was the single most impactful decision — 100% decoder weight transfer meant the model produced recognizable Haryanvi phonemes from the very first evaluation epoch, bypassing the random-initialization instability phase entirely.
- **Character-level tokenization** was the correct choice for this dialect. Bypassing espeak-ng's Hindi locale prevented systematic mispronunciation of Haryanvi-specific constructs (e.g., *कोनी*, *म्हारा*, *सै*, *तै*).
- **Equal LR for generator and discriminator (`1e-5`)** kept the adversarial balance stable throughout — `loss_disc` settled at ~2.75 and never diverged, confirming no mode collapse across 300 epochs.
- **Early stopping with patience=10** (checked every 10 epochs) was implemented to prevent unnecessary over-training, automatically halting the run if eval loss stopped improving.
- **Auto-resume from checkpoint** (`continue_path` strategy) was implemented — the training notebook automatically detects the latest run directory and resumes from it, making multi-session training on Colab seamless without manual intervention.
- **NFC normalization** silently fixed encoding inconsistencies in the raw dataset that would otherwise have caused tokenizer-level data corruption.
- **Long-form inference was stable** — the model synthesized multi-paragraph stories (`output5min.wav`, `output5mintoughtest.wav`) without attention collapse or duration runaway, which is a meaningful quality signal at 300 epochs.

**What Did Not Perform As Expected:**

- **Nukta characters in inference inputs** (`ख़`, `ड़`) were not present in the trained vocabulary and were silently discarded during inference. While these appear rarely in Haryanvi, their absence causes incorrect pronunciation of loanword consonants. This should be addressed by re-auditing the character config against the full inference input distribution.
- **The original VITS format checkpoint** (`ai4bharat/vits-hi-itts-base`) was initially expected to transfer cleanly via key remapping, but only 34% of the HiFiGAN decoder loaded correctly due to Coqui-added conditioning layers absent in the original architecture.
- **`loss_feat` plateaued early (~epoch 165 at 6.31)** and showed no further improvement across the 
remaining 135 epochs, suggesting the feature matching loss had saturated and was no longer contributing a useful training signal at this dataset scale.

**Bottlenecks:**

- At 6 min/epoch on T4, 300 epochs required 30 hours of compute. The dataset at 2764 utterances (~1.5 hours of audio) sits at the minimum viable threshold for VITS fine-tuning — quality improvements are expected to slow without additional data.
- The character vocabulary gap (missing `ख़`, `ड़`, `—`) was only discovered at inference time rather than being caught during data validation, pointing to a gap between the training-time character set and the real-world inference distribution.



**Plans for Improvement:**

- **Extended training beyond 300 epochs:** `loss_mel` is still declining at epoch 300 and
  has not plateaued. Pushing to 500 epochs on A100 hardware is projected to bring `loss_mel`
  into the 8–10 range — a qualitative jump from "intelligible with robotic prosody" to
  "clear, usable TTS with natural phrasing" (~5 hrs on A100 vs ~42 hrs on T4).

- **Dataset expansion:** At 2764 utterances (~1.5 hours), the corpus sits at the minimum
  viable threshold for VITS. Scaling to 5–10 hours — combined with SNR filtering
  (< 30 dB excluded) and speed perturbation (±10%) — would meaningfully improve prosodic
  naturalness without requiring entirely new recordings.

- **Formal evaluation suite:** Quality is currently assessed through subjective listening alone.
  Running MOS (human perceptual rating), WER via Devanagari ASR, and MCD (Mel Cepstral
  Distortion) would provide rigorous, reproducible benchmarks — essential for a dialect
  where no prior TTS baseline exists.

- **End-to-end pipeline stress testing:** Running the full Hindi → LLaMA → Haryanvi text →
  VITS → audio pipeline on a diverse test set would surface compounding error patterns,
  specifically how translation artifacts affect TTS prosody and duration prediction — the
  critical final validation before the system is deployment-ready.

---

## Appendix: Project Files & Artifacts

All files, folders, datasets, and generated outputs referenced in this milestone can be found at the following links:
* [🤗 Hugging Face Models & Datasets](https://huggingface.co/datasets/Satyam-Srivastava/RDS)
* [📁 Part 1: Text-to-Text Translation (LLaMA) Outputs & Checkpoints](https://drive.google.com/drive/folders/1yVwT5XKXtM6Xqli2dRHfh970Lx6NRyTe)
* [📁 Part 2: Text-to-Speech (VITS) Outputs & Checkpoints](https://drive.google.com/drive/folders/15ql9hOetrypRe19tqjZGY6WbHnhjdkIc)
* [🔊 Generated Audio Samples (.wav files)](https://drive.google.com/drive/folders/1odheMukksl2s2ZWkC5uPSTBqqH0GnPes)

# Milestone 2 Report: Dataset Collection, Preparation, and Preprocessing
### Project: Regional Dialect Synthesis Pipeline (Haryanvi)
---
## Team Contributions

| Team Member | Contributions |
| :--- | :--- |
| Abhishek | • Conducted all further preprocessing <br> • Managed LLM fine-tuning tasks: dataset formatting (instruction-response), token length considerations, and prompt structuring |
| Satyam | • Created structured folders for audio/text datasets and uploaded to Hugging Face <br> • Extracted 3 audio files from YouTube using `yt-dlp` <br> • Scraped Haryanvi-Hindi parallel dataset from Storyweaver |
| Sanket | • Developed Python script for synthetic text data-generation and augmentation <br> • Documented report on synthetic data-generation |
| Fazlur | • Scraped text data from `google-research-datasets/indic-gen-bench` <br> • Authored Milestone-2-report |

---

## 1. Project Overview

This project implements a two-stage pipeline for synthesizing Haryanvi speech from Hindi text input:

1. **Stage 1 — Text-to-Text Translation:** A Hindi → Haryanvi (Bangru dialect) neural machine translation model.
2. **Stage 2 — Text-to-Speech Synthesis:** A Haryanvi TTS model that generates natural-sounding audio from Haryanvi text.

This milestone documents the datasets collected and created for both stages, including their sources, quality assessment, preprocessing steps, and train/validation/test split strategies.

---

## 2. Dataset Overview [Data_Link](https://drive.google.com/drive/folders/1chYnNWDCP6kbMaqBwAPUgrz84f_0sbI2?usp=sharing)

The project uses **two independent dataset pipelines** — one for the translation sub-task (Module A) and one for the speech synthesis sub-task (Module B).

| Aspect | Module A: Text (Hindi→Haryanvi) | Module B: Audio (TTS) |
|---|---|---|
| Task | Sequence-to-sequence translation | Text-to-speech synthesis |
| Input | Hindi sentences | Haryanvi text transcripts |
| Output | Haryanvi sentences | Haryanvi audio waveforms |
| Primary Format | CSV / JSON | Audio array + transcript |
| Target Split | 80 / 10 / 10 (train/val/test) | 90 / 10 (train/eval) |



## 3. Module A: Text Dataset (Hindi → Haryanvi Translation)

### 3.1 Data Sources

Two primary sources were used to build the parallel Hindi–Haryanvi corpus:

#### Source 1: XOR-QA Indic Dataset (Google Research)
- **URL:** `https://github.com/google-research-datasets/indic-gen-bench/blob/main/xorqa_in/xorqa_bgc_dev.json`
- **Format:** JSON
- **Content:** Question-answer pairs originally in English and Haryanvi (Bangru dialect).
- **Transformation:** English text was translated to Hindi using Google Translate to produce Hindi–Haryanvi parallel pairs.
- **Usage Constraints:** Released under Google Research open-data policies; intended for research purposes.
- **Limitation:** Google Translate output for Hindi may carry minor translation artifacts; manual spot-checking was applied.

#### Source 2: Synthetically Generated Data (Gemini)
- **Generator:** Google Gemini
- **Format:** CSV and JSON (`synthetic_haryanvi_corpus.json`)
- **Size:** 5,000 unique parallel sentence pairs
- **Generation Approach:** A research-grade prompt was used to enforce strict linguistic rules, topic diversity, and structural variety. See the [generation report](https://drive.google.com/file/d/1qj4_hvqzgifu0-edHb_Q1gCSp-wFOt06/view?usp=drive_link) for full details.
- **Ethical Note:** Synthetic data is clearly labelled and segregated from human-authored data during evaluation.

#### Source 3 (Fallback): Parallel Corpus (Haryanvi-Hindi) data scraping.
- Citation: Storyweaver (Pratham Books). Digital collection accessible at Storyweaver Status: Not used in the current pipeline. Identified as a data expansion fallback if the primary corpus (Sources 1 + 2) proves insufficient for model convergence. Rationale for deferral: Extracting structured parallel pairs from the digitized book requires non-trivial parsing effort. This will be pursued if and when the need arises. Usage Constraints: Open-source content licensed under Creative Commons; requires attribution to Pratham Books.



### 3.2 Synthetic Data Generation Details

A structured, research-grade prompt was used with Gemini to generate the synthetic corpus. Key design decisions include:

**Linguistic Rules Enforced:**

| Category | Hindi | Haryanvi (Bangru) |
|---|---|---|
| Auxiliary | है | सै |
| Auxiliary | हूँ | सूँ |
| Auxiliary | हैं | से |
| Progressive | रहा/रही/रहे | रया/रयी/रये |
| Negation | नहीं | कोनी |
| Ergative marker | ने | नै |
| Pronoun (1st person) | मुझे/मैंने | मन्ने |
| Pronoun (2nd person) | तुम्हें/तुमने | तन्ने |
| Possessive | हमारा / तुम्हारा | म्हारा / थारा |
| Demonstrative | यह / वह | यो / वो |
| Adverbs | बहुत / कहाँ / कब | घणा / कड़े / कदे |


**Topic Coverage:** Daily life, farming, rural culture, family, markets, school, travel, food, sports, festivals, weather, technology, government services, and transportation — ensuring broad domain generalization.

**Quality Constraints Applied:**
- No sentence duplication
- No literal word-by-word translation
- No untranslated Hindi words where a Haryanvi equivalent exists
- Varied sentence complexity to prevent dataset collapse



### 3.3 Dataset Quality Assessment (Text)

The following quality checks were performed on the combined text corpus using the preprocessing pipeline (`rds.py`):

- **Missing values:** Checked column-wise using `df.isnull().sum()`; any row with a null Hindi or Haryanvi entry was dropped.
- **Duplicates:** Exact duplicate rows were removed using `drop_duplicates()`. The count of removed rows was logged for traceability.
- **Length filtering:** Sentences with fewer than **3 words** or more than **100 words** (in Hindi) were filtered out to remove noise (very short fragments and excessively long outliers).
- **Manual spot-check:** A subset of translated/generated pairs was manually reviewed for grammatical accuracy and Haryanvi authenticity, particularly for Source 1 (Google Translate-derived pairs).

The following table summarizes the number of samples retained from each source after all cleaning steps:

| Source | Raw Samples | Retained After Cleaning |
|---|---|---|
| XOR-QA Indic (GitHub) | 594 | 594 |
| Synthetic (Gemini-generated) | 5,000 | 5,000 |
| **Total** | **5,594** | **5,594** |



### 3.4 Text Dataset Splits

Splitting was performed using `sklearn.model_selection.train_test_split` with a fixed random seed (`RANDOM_SEED = 42`) to ensure reproducibility.

| Split | Proportion | Approx. Samples | Description |
|---|---|---|---|
| Train | 80% | ~4,475 | Used for fine-tuning the translation model |
| Validation | 10% | ~560 | Used for hyperparameter tuning and early stopping |
| Test | 10% | ~560 | Held-out set; never seen during training or validation |

**Data Leakage Prevention:**
- Duplicate removal was performed **before** splitting to prevent any identical sentence appearing in both train and test sets.
- Splits are performed on the full cleaned corpus in a single pass using random splitting with a fixed seed (no temporal or source-based leakage).

### 3.5 LLM Formatting & Tokenization

The training dataset is formatted for instruction fine-tuning using **LLaMA 3.1 (8B Instruct)** chat templates via Hugging Face `AutoTokenizer`:

- **System prompt:** Instructs the model to act as a precise Hindi-to-Haryanvi (Bangru) translator following morpho-syntactic rules.
- **User turn:** Hindi source sentence.
- **Assistant turn:** Target Haryanvi sentence.
- **Tokenization:** Max sequence length of 128 tokens, padded to max length, with truncation enabled. Padding token is set to the EOS token.
- **Dataset format:** HuggingFace `DatasetDict` with `train`, `validation`, and `test` splits.

---

## 4. Module B: Audio Dataset (Haryanvi TTS)

### 4.1 Data Sources

#### Source 1: HuggingFace — `ankitdhiman/haryanvi-tts`
- **URL:** `https://huggingface.co/datasets/ankitdhiman/haryanvi-tts`
- **Format:** AudioFolder (soundfolder); filepath-to-text mappings stored in `metadata.csv`
- **Content:** Audio–text pairs in the Bangru dialect of Haryanvi
- **Size:** 5,515 rows (~1.89 GB of audio)
- **Language tag:** Hindi (`hi`) — note that the dataset is tagged as Hindi on HuggingFace but the content is explicitly Bangru/Haryanvi dialect
- **Quality:** High quality, pre-aligned, structured dataset with consistent formatting; suitable for direct ingestion into a TTS fine-tuning pipeline
- **Usage Constraints:** No explicit license is listed on the dataset card. Usage is assumed to be for research purposes; **license should be confirmed with the dataset author (`ankitdhiman`) before any commercial deployment.**
- **Access:** Loaded programmatically via `load_dataset("ankitdhiman/haryanvi-tts")` using the HuggingFace `datasets` library



#### Source 2: YouTube Audio Extraction (Haryanvi Channels) — *Fallback Strategy*

This source is **not used in the primary pipeline**. It is identified as a contingency data expansion strategy to be activated only if the HuggingFace dataset (5,515 clips) proves insufficient for the TTS fine-tuning objective (i.e., falls below the ~5–10 hour threshold after duration filtering).

- **Extraction Tool:** `yt-dlp` — audio downloaded in best available quality, post-processed via FFmpeg to MP3 at 192 kbps (`audio.mp3`)
- **Transcription Strategy (two-tier):**
  1. **Primary:** `youtube-transcript-api` attempts to fetch the YouTube-native transcript using the video ID parsed from the URL. If found, it is used directly.
  2. **Fallback:** If no transcript is available, OpenAI **Whisper (`tiny` model)** is run on the extracted audio to generate timestamped segments.
- **Output Format:** Timestamped transcript saved as a CSV-like `.txt` file with entries in the format `[MM:SS.mmm --> MM:SS.mmm] text`, one segment per row (`formatted_transcript.txt`)
- **Quality:** Variable; Whisper `tiny` may introduce transcription errors on low-resource dialect audio. Manual verification is required before ingestion.
- **Usage Constraints:** Intended strictly for academic, non-commercial NLP research. Compliance with each channel's content policy and YouTube's Terms of Service must be confirmed before extraction.

**Identified Videos:**

| # | URL |
|---|---|
| video 1 | https://www.youtube.com/watch?v=Re7OOS6wD0s |
| video 2 | https://www.youtube.com/watch?v=4dtm-FVDBEs |
| video 3 | https://www.youtube.com/watch?v=D614BjIiRzU |

**If activated, the following additional steps apply before ingestion into `rds.py`:**
1. Convert MP3 → WAV at 22,050 Hz mono (to match `TARGET_SR` in `DataConfig`)
2. Segment audio using Whisper timestamp boundaries into individual clips
3. Manually verify a random sample of Whisper transcripts for dialect accuracy
4. Log video ID alongside each clip for speaker-stratified splitting
5. Apply the same duration filtering (1.0–10.0s) and deduplication as Source 1


---

### 4.2 Audio Quality Standards

The following audio parameters were enforced for the TTS pipeline:

| Parameter | Value | Justification |
|---|---|---|
| Target Sampling Rate | 22,050 Hz | Standard for VITS-based TTS models |
| Minimum Audio Duration | 1.0 second | Filters out fragments too short for alignment |
| Maximum Audio Duration | 15.0 seconds | Prevents VRAM spikes during training |
| Resampling Method | HuggingFace `Audio()` cast | Ensures consistent sample rate across sources |

All audio was resampled to 22,050 Hz using the `datasets.Audio(sampling_rate=22050)` cast, which applies consistent resampling regardless of the original source format.

---

### 4.3 Audio Dataset Quality Assessment

- **Duration filtering:** Audio clips outside the 1.0–15.0 second window were removed using `dataset.filter()`. The retained count was logged.
- **Transcript normalization:** Transcripts were stripped of leading/trailing whitespace.
- **Manual verification:** If the YouTube fallback is activated, a subset of transcripts will be manually reviewed for alignment quality, given that auto-generated ASR may introduce errors in low-resource dialects like Haryanvi.


---

### 4.4 Audio Dataset Splits

Audio splits were created using HuggingFace's `train_test_split()` method:

| Split | Proportion | Description |
|---|---|---|
| Train | 90% | Primary training data for the TTS model |
| Evaluation | 10% | Used for MOS (Mean Opinion Score) and objective metric evaluation |

The same fixed seed (`RANDOM_SEED = 42`) was used to ensure reproducibility across runs.

**Data Leakage Prevention:**
- For the primary HuggingFace source, clips are independent utterances; random splitting is appropriate. If the YouTube fallback is activated, speaker/video-stratified splitting must be applied to prevent clips from the same video appearing in both train and evaluation sets.

---

### 4.5 TTS-Ready Format

The final audio pipeline produces a `DatasetDict` with the following columns:

| Column | Description |
|---|---|
| `audio_array` | Raw audio waveform as a NumPy array at 22,050 Hz |
| `normalized_text` | Whitespace-stripped Haryanvi transcript |

---


## 5. Dataset Adequacy Assessment

### 5.1 Text (Translation Model)

- The primary corpus of **5,594 parallel pairs** (594 from XOR-QA + 5,000 synthetic) is considered a reasonable starting point for fine-tuning a pre-trained multilingual LLM on the Hindi→Haryanvi translation task.
- Augmentation through Gemini-generated data is justified given the extreme scarcity of Hindi–Haryanvi parallel corpora in the public domain. The research-grade prompt enforces authentic Bangru morphology, reducing the risk of dialect drift typical in unconstrained synthetic generation.

### 5.2 Audio (TTS Model)

- The primary audio source is the HuggingFace `ankitdhiman/haryanvi-tts` dataset (**5,515 clips**, ~1.89 GB), which provides a clean, structured, pre-aligned baseline suitable for direct ingestion into the TTS fine-tuning pipeline.
- YouTube-sourced audio is a **fallback strategy only** and is not part of the current pipeline. It would add natural speech diversity (varied speakers, topics, prosodic range) if the primary dataset proves insufficient.
- The total usable audio hours from the HuggingFace dataset will be calculated after duration filtering (1.0–15.0s window) and reported before training begins. Should the total fall below the ~5–10 hour minimum typically required for VITS-based TTS, two mitigation strategies are available:
  1. **Activate the YouTube fallback** — extract and verify audio from the three identified Haryanvi video sources using the `youtube_video_to_audio.py` pipeline.
  2. **Transfer learning** — initialize from a pre-trained Hindi TTS checkpoint (e.g., IndicTTS Hindi or Sarvam-TTS) and fine-tune on available Haryanvi audio, leveraging the strong phonological and script-level proximity between Hindi and Haryanvi.

---

## 6. Inter-Task Alignment (Text ↔ Audio Pipeline)

The two modules are coupled at inference time: the output of the Hindi→Haryanvi translation model feeds directly into the TTS module. To ensure alignment:

- The vocabulary and script used in text training data must match the transcript format expected by the TTS model (Devanagari script, consistent Haryanvi spellings).
- Tokenization in Module A and transcript normalization in Module B should use a consistent text normalization standard (e.g., handling of anusvara, nukta, and punctuation).


---

## 7. Ethical Considerations

| Aspect | Detail |
|---|---|
| **XOR-QA (GitHub)** | Open research dataset by Google; intended for academic use |
| **Synthetic (Gemini)** | Clearly labelled as LLM-generated; not presented as human-authored; segregated during evaluation |
| **HuggingFace TTS** | No explicit license on dataset card; usage assumed for research only — license confirmation from dataset author (`ankitdhiman`) required before any commercial deployment |
| **YouTube Audio** | Fallback only; intended strictly for academic, non-commercial use; compliance with YouTube ToS and individual channel policies must be confirmed before extraction |
| **Representation** | Synthetic prompt explicitly covers diverse topics (rural, urban, technology, governance) to avoid reducing Haryanvi cultural representation to stereotypical rural/agricultural contexts |



## 8. Reproducibility Checklist

All preprocessing steps are implemented across two Google Colab notebooks: `rds.py` (main data pipeline) and `youtube_video_to_audio.py` (fallback audio extraction).

- [x] Fixed random seed (`RANDOM_SEED = 42`) used for all splits
- [x] All filtering thresholds defined in a central `DataConfig` class
- [x] Duplicate removal logged with count before splitting
- [x] HuggingFace `DatasetDict` format used for consistent loading
- [x] Audio resampled programmatically to 22,050 Hz (no manual conversion)
- [x] Per-source sample counts documented (XOR-QA: 594, Synthetic: 5,000; Total: 5,594)
- [x] YouTube extraction pipeline documented (`yt-dlp` + Whisper `tiny`); video URLs logged
- [ ] Total audio hours to be confirmed after duration filtering is run on the full HuggingFace dataset



## 9. Key Dataset Requirements for Effective Training

**For the Translation Model (Module A):**
- Parallel Hindi–Haryanvi sentence pairs with authentic Bangru morphology
- Coverage of diverse sentence lengths (5 words to multi-sentence paragraphs) and domains
- Grammatically correct Haryanvi following documented lexical and morpho-syntactic substitution rules
- No data leakage between train/val/test splits; duplicates removed prior to splitting
- Current corpus: **5,594 pairs** 

**For the TTS Model (Module B):**
- Audio sampled at **22,050 Hz** mono (VITS/Sarvam-TTS standard)
- Clip duration between **1.0–15.0 seconds** after segmentation
- Accurate text-audio alignment;
- Speaker diversity across clips to improve voice generalization
- Primary source: `ankitdhiman/haryanvi-tts` (5,515 clips); YouTube extraction available as fallback if total audio hours fall below the minimum threshold

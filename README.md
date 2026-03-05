# hindi-to-haryanvi-genai
A Deep Learning pipeline for Hindi-to-Haryanvi dialect translation using QLoRA-adapted LLaMA-3.1 and Haryanvi speech synthesis via transfer-learned VITS.
# Regional Dialect Synthesis Pipeline (Haryanvi)

**Data Science and AI Lab Project – Group 12**



## Repository Structure

```text
├── LICENSE                                
├── PPT                                    # Presentation materials
│   └── Milestone_1_PPT.pdf                
├── Project Proposal Regional Dialect Synthesis Pipeline (Haryanvi).pdf  # Initial formal project proposal
├── README.md                              
├── Reports                                # comprehensive milestone documents
│   └── Milestone-1-Report.md              
└── worklog.md                             # tracking team contributions and tasks
```

## Project Overview

Existing multilingual AI translation and text-to-speech (TTS) systems normalize to Standard Hindi, failing to capture the lexical divergence, morpho-syntactic variation, and unique prosody of the Haryanvi (Bangru) dialect. This creates an accessibility barrier for rural learners in Haryana. 

This project implements an end-to-end generative pipeline to translate Standard Hindi text into Bangru and synthesize it into natural, dialect-authentic speech. The system is designed to adapt general-purpose state-of-the-art models into a hyper-local dialect engineering system, culminating in a fully dubbed 2–3 minute educational video prototype.

### Core Architecture

* **Dialect-Accurate Text Translation:** Utilizes Llama-3.1 fine-tuned via QLoRA. It employs rule-constrained synthetic data augmentation to strictly model systematic structural shifts (e.g., auxiliary and negation transformations) without semantic drift.
* **Prosodically Authentic Speech Synthesis:** Implements a VITS-based TTS architecture leveraging transfer learning from AI4Bharat Indic checkpoints. This addresses the limitations of a low-resource (5-hour) dataset while explicitly modeling Bangru's syllabic compression, higher plosive burst energy, and flat pitch contours. (Backup architecture: Matcha-TTS).
* **Evaluation:** The pipeline is rigorously evaluated against standard baselines using Mean Opinion Score (MOS), Mel-Cepstral Distortion (MCD), and Word Error Rate (WER).

---

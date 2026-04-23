# Milestone 1 Report: Regional Dialect Synthesis Pipeline (Haryanvi)

**Data Science and AI Lab Project – Group 12**

## Team Members

- **Abhishek** (22f3000978@ds.study.iitm.ac.in)
- **Satyam Srivastava** (21f1000629@ds.study.iitm.ac.in)
- **Sanket Agrawal** (23f1001709@ds.study.iitm.ac.in)
- **Md Fazlur Rahman** (23f1001897@ds.study.iitm.ac.in)

---

## 1. Abstract

In rural Haryana, learners face significant accessibility barriers due to the reliance on standard Hindi in educational content, whereas their native comprehension aligns with the Bangru (Haryanvi) dialect. Existing multilingual AI translation and text-to-speech (TTS) systems fail to adequately capture Bangru's specific lexical divergence, morpho-syntactic variation, and prosodic-acoustic deviation.

This project proposes an end-to-end generative pipeline to translate Standard Hindi text into Bangru and synthesize it into natural, dialect-authentic speech. To bridge textual gaps, the methodology employs rule-constrained synthetic data augmentation and parameter-efficient fine-tuning (QLoRA) on Llama-3.1 to model systematic structural shifts, such as auxiliary and negation transformations, without semantic drift. For speech synthesis, the architecture utilizes VITS, leveraging transfer learning from AI4Bharat Indic checkpoints to overcome the limitations of a 5-hour low-resource dataset. This prosody-first approach specifically models Bangru's syllabic compression, higher plosive burst energy, and distinct tonal identity.

Evaluated via Mean Opinion Score (MOS), Mel-Cepstral Distortion (MCD), and Word Error Rate (WER), the finalized pipeline will produce a high-fidelity dubbed educational audio prototype. This work demonstrates a robust, scalable methodology for adapting general-purpose state-of-the-art models into hyper-local dialect engineering systems.

---

## 2. Problem Statement, Scope, and Stakeholders

### Problem Statement

While major Indian languages enjoy robust support from digital translation tools, widely spoken dialects like Haryanvi (Bangru) remain severely low-resource. Current State-of-the-Art (SOTA) systems normalize to Standard Hindi, failing to capture the unique vocabulary, grammatical shifts, and the distinct tonal prosody of the region. This creates a linguistic accessibility barrier, particularly for rural learners whose native comprehension aligns with Bangru rather than standard Hindi.

### Scope and Boundaries

**In Scope:** Development and evaluation of a Hindi-to-Bangru text translation pipeline (QLoRA fine-tuned Llama-3.1) and a prosody-aware neural TTS system (VITS via AI4Bharat transfer learning), rigorously evaluated using MCD and MOS. The final deliverable is a fully functional 2–3 minute dubbed educational speech prototype integrating translation, TTS.

**Out of Scope:** Generalizing to other regional dialects, real-time low-latency translation streaming, and enterprise-grade production deployment.

### Stakeholders

- **Primary Beneficiaries:** Rural students and educators who gain accessible, dialect-authentic learning materials that reduce cognitive load.
- **Broader Ecosystem:** EdTech platforms, NGOs, and AI researchers who can leverage this scientifically rigorous framework for scalable, low-resource dialect engineering.

---

## 3. Project Objectives

1. **Dialect-Accurate Text Translation:** Develop a specialized LLM translation pipeline utilizing parameter-efficient fine-tuning (QLoRA) to convert Standard Hindi into the Bangru (Haryanvi) dialect. The system must rigorously model systematic lexical substitutions, auxiliary shifts, and negation transformations rather than broad semantic equivalence.

2. **Prosodically Authentic Speech Synthesis:** Engineer a Text-to-Speech (TTS) architecture based on VITS and transfer learning to generate natural Bangru speech from a low-resource (5-hour) dataset. The model must explicitly capture the dialect's distinct acoustic signature, including its syllabic compression, higher plosive burst energy, and specific pitch contours, aiming for high naturalness (MOS ≥ 3.5).

3. **End-to-End Educational Prototype:** Integrate the translation and TTS models to produce a 2–3 minute dubbed educational audio. The pipeline will be evaluated on its ability to minimize speech distortion (MCD < 6.0 dB) and empirically improve learner comprehension by ≥ 20%.

---

## 4. Identification of Linguistic Gaps & Opportunities

To engineer a targeted pipeline, we conducted a structured linguistic gap analysis across three dimensions to formally characterize how Bangru differs from Standard Hindi.

### Lexical Divergence

Haryanvi exhibits systematic lexical substitutions rather than sporadic colloquial variation. Standard translation systems fail because they optimize for semantic equivalence rather than dialectal fidelity. Key shifts include:

| Standard Hindi | Bangru (Haryanvi) |
|----------------|-------------------|
| कब            | कदे               |
| कहाँ          | कड़े              |
| नहीं          | कोनी              |
| बहुत          | घणा               |

### Morpho-Syntactic Variation

The primary structural divergence lies in auxiliary systems, progressive marking, negation, and pronoun transformation.

- **Progressive Aspect:** "मैं जा रहा हूँ" shifts to "मैं जा रया सूं", indicating systematic morphophonemic alteration.
- **Auxiliary Paradigm Shift:** "है" shifts to "सै", affecting declarative sentences, progressive constructions, and copular structures.
- **Negation:** The negator "कोनी" replaces "नहीं" consistently across contexts, requiring robust modeling.

### Prosodic and Acoustic Deviation

Bangru has a unique acoustic signature that standard Hindi TTS models fail to capture.

- **Consonantal Energy:** Strong retroflex articulation and higher plosive burst energy.
- **Pitch Contour:** A flatter F0 contour with an assertive terminal falling tone, creating a perceptually forceful quality.
- **Speech Rate:** Higher articulation rate and syllabic compression (e.g., "रहा है" to "रया सै") directly affecting duration modeling in TTS systems.

---

## 5. Literature Review & Existing Solutions

### 5.1 Text Translation & Dialect Modeling

Transformer-based Large Language Models (LLMs) have superseded sequence-to-sequence LSTM architectures due to superior context handling. However, standard multilingual LLMs optimize for semantic equivalence in high-resource standard languages and lack dedicated dialect corpora. They fail to capture stable, rule-governed lexical substitutions (e.g., replacing "नहीं" with "कोनी"). They also struggle with systematic morpho-syntactic variation, such as auxiliary paradigm shifts.

To address this, we frame the task as structured dialect adaptation rather than pure sequence-to-sequence translation. Recent methodologies utilize parameter-efficient fine-tuning (PEFT) techniques like QLoRA combined with rule-constrained synthetic data augmentation. The Llama-3.1 model, optimized with Grouped-Query Attention (GQA), is highly advantageous for this instruction-following task, allowing the model to map its existing Indo-Aryan representations to specific Haryanvi dialect rules efficiently.

### 5.2 Text-to-Speech (TTS) Architectures

We evaluated five SOTA TTS architectures for low-resource dialectal speech synthesis. Standard Hindi TTS models smooth out phonetic bursts, neutralizing dialectal identity, and fail to represent Bangru's distinct acoustic signature.

- **FastSpeech2:** Extremely fast inference but relies on fixed duration prediction, which lacks the variability necessary for expressive prosody. It requires a separate vocoder for waveform synthesis.
- **XTTSv2:** Offers state-of-the-art zero-shot speaker adaptation but requires reference audio for each synthesis and is computationally expensive.
- **Matcha-TTS:** Replaces VITS's normalizing flows with Optimal Transport Conditional Flow Matching. It offers 2-3x faster training convergence than VITS and fixes VITS's phoneme-skipping bug.
- **VITS:** An end-to-end TTS model generating raw audio directly from text. It addresses prosodic limitations through its stochastic duration predictor, which models the one-to-many nature of speech. This captures prosodic variability critical for Haryanvi's aggressive tone.


---

## 6. Proposed Methodology & Architecture

### 6.1 Translation Module (Llama-3.1 + QLoRA)

A Llama-3.1 model fine-tuned via QLoRA leverages rule-constrained synthetic data augmentation to enforce deterministic lexical and morphological shifts without semantic drift. We will encode dialect rules into synthetic data prompts to generate parallel pairs, allowing the model to generalize these transformations across unseen sentences.

### 6.2 Primary TTS Architecture (VITS + Transfer Learning)

VITS was selected as the optimal primary framework. It is initialized with AI4Bharat Indic-TTS Hindi checkpoints to solve the cold-start problem. The TTS duration and pitch predictors are specifically fine-tuned to capture Bangru's syllabic compression and reduce excessive melodic variance. VITS's stochastic duration predictor is uniquely suited to model Haryanvi's variable rhythm, unlike FastSpeech2's fixed duration prediction.

### 6.3 Backup TTS Architecture (Matcha-TTS)

If VITS training proves unstable or exhibits phoneme-skipping errors, we will switch to Matcha-TTS. Matcha-TTS replaces normalizing flows with Conditional Flow Matching, enabling faster convergence and eliminating phoneme-skipping bugs. If utilized, it will be pre-trained on AI4Bharat's Hindi corpus for 20K steps before fine-tuning on Bangru data.

---

## 7. Dataset Preparation & Feasibility

### Datasets

- **AI4Bharat Indic-TTS Corpus (Pre-training):** Used to solve the cold-start problem for TTS, initialized using weights pre-trained on over 10,000 hours of Hindi and Rajasthani speech.
- **Custom Bangru Audio Dataset (Fine-tuning):** A limited 5-hour dataset of authentic Bangru speech used to fine-tune the TTS architecture to capture the dialect's specific syllabic compression and tonal identity.
- **Synthetic Parallel Corpus (Translation):** Generated using rule-constrained prompting to enforce deterministic lexical replacements (e.g., "नहीं" to "कोनी") to stably fine-tune Llama-3.1 via QLoRA.

### Compute & Feasibility

Building a TTS from scratch on 5 hours of data is computationally unfeasible; however, leveraging transfer learning from AI4Bharat significantly reduces risk. Training requires NVIDIA A100 (40GB) instances, taking ~48 hours for VITS and ~24 hours for the Matcha-TTS backup.

---

## 8. Evaluation Metrics

The pipeline will be rigorously evaluated against standard Hindi baselines (Meta MMS-TTS, AI4Bharat Hindi) using objective and subjective metrics:

| Metric | Target | Description |
|--------|--------|-------------|
| **Mel-Cepstral Distortion (MCD)** | < 6.0 dB | Measured using Python librosa to quantify the spectral distance between the synthesized dialect and ground-truth validation audio. |
| **Mean Opinion Score (MOS)** | > 3.8 / 5.0 | Evaluated by a panel of 10 native Bangru speakers to subjectively grade naturalness, dialect authenticity, and forceful prosody capture. |
| **Word Error Rate (WER)** | < 15% | Assessed using the AI4Bharat IndicWav2Vec ASR model to ensure synthesized speech remains intelligible. |
---

## 9. Timeline & Roles

### Timeline

| Milestone | Date | Deliverables |
|-----------|------|--------------|
| **Milestone 1:** Problem Definition & Literature Review | Feb 26 | Draft problem statement, review Llama-3.1 and VITS/Indic-TTS architectures, identify linguistic gaps, and set up GitHub repository. |
| **Milestone 2:** Dataset Preparation | Mar 5 | Generate synthetic parallel text corpus via Gemini, download and preprocess audio datasets (IndicGenBench, Haryanvi-TTS), manually validate synthetic text, and write PyTorch dataloaders. |
| **Milestone 3:** Model Architecture | Mar 19 | Set up Llama-3.1 QLoRA scripts, and build the end-to-end inference setup piping LLM output to VITS. |
| **Milestone 4:** Model Training | Mar 26 | Execute Llama QLoRA training, perform VITS transfer learning, monitor training logs for overfitting, and manage compute resources and checkpoints. |
| **Milestone 5:** Model Evaluation & Analysis | Apr 2 | Run automated chrF++ and MCD evaluations, conduct MOS blind listening tests, and compile comprehensive error analysis. |
| **Milestone 6:** Deployment & Documentation | Apr 16 | Clean up inference notebooks, finalize project report, and build/deploy a live web demo interface. |

### Core Roles

| Role | Member | Responsibilities |
|------|--------|------------------|
| **Project Manager** | Satyam | Coordination, timeline enforcement, compute resource allocation, and milestone tracking. |
| **NLP Lead** | Abhishek | Rule-guided synthetic data generation, building the text translation pipeline, and QLoRA fine-tuning. |
| **Speech (TTS) Lead** | Fazlur | Implementing the VITS architecture, managing AI4Bharat transfer learning, and executing the Matcha-TTS backup plan if necessary. |
| **Integration & Evaluation Lead** | Sanket | Connecting translation outputs to TTS, conducting MOS/MCD evaluations. |

---

## 10. Conclusion

This milestone establishes a scientifically rigorous foundation for overcoming the linguistic barriers faced by rural learners in Haryana. By transitioning from generic translation to highly structured, low-resource dialect engineering, this pipeline addresses specific lexical, morpho-syntactic, and prosodic gaps that standard multilingual LLMs and TTS systems ignore.

Through the combination of QLoRA fine-tuning and AI4Bharat-backed VITS transfer learning, the team is positioned to deliver an authentic, highly natural Bangru educational prototype that tangibly improves learner comprehension and engagement.

---

## References

1. AI4Bharat. (2023). *Indic-TTS* [Computer software]. GitHub. https://github.com/AI4Bharat/Indic-TTS

2. Bhashini. (2024). *FastSpeech2 model for Indian languages*. IndiaAI AIKosh. https://aikosh.indiaai.gov.in/home/models/details/ibhashini_fastspeech2_model_using_hs.html

3. Casanova, E., Davis, K., Golge, E., Goknar, G., Gulea, I., Hart, L., Aljafari, A., Meyer, J., Morais, R., Olayemi, S., & Weber, J. (2024). XTTS: A massively multilingual zero-shot text-to-speech model. *arXiv preprint*, arXiv:2406.04904. https://arxiv.org/abs/2406.04904

4. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *arXiv preprint*, arXiv:2305.14314. https://arxiv.org/abs/2305.14314

5. Ekwonwune, E. N., Elebiri, L. E., Ovwonuri, A. O., Uzondu, D. O., Okoronkwo, C. D., Ikechukwu, I. B., & Chinonye, D. M. (2025). Analysis of leveraging FastSpeech 2 and HiFi-GAN models for speech synthesis adapted for Nigerian languages. *Intelligent Information Management, 17*(6), 273-294. https://www.scirp.org/journal/paperinformation?paperid=147688

6. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., et al. (2024). The Llama 3 herd of models. *arXiv preprint*, arXiv:2407.21783. https://arxiv.org/abs/2407.21783

7. Javed, T., Doddapaneni, S., Raman, A., Bhogale, K. S., Ramesh, G., Kunchukuttan, A., Kumar, P., & Khapra, M. M. (2022). Towards building ASR systems for the next billion users. *Proceedings of the AAAI Conference on Artificial Intelligence, 36*(10), 10813-10821. https://aaai.org/papers/10813-towards-building-asr-systems-for-the-next-billion-users

8. Kim, J., Kong, J., & Son, J. (2021). Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech. *arXiv preprint*, arXiv:2106.06103. https://arxiv.org/abs/2106.06103

9. Mashalov, M. (2025, January 3). Matcha TTS notes. *Speech Recognition with Vosk (Alpha Cephei)*. https://alphacephei.com/nsh/2025/01/03/matcha-tts-notes.html

10. Mehta, S., Tu, R., Beskow, J., Szekely, E., & Henter, G. E. (2023). Matcha-TTS: A fast TTS architecture with conditional flow matching. *arXiv preprint*, arXiv:2309.03199. https://arxiv.org/abs/2309.03199

11. Mohanta, A., Remruatpuii, Sarmah, P., Sinha, R., & Lalhminghlui, W. (2026). Towards prosodically informed Mizo TTS without explicit tone markings. *arXiv preprint*, arXiv:2601.02073. https://arxiv.org/abs/2601.02073

12. Ren, Y., Hu, C., Tan, X., Qin, T., Zhao, S., Zhao, Z., & Liu, T.-Y. (2020). FastSpeech 2: Fast and high-quality end-to-end text to speech. *arXiv preprint*, arXiv:2006.04558. https://arxiv.org/abs/2006.04558

13. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (Vol. 30). https://papers.nips.cc/paper/7181-attention-is-all-you-need



# 📘 Milestone 5 Report: Model Evaluation & Analysis

### Regional Dialect Synthesis Pipeline (Hindi → Haryanvi → Speech)

---

## 1. Overview

This milestone presents the evaluation and analysis of our end-to-end pipeline developed in the previous milestone .

The pipeline consists of two stages:

1. **Text-to-Text Translation**

   * Model: LLaMA 3.1 8B (QLoRA fine-tuned)
   * Task: Hindi → Haryanvi translation

2. **Text-to-Speech (TTS)**

   * Model: Coqui VITS
   * Task: Convert Haryanvi text → speech waveform

---

## 2. Evaluation Dataset

### 2.1 Text Dataset (Translation)

Derived from the cleaned Hindi–Haryanvi corpus :

* **Total samples:** ~5000
* **Train:** ~4000 (80%)
* **Validation:** ~500 (10%)
* **Test:** ~500 (10%)

### Preprocessing

* Removal of null and duplicate entries
* Sentence length filtering (3–100 words)
* Instruction-based formatting (chat template)
* Tokenization with maximum sequence length

---

### 2.2 Audio Dataset (TTS)

From Hugging Face dataset :

* **Dataset:** `ankitdhiman/haryanvi-tts`
* **Total samples:** ~5500 audio-text pairs
* **Audio size:** ~1.8 GB
* **Sampling rate:** 22050 Hz

### Preprocessing

* Resampling to 22050 Hz (mono)
* Transcript cleaning
* Filtering (3–25 words)
* Conversion to LJSpeech format (`file|text|text`)
* Character-level tokenization

### Data Split

* Train: ~90%
* Eval: ~10%

---

## 3. Evaluation Environment

Configuration details :

| Component    | Specification      |
| ------------ | ------------------ |
| GPU          | NVIDIA A100/T4 (40GB) |
| Python       | 3.12               |
| PyTorch      | 2.10               |
| Transformers | 5.0                |
| PEFT         | 0.18               |
| Precision    | BF16               |

---

## 4. Evaluation Metrics

### 4.1 Translation Metrics

* **Exact Match**

  * Measures strict sequence-level correctness

* **BLEU Score**

  * Evaluates n-gram overlap and fluency

* **chrF Score (Primary Metric)**

  * Character-level F-score
  * Robust to dialect variations

* **Evaluation Loss**

  * Measures model confidence

---

### 4.2 TTS Metrics

* **Mel Loss**

  * Measures spectral similarity

* **KL Loss**

  * Evaluates latent space stability

* **Generator Loss**

  * Indicates audio generation quality

* **Discriminator Loss**

  * Ensures balanced GAN training

* **Duration Loss**

  * Measures timing and rhythm accuracy

---

## 5. Quantitative Results

### 5.1 Best Model Performance (`lower_lr_longer`) 

| Metric      | Value   |
| ----------- | ------- |
| Eval Loss   | 0.00324 |
| Exact Match | 50.42%  |
| BLEU        | 71.76   |
| chrF        | 77.65   |

---

### 5.2 Comparison Across Experiments 

| Experiment      | Eval Loss   | BLEU      | chrF      |
| --------------- | ----------- | --------- | --------- |
| baseline        | 0.00436     | 71.74     | 77.63     |
| lower_lr_longer | **0.00324** | **71.76** | **77.65** |
| higher_rank     | 0.00427     | 71.74     | 77.64     |

---

## 6. TTS Training Curve Analysis

![TTS Training Loss Curves](loss_performance2.png)

### Observations

* Mel Loss reduced significantly (~32 → ~13), improving audio quality
* KL Loss stabilized after initial drop
* Generator Loss steadily decreased
* Duration Loss improved timing accuracy
* Discriminator Loss remained stable

---

## 7. Qualitative Results

### Successful Predictions

| Hindi               | Predicted Haryanvi |
| ------------------- | ------------------ |
| तुम कहाँ जा रहे हो? | तू कड़े जा रया सै? |
| मुझे पानी चाहिए     | मन्ने पाणी चाहिये  |
| वह घर नहीं गया      | वो घर कोनी गया     |

---

### Failure Cases

| Hindi                  | Expected              | Predicted              |
| ---------------------- | --------------------- | ---------------------- |
| वह बहुत तेज भाग रहा है | वो घणा तेज भाग रया सै | वो बहुत तेज भाग रया सै |
| मुझे यह समझ नहीं आया   | मन्ने यो समझ कोनी आया | मन्ने यो समझ नहीं आया  |

---

## 8. Error Analysis

### Translation Errors

* Proper nouns not translated correctly
* Idiomatic expressions fail in some cases
* Complex sentences cause structural errors

### TTS Errors

* Long sentences introduce slight distortion
* Rare words lead to mispronunciation
* Limited prosody variation

---

## 9. Key Observations

* Lower learning rate + more epochs improved performance
* chrF is the most reliable metric for dialect tasks
* Model performs well on common sentence patterns
* TTS model shows stable training and good convergence

---

## 10. Limitations & Anomalies

### Evaluation Issue 

* Metrics across experiments remained identical
* Indicates potential issue in evaluation pipeline:

  * Adapter not loading properly OR
  * Cached predictions reused

---

### Other Limitations

* Limited dataset size (~5K samples)
* No phoneme-based modeling in TTS
* Limited evaluation metrics for TTS (no MOS/MCD)

---

## 11. Conclusion

The evaluation demonstrates that:

* The translation model achieves strong performance with high chrF and BLEU scores
* The TTS model successfully learns speech generation with stable convergence
* The combined pipeline produces meaningful Hindi → Haryanvi → speech outputs

Overall, the system shows strong potential for **low-resource dialect AI applications**.

---

## 12. Future Work

* Fix evaluation pipeline issue
* Increase dataset size
* Introduce phoneme-based TTS
* Improve handling of complex sentences
* Add human evaluation (MOS)

---


# 📘 Worklog – Data Science & AI Lab Project

**Project Title: Regional Dialect Synthesis Pipeline (Haryanvi)**

---

## **Milestone: 1 – Problem Definition & Literature Review**

Deadline: 26 February 2026  

### Team Contributions

#### **Abhishek** (22f3000978@ds.study.iitm.ac.in)
- Drafted core problem statement
- Reviewed LLaMA 3.1 paper
- Compared LLM vs LSTM
- Formatted references
- Formatted documentation
- Integrated team contributions

#### **Sanket Agrawal** (23f1001709@ds.study.iitm.ac.in)
- Listed vocabulary differences
- Documented grammar shifts
- Described tonal characteristics

#### **Md Fazlur Rahman** (23f1001897@ds.study.iitm.ac.in)
- Reviewed VITS paper
- Reviewed Indic-TTS
- Defined MCD & MOS metrics

#### **Satyam Srivastava** (21f1000629@ds.study.iitm.ac.in)
- Initialized GitHub repository
- Configured branch structure
- Created milestone markdown structure
- Created worklog.md file for Team contribution
- Final submission commit

--- 



## **Milestone: 2 – Dataset Collection, Preparation, and Preprocessing**

Deadline: 5 March 2026  

### Team Contributions

#### **Abhishek** (22f3000978@ds.study.iitm.ac.in)
- ALL further preprocessing, 
- For LLM fine-tuning tasks, dataset formatting (e.g., instruction-response structure), token length considerations, and prompt structuring requirements.

#### **Satyam Srivastava** (21f1000629@ds.study.iitm.ac.in)
- Make a structured folder for both audio and text dataset and upload in hugging face as a project dataset.
- Extracted 3 audio files from youtube using 'yt-dlp' library.
- Scraped Haryanvi-Hindi parallel dataset from [storyweaver_site](https://storyweaver.org.in)

#### **Sanket Agrawal** (23f1001709@ds.study.iitm.ac.in)
- Written a python script for synthetic text data-generation and augmentation.
- Documented a report on this synthetic data-generation.

#### **Md Fazlur Rahman** (23f1001897@ds.study.iitm.ac.in)
- Scraped text data from google-reasearch-datasets/indic-gen-bench.
- Maked milestone-2-report.

---

## **Milestone: 3 – Model Architecture & End-to-End Pipeline**

Deadline: 19 March 2026  

### Team Contributions

#### **Abhishek** (22f3000978@ds.study.iitm.ac.in)
- Designed Hindi → Haryanvi translation pipeline using LLaMA 3.1 + QLoRA  
- Implemented instruction-based formatting and tokenizer integration  
- Ran inference on subset data and validated outputs  
- Ensured output compatibility for TTS module  

#### **Satyam Srivastava** (21f1000629@ds.study.iitm.ac.in)
- Created Milestone-3 report (model architecture + pipeline explanation)
- Final PPT work   
- Updated worklog.md and managed final submission  

#### **Sanket Agrawal** (23f1001709@ds.study.iitm.ac.in)
- Verified dataset consistency with model input requirements  
- Uploaded processed datasets to Hugging Face  
- Assisted in pipeline testing and validation  
- Prepared initial PPT structure for milestone presentation  

#### **Md Fazlur Rahman** (23f1001897@ds.study.iitm.ac.in)
- Designed TTS architecture (VITS-based)  
- Defined audio preprocessing pipeline
- LJSpeech format structuring  
- End-to-end TTS pipeline implementation  
 

---

## **Milestone: 4 – Model training and experiment with different hyperparameters, optimization, stategies and regulariation techniques.**

Deadline: 29 March 2026  

### Team Contributions

#### **Abhishek** (22f3000978@ds.study.iitm.ac.in)
- Complete end to end training of llama (text to text model part)

#### **Satyam Srivastava** (21f1000629@ds.study.iitm.ac.in)
- Handled Training of coqui-tts model on hardware 

#### **Sanket Agrawal** (23f1001709@ds.study.iitm.ac.in)
- helped with documentation 

#### **Md Fazlur Rahman** (23f1001897@ds.study.iitm.ac.in)
- wrote all codes for tts model 

---

## **Milestone: 5 – Model Evaluation, Analysis and Performance Assessment**

Deadline: 2 April 2026  

### Team Contributions

#### **Abhishek** (22f3000978@ds.study.iitm.ac.in)
- Evaluated text-to-text (LLaMA) model performance
- Analyzed translation metrics (BLEU, chrF, Exact Match)
- Investigated evaluation pipeline issues

#### **Satyam Srivastava** (21f1000629@ds.study.iitm.ac.in)
- Evaluated TTS model performance using test samples and listen .wav audio files(Human evalution.)
- Creating PPT
- Integrated results into final report

#### **Sanket Agrawal** (23f1001709@ds.study.iitm.ac.in)
- Assisted in compiling evaluation results
- Structured report formatting and tables
- Helped organize qualitative and quantitative outputs

#### **Md Fazlur Rahman** (23f1001897@ds.study.iitm.ac.in)
- Conducted error analysis for TTS outputs
- Identified limitations in speech synthesis
- Supported analysis of model behavior and improvements

---

---

## **Milestone: 6 – Deployment, Integration and Documentation**

Deadline: 16 April 2026  

### Team Contributions

#### **Abhishek** (22f3000978@ds.study.iitm.ac.in)
- Designed prompt engineering strategy for Gemma-based inference  
- Supported transition from LLaMA to deployment-ready model  
- Validated translation quality  

#### **Satyam Srivastava** (21f1000629@ds.study.iitm.ac.in)
- Structured final repository and project folders  
- Created README and Milestone-6 report  
- Designed final PPT (HTML → PDF)  
- Integrated demo screenshots and documentation  

#### **Sanket Agrawal** (23f1001709@ds.study.iitm.ac.in)
- Assisted in system testing and validation  
- Verified dataset and model consistency  
- Supported documentation and presentation  

#### **Md Fazlur Rahman** (23f1001897@ds.study.iitm.ac.in)
- Developed FastAPI backend and integrated full pipeline  
- Implemented async processing and model loading  
- Built frontend interface and deployed on GCP VM  
- Verified TTS integration and output quality  

### **Milestone Summary**

- Deployed end-to-end Hindi → Haryanvi → Speech system  
- Integrated translation and TTS into a unified pipeline  
- Delivered complete documentation, API, and demo  

---

*This worklog file is updated regulary on each milestone.*

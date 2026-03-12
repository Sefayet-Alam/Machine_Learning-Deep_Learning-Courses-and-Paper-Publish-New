# 7-Day ML/DL Paper Plan

A realistic and focused 7-day plan to help me:

1. Build enough understanding of the most relevant ML/DL/NLP topics.
2. Complete a **strong paper draft on Bangla cyberbullying detection**.
3. Build a **solid thesis-aligned foundation for multimodal sarcasm detection**.
4. Finish the week with working models, results, figures, and a reproducible project structure.

> **Main recommendation for this week:** focus primarily on **Bangla Cyberbullying Detection using Classical ML + Transformer Baselines**, because Bangla multimodal sarcasm data is scarce and hard to build properly within a week.
>
> **Secondary thesis track:** build understanding and a starter pipeline for **multimodal sarcasm detection using CLIP-based cue learning**, using an existing English benchmark first.

---

## Final Strategy

### Primary Paper to Finish This Week
**Bangla Cyberbullying Detection using Classical ML and Transformer Baselines**

Why this is the best choice:
- Bangla cyberbullying datasets already exist.
- The task is feasible within 7 days.
- You can implement both classical ML baselines and transformer baselines quickly.
- It is much more practical than trying to build a high-quality Bangla multimodal sarcasm dataset from scratch in one week.

### Secondary / Thesis-Aligned Track
**Multimodal Sarcasm Detection using CLIP-based Cue Learning**

Why this stays important:
- It aligns directly with the thesis topic.
- The reference paper uses CLIP + cue/prompt learning.
- This week, the realistic goal is to understand it deeply and prepare a starter implementation path.

---

## Day 01 — Build the Foundation You Actually Need

### To Read
1. **Reference paper**
   - [A multi-modal sarcasm detection model based on cue learning](https://www.nature.com/articles/s41598-025-94266-w.pdf)

2. **Transformer basics**
   - [Hugging Face — Natural Language Processing and Large Language Models](https://huggingface.co/learn/llm-course/en/chapter1/2)
   - [Hugging Face — How do Transformers work?](https://huggingface.co/learn/llm-course/en/chapter1/4)

3. **Classification and evaluation basics**
   - [Google ML Crash Course — Classification](https://developers.google.com/machine-learning/crash-course/classification)
   - [Google ML Crash Course — Accuracy, Precision, Recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)

### To Watch
1. [CampusX — 100 Days ML playlist](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH)
   - Only watch the most relevant parts for:
     - train/test split
     - evaluation metrics
     - overfitting/underfitting
     - basic text preprocessing

2. NLP basics from the existing resource list:
   - [NLP intro](https://youtu.be/fLvJ8VdHLA0?si=QgK5-wWPDIvVBS9o)
   - [Tokenization](https://youtu.be/YdreZtH8oWk?si=XnwgJUBV9BCRomva)
   - [Vector embedding short](https://youtube.com/shorts/FJtFZwbvkI4?si=_9O9ZUEK5bF_9eHE)
   - [Vector embedding explained](https://youtu.be/dN0lsF2cvm4?si=f9oNtbwSHmsMJLSN)

### To Do
1. Read the sarcasm paper fully once.
2. Create a note file with these headings:
   - problem
   - dataset
   - model
   - cue learning idea
   - training setup
   - evaluation metrics
   - limitations
3. Write one short paragraph each on:
   - what sarcasm detection is
   - what cyberbullying detection is
   - why Bangla cyberbullying is more feasible this week
4. Decide the exact working direction:
   - **Main paper:** Bangla cyberbullying detection
   - **Secondary thesis track:** multimodal sarcasm detection reproduction / understanding

### Code to Implement
Create the project structure:

```text
project/
├── data/
├── notebooks/
├── src/
├── results/
├── models/
└── paper_draft/
```

Create a starter notebook:
- `01_data_loading_and_baselines.ipynb`

Implement:
- dataset loading
- train/validation/test split
- basic text cleaning pipeline
- label distribution summary
- metric functions for:
  - accuracy
  - precision
  - recall
  - F1-score

### Expected Outcomes
By the end of Day 1, I should have:
- a clear understanding of the main idea of the sarcasm paper
- a final decision to focus on Bangla cyberbullying as the main fast paper
- project folders ready
- a notebook that loads data and evaluates models

---

## Day 02 — Understand Datasets and Lock the Exact Paper Problem

### To Read
1. **Bangla cyberbullying dataset / related work**
   - [Bengali cyberbullying detection: A comprehensive dataset for advanced text analysis](https://www.sciencedirect.com/science/article/pii/S2352340925009266)

2. **Additional dataset options**
   - [Mendeley — Bangla Multilabel Cyberbully, Sexual Harassment, Threat and Spam Dataset](https://data.mendeley.com/datasets/sz5558wrd4/2)
   - [Kaggle — Bengali Cyber Bullying Dataset](https://www.kaggle.com/datasets/moshiurrahmanfaisal/bangla-cyber-bullying-dataset)

3. **For multimodal sarcasm benchmark awareness**
   - [MMSD2.0 paper](https://aclanthology.org/2023.findings-acl.689/)
   - [MMSD2.0 dataset repository](https://github.com/joeying1019/mmsd2.0)

### To Watch
1. [FreeCodeCamp — Data Science full course](https://www.youtube.com/watch?v=r-uOLxNrNk8)
   - Watch only the parts needed for:
     - problem framing
     - data preprocessing
     - evaluation pipeline

2. Optional overview:
   - [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

### To Do
1. Compare 2–3 Bangla cyberbullying datasets.
2. Select one dataset as the **main dataset**.
3. Decide the task setup:
   - **Binary classification first**: bullying vs non-bullying
   - multiclass only if time remains
4. Draft a working title, for example:
   - **Bangla Cyberbullying Detection using Transformer Baselines and Error Analysis**
5. Write 2–3 research questions, such as:
   - Which model works best for Bangla cyberbullying detection?
   - How much better are transformer models than classical ML baselines?
   - Which error categories remain difficult?

### Code to Implement
In `01_data_loading_and_baselines.ipynb`:
- load the chosen dataset
- inspect class distribution
- remove duplicates
- handle missing rows
- split into train/validation/test
- save processed CSV files

Implement the first serious baseline:
- **TF-IDF + Logistic Regression**

### Expected Outcomes
By the end of Day 2, I should have:
- one final dataset selected
- a specific paper title
- a cleaned and processed dataset
- a working TF-IDF + Logistic Regression baseline

---

## Day 03 — Build the Main Classical Baseline Models

### To Read
1. Revisit metrics:
   - [Google ML Crash Course — Accuracy, Precision, Recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)

2. Revisit transformer intuition:
   - [Hugging Face — How do Transformers work?](https://huggingface.co/learn/llm-course/en/chapter1/4)

### To Watch
From the existing list, only use what is directly needed for implementation:

- [Naive Bayes](https://youtu.be/8vv9julkQEA?si=yE7_boraLqzaWoZH)
- [Decision Tree Implementation](https://youtu.be/PHxYNGo8NcI?si=lldNKJKZaZGZQx6D)
- [Random Forest Basics](https://youtu.be/gkXX4h3qYm4?si=bkULumqFr7zUQObx)
- [Random Forest Implementation](https://youtu.be/ok2s1vV9XW0?si=XjXDX7PjKUW_M_ex)
- [SVM Basics](https://youtu.be/NDqACjz5j8g?si=niJ3vNsVK_PaUnhL)
- [SVM Implementation](https://youtu.be/FB5EdxAGxQg?si=Tye3KLmUG99Em5tF)

### To Do
Implement and compare these classical models:
- TF-IDF + Logistic Regression
- TF-IDF + Naive Bayes
- TF-IDF + SVM
- TF-IDF + Random Forest

For each model, record:
- accuracy
- macro F1
- weighted F1
- precision
- recall
- confusion matrix

### Code to Implement
Create a reusable training/evaluation pipeline:
- TF-IDF vectorizer
- training loop for multiple models
- evaluation table
- confusion matrix plotting

Save outputs to:
- `results/classical_baselines.csv`

### Expected Outcomes
By the end of Day 3, I should have:
- 3–4 working classical baseline models
- a clean comparison table
- one best classical baseline
- enough material to write the classical experiments section

---

## Day 04 — Build the Deep Learning / Transformer Model

### To Read
1. Hugging Face fundamentals:
   - [Hugging Face — Transformers, what can they do?](https://huggingface.co/learn/llm-course/chapter1/3)
   - [Hugging Face — How do Transformers work?](https://huggingface.co/learn/llm-course/en/chapter1/4)

2. If needed, revisit NLP basics:
   - [NLP intro](https://youtu.be/fLvJ8VdHLA0?si=QgK5-wWPDIvVBS9o)

### To Watch
Use RNN/LSTM videos only for background, not as the main work of the day:
- [RNN explained](https://youtu.be/AsNTP8Kwu80?si=HSkM9MOoSNQlpbNT)
- [RNN implementation](https://youtu.be/0_PgWWmauHk?si=xC67ecbKAze2FD13)
- [LSTM](https://youtu.be/b61DPVFX03I?si=2R83brFKALCbcBLZ)

### To Do
Train at least one transformer-based classifier.

Recommended options:
- multilingual BERT
- XLM-RoBERTa
- Bangla-specific BERT if easily available

The goal is **not** to try too many models. The goal is to get:
- one strong classical baseline
- one strong transformer baseline

### Code to Implement
Create a new notebook:
- `02_transformer_baseline.ipynb`

Implement:
- tokenizer
- input encoding
- transformer-based sequence classification
- validation loop
- early stopping
- checkpoint saving

Save outputs to:
- `models/best_transformer_model/`
- `results/transformer_results.csv`

### Expected Outcomes
By the end of Day 4, I should have:
- one working transformer model
- comparison against classical baselines
- a strong modern baseline for the paper

---

## Day 05 — Start the Paper Properly and Begin the Thesis Track

### To Read
1. Re-read the reference sarcasm paper:
   - [A multi-modal sarcasm detection model based on cue learning](https://www.nature.com/articles/s41598-025-94266-w.pdf)

2. CLIP:
   - [OpenAI CLIP repository](https://github.com/openai/CLIP)

3. Prompt tuning / soft prompts:
   - [Hugging Face PEFT — Prompt tuning](https://huggingface.co/docs/peft/main/en/package_reference/prompt_tuning)
   - [Hugging Face PEFT — Soft prompts overview](https://huggingface.co/docs/peft/main/en/conceptual_guides/prompting)

### To Watch
From the current resource list:
- [Resource 1](https://youtu.be/5HQCNAsSO-s?si=pzWjxyfnoiplFPse)
- [Resource 2](https://youtu.be/4YGkfAd2iXM?si=_RFMwVQvnLreFPbg)
- [Resource 3](https://youtu.be/JgnbwKnHMZQ?si=VGkSGGG79Fq4R88F)

### To Do
Split the day into two tracks.

#### Main Paper Track
Start writing these sections:
- Abstract
- Introduction
- Related Work
- Dataset
- Methodology

#### Thesis / Sarcasm Track
Only do the realistic minimum today:
- understand CLIP
- understand prompt / cue learning
- inspect MMSD2.0 structure
- prepare a starter notebook for sarcasm work

### Code to Implement
#### Main Paper
- save final predictions from the best text models
- generate confusion matrix and class-wise metrics

#### Sarcasm Track
Create:
- `03_mmsd2_clip_baseline.ipynb`

Implement:
- dataset loading
- sample inspection of text-image pairs
- note modality-related fields
- CLIP embedding extraction starter code

### Expected Outcomes
By the end of Day 5, I should have:
- the first half of the cyberbullying paper drafted
- a real conceptual understanding of CLIP and prompt tuning
- a starter notebook for multimodal sarcasm work

---

## Day 06 — Improve Experiments, Run Ablations, and Do Error Analysis

### To Read
1. Reproducibility / reporting checklist:
   - [ACL checklist PDF](https://aclanthology.org/attachments/2025.findings-emnlp.1404.checklist.pdf)

2. Experiment tracking:
   - [Weights & Biases quickstart](https://docs.wandb.ai/get-started)

### To Watch
No heavy new videos today.
Only use short targeted videos if I get blocked in implementation.

### To Do
For the **cyberbullying paper**, do the following:

1. Compare:
- best classical baseline
- best transformer baseline

2. Run at least 2 additional checks, such as:
- with preprocessing vs without preprocessing
- class weighting vs no class weighting
- smaller vs larger max sequence length
- binary vs multiclass if possible

3. Perform error analysis:
- examples the model gets right
- examples it gets wrong
- slang / abusive words / spelling variation issues
- ambiguous or borderline cases

4. Write these sections:
- Results
- Error Analysis
- Limitations

For the **sarcasm thesis track**:
- write a summary of the cue-learning paper in my own words
- list 3 possible extension directions

### Code to Implement
For cyberbullying:
- ablation result table
- error analysis export
- sample misclassification analysis cells

For sarcasm:
- simple CLIP embedding extraction test on a few examples if feasible

### Expected Outcomes
By the end of Day 6, I should have:
- final model comparison results
- a solid error analysis section
- an almost complete cyberbullying paper draft
- a much stronger understanding of how to extend the sarcasm thesis work

---

## Day 07 — Finalize the Paper and Package Everything Cleanly

### To Read
1. Re-read the checklist:
   - [ACL checklist PDF](https://aclanthology.org/attachments/2025.findings-emnlp.1404.checklist.pdf)

2. Revisit the cyberbullying dataset paper for framing if needed:
   - [Bengali cyberbullying detection: A comprehensive dataset for advanced text analysis](https://www.sciencedirect.com/science/article/pii/S2352340925009266)

### To Watch
No major new video today.
Today is for writing, polishing, and packaging.

### To Do
Finalize the cyberbullying paper in this order:
1. Title
2. Abstract
3. Introduction
4. Related Work
5. Dataset
6. Methodology
7. Experimental Setup
8. Results
9. Error Analysis
10. Limitations
11. Conclusion

Prepare:
- result tables
- confusion matrix figure
- model comparison figure
- appendix / supplementary notes if needed

Package the project for reproducibility:
- `README.md`
- `requirements.txt`
- clean notebook names
- saved result files
- saved best model

For the sarcasm track, prepare a mini deliverable:
- 2–3 page summary of the cue-learning paper
- proposed thesis extension idea
- dataset options and risks
- starter notebook / pseudocode pipeline

### Code to Implement
Final cleanup:
- export best results to CSV
- save figures as PNG
- save model checkpoint
- save test predictions

### Expected Outcomes
By the end of Day 7, I should have:
- a complete cyberbullying paper draft
- working classical ML and transformer models
- tables and figures ready
- a clean and reproducible folder / repo structure
- a thesis-ready understanding of multimodal sarcasm detection
- a small starter pipeline for the sarcasm thesis work

---

## What Exactly to Code During the Week

### For the Main Cyberbullying Paper
Implement these in order:
1. Data loading and cleaning
2. Train/validation/test split
3. TF-IDF + Logistic Regression
4. TF-IDF + Naive Bayes
5. TF-IDF + SVM
6. One transformer fine-tuning pipeline
7. Evaluation metrics:
   - accuracy
   - precision
   - recall
   - macro F1
   - weighted F1
   - confusion matrix
8. Error analysis export
9. Final figures and result tables

### For the Sarcasm Thesis Track
Implement only the realistic minimum this week:
1. Read and summarize the cue-learning paper
2. Load a reliable dataset like MMSD2.0
3. Inspect text-image pairs
4. Build a simple CLIP embedding extraction notebook
5. Write down how cue learning / prompt tuning would be added next

---

## What Not to Waste Time On This Week

Do **not** spend the week trying to:
- complete all of 100 Days of ML
- deeply study every classical ML algorithm mathematically
- build a transformer from scratch
- build a large Bangla sarcasm dataset manually
- scrape many random websites without a proper labeling plan
- fully master RNN/LSTM theory

The real priority is:
- one good dataset
- one clean baseline suite
- one solid transformer model
- one complete paper draft
- one strong thesis-aligned starter track

---

## Suggested Final Output After 7 Days

### Paper 01 (Main)
**Bangla Cyberbullying Detection using Classical ML and Transformer Baselines**

Should include:
- dataset description
- preprocessing
- classical ML baselines
- transformer baseline
- evaluation table
- confusion matrix
- error analysis
- limitations

### Paper / Thesis Track 02 (Starter)
**Multimodal Sarcasm Detection with CLIP-based Cue Learning**

Should include:
- paper summary
- understanding of CLIP and cue learning
- dataset shortlist
- starter notebook
- next-step implementation plan

---

## Notes

- Bangla multimodal sarcasm detection is still a strong thesis idea, but it is riskier for a 7-day deadline because data collection and annotation are much harder.
- Bangla cyberbullying detection is the best short-term publication path.
- If extra time remains after Day 7, the next best step is to strengthen the sarcasm thesis track using an English benchmark first, then attempt a Bangla/Banglish pilot extension.

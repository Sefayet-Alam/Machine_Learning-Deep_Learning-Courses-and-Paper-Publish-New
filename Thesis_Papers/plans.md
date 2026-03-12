# 7-Day Intensive Execution Plan (Notebook-First Workflow)

## Goal
By the end of Day 7, I should have:

- a working cyberbullying detection model
- a working multimodal sarcasm detection model
- a full draft paper for cyberbullying detection
- a strong thesis/paper draft for sarcasm detection
- enough understanding to explain the datasets, models, metrics, experiments, and design choices clearly


---

## Reference Papers

### Cyberbullying Detection Paper
https://link.springer.com/chapter/10.1007/978-3-031-12638-3_8

### Multimodal Sarcasm Detection Paper
https://www.nature.com/articles/s41598-025-94266-w?utm_source=chatgpt.com#Sec9

---

## Courses Already Completed

### Completed
- CampusX 100 Days ML  
  https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH

- freeCodeCamp Data Science  
  https://www.youtube.com/watch?v=r-uOLxNrNk8

### Planned but not completed
- CampusX Deep Learning Playlist  
  https://www.youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn

- Google ML Crash Course  
  https://developers.google.com/machine-learning/crash-course

- Deep Learning Full Course  
  https://www.youtube.com/watch?v=5HQCNAsSO-s

- Twitter Sentiment Analysis  
  https://www.youtube.com/watch?v=4YGkfAd2iXM

- RNN Sentiment Analysis  
  https://www.youtube.com/watch?v=JgnbwKnHMZQ

---

## Extra Resources To Use This Week

### NLP / Transformers / Practical Fine-Tuning
- Text Classification with Hugging Face Transformers  
  https://www.youtube.com/watch?v=VM5ex48VNCM

### Multimodal / CLIP
- Hugging Face CLIP Introduction  
  https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/clip-and-relatives/Introduction

- CMU Multimodal ML Playlist  
  https://www.youtube.com/playlist?list=PLTLz0-WCKX616TjsrgPr2wFzKF54y-ZKc

### Prompt Learning
- Prompt Tuning Tutorial  
  https://www.youtube.com/watch?v=nPbRCubUN7A

### Core ML Revision
- Google ML Crash Course  
  https://developers.google.com/machine-learning/crash-course

### Notebook Best Practices
- Jupyter Notebook documentation  
  https://jupyter-notebook.readthedocs.io/en/stable/

---

## Weekly Rules

- Do not restart full ML/DL courses from the beginning.
- Only study the parts directly needed for the two papers.
- Every day must produce:
  - one learning output
  - one coding output
  - one writing output
- Watch videos actively and take notes.
- Implement immediately after watching.
- Save every result, even if accuracy is bad.
- Write a little every day.
- Do not leave all writing for Day 7.
- Use **one notebook per major task**, not one giant notebook for everything.
- Add markdown headers inside each notebook.
- At the end of each notebook, save outputs into `results/`.

---

## Project Folder Structure To Create On Day 1

```bash
project/
├── cyberbullying_paper/
│   ├── data/
│   ├── notebooks/
│   │   ├── 01_data_loading.ipynb
│   │   ├── 02_baseline_model.ipynb
│   │   ├── 03_weighted_experiments.ipynb
│   │   ├── 04_error_analysis.ipynb
│   │   └── 05_results_tables.ipynb
│   ├── results/
│   ├── paper/
│   ├── models/
│   └── progress.md
├── sarcasm_thesis/
│   ├── data/
│   ├── notebooks/
│   │   ├── 01_data_loading.ipynb
│   │   ├── 02_multimodal_baseline.ipynb
│   │   ├── 03_clip_baseline.ipynb
│   │   ├── 04_cue_prompt_model.ipynb
│   │   └── 05_results_tables.ipynb
│   ├── results/
│   ├── paper/
│   ├── models/
│   └── progress.md
└── README.md
```

---

## Suggested Master Notebook List

### Cyberbullying
- `01_data_loading.ipynb`
- `02_baseline_model.ipynb`
- `03_weighted_experiments.ipynb`
- `04_error_analysis.ipynb`
- `05_results_tables.ipynb`

### Sarcasm
- `01_data_loading.ipynb`
- `02_multimodal_baseline.ipynb`
- `03_clip_baseline.ipynb`
- `04_cue_prompt_model.ipynb`
- `05_results_tables.ipynb`

---

## Day 1 — Understand the Papers + Set Up the Environment

### Main Goal
Understand both papers clearly and set up everything needed for the week.

### Topics To Cover
- train / validation / test split
- binary vs multiclass classification
- accuracy, precision, recall, F1
- confusion matrix
- overfitting
- transformer basics
- multimodal learning basics
- research paper structure
- notebook organization and experiment tracking

### What To Read

#### Google ML Crash Course
Read only these modules:
- Classification  
  https://developers.google.com/machine-learning/crash-course/classification
- Datasets, Generalization, and Overfitting  
  https://developers.google.com/machine-learning/crash-course/overfitting
- Neural Networks  
  https://developers.google.com/machine-learning/crash-course/neural-networks
- Embeddings  
  https://developers.google.com/machine-learning/crash-course/embeddings
- Intro to Large Language Models  
  https://developers.google.com/machine-learning/crash-course/large-language-models

#### Papers
Read these sections from both reference papers:
- Abstract
- Introduction
- Methodology / Proposed Method
- Experimental Setup
- Results / Discussion
- Conclusion

### What To Watch
- Text Classification with Hugging Face Transformers  
  https://www.youtube.com/watch?v=VM5ex48VNCM

- Twitter Sentiment Analysis  
  https://www.youtube.com/watch?v=4YGkfAd2iXM

### What To Implement

#### Environment Setup
Install the following:
- torch
- torchvision
- transformers
- datasets
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- jupyter
- notebook
- ipykernel
- pillow
- opencv-python
- tqdm

#### Notebook To Create
Create these first notebooks:
- `cyberbullying_paper/notebooks/01_data_loading.ipynb`
- `sarcasm_thesis/notebooks/01_data_loading.ipynb`

#### What To Do In The Notebook
- load a CSV dataset
- show first 5 rows
- show missing values
- show class distribution
- perform train/val/test split
- save cleaned split files
- add markdown cells explaining what was done

### What To Write
Create:
- `cyberbullying_paper/paper/outline.md`
- `sarcasm_thesis/paper/outline.md`

Write in each:
- title idea
- problem definition
- dataset name
- model idea
- evaluation metrics
- expected contribution
- paper section headings

### End-of-Day Deliverables
- full project folders ready
- Python/Jupyter environment ready
- paper outlines written
- two data-loading notebooks created
- both papers understood at high level

---

## Day 2 — Build the First Cyberbullying Baseline

### Main Goal
Train a full text classification baseline for cyberbullying detection.

### Topics To Cover
- text preprocessing
- tokenization
- transformer fine-tuning
- multiclass classification
- label encoding
- F1 score interpretation
- training loops inside notebooks

### What To Read
- cyberbullying paper carefully again  
  https://link.springer.com/chapter/10.1007/978-3-031-12638-3_8

- Hugging Face text classification docs  
  https://huggingface.co/docs/transformers/tasks/sequence_classification

### What To Watch
- Text Classification with Hugging Face Transformers  
  https://www.youtube.com/watch?v=VM5ex48VNCM

- RNN Sentiment Analysis  
  https://www.youtube.com/watch?v=JgnbwKnHMZQ  
  Watch only for intuition, not as final method.

### What To Implement

#### Notebook To Create
- `cyberbullying_paper/notebooks/02_baseline_model.ipynb`

#### Recommended Steps In The Notebook
1. load dataset
2. inspect labels
3. perform minimal cleaning
4. encode labels
5. tokenize text
6. fine-tune model
7. evaluate on validation set
8. save model checkpoint
9. save plots and metrics in `results/`

#### Model Choices
Try in this order:
1. `xlm-roberta-base`
2. another Bangla-friendly transformer only if setup is easy

#### Results To Save
- training loss
- validation accuracy
- validation F1
- classification report
- confusion matrix
- saved model checkpoint

### What To Write
In `cyberbullying_paper/paper/draft.md` write:
- Introduction
- Problem Statement
- Dataset Description
- Baseline Method

### End-of-Day Deliverables
- first trained cyberbullying model
- first notebook with full training flow
- first results table
- first draft sections written

---

## Day 3 — Improve the Cyberbullying Paper and Experiments

### Main Goal
Make the cyberbullying work more paper-worthy through comparisons and error analysis.

### Topics To Cover
- class imbalance
- macro F1 vs weighted F1
- error analysis
- misclassification patterns
- ablation study basics
- experiment logging
- organizing multiple notebook runs

### What To Read
- Google ML Crash Course classification module again  
  https://developers.google.com/machine-learning/crash-course/classification

- Google ML Crash Course overfitting module again  
  https://developers.google.com/machine-learning/crash-course/overfitting

### What To Watch
From CampusX Deep Learning playlist, watch only relevant videos on:
- ANN basics
- overfitting
- regularization
- evaluation

Playlist:  
https://www.youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn

### What To Implement

#### Notebooks To Create
- `cyberbullying_paper/notebooks/03_weighted_experiments.ipynb`
- `cyberbullying_paper/notebooks/04_error_analysis.ipynb`

#### Experiments To Run
1. baseline transformer
2. transformer + class weighting
3. transformer + preprocessing variation

#### Preprocessing Variations To Try
- remove URLs
- normalize repeated spaces
- remove excessive punctuation
- compare with and without emoji removal

#### Results To Save
- comparison table of all 3 runs
- best checkpoint
- per-class precision/recall/F1
- 10 misclassified examples with explanation notes

### What To Write
Add to cyberbullying paper:
- Experimental Setup
- Results
- Error Analysis
- Discussion

### End-of-Day Deliverables
- best cyberbullying model selected
- comparison results saved
- draft paper around 60% done
- error cases collected in notebook form

---

## Day 4 — Learn Multimodal Sarcasm Detection Properly

### Main Goal
Understand the sarcasm paper deeply enough to start implementing a practical baseline.

### Topics To Cover
- multimodal learning
- image encoder vs text encoder
- CLIP basics
- embeddings
- fusion methods
- prompt learning
- cue learning
- sarcasm classification setup
- image-text dataset organization in notebooks

### What To Read
- sarcasm paper again  
  https://www.nature.com/articles/s41598-025-94266-w?utm_source=chatgpt.com#Sec9

- Hugging Face CLIP introduction  
  https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/clip-and-relatives/Introduction

### What To Watch
- CMU Multimodal ML Playlist  
  https://www.youtube.com/playlist?list=PLTLz0-WCKX616TjsrgPr2wFzKF54y-ZKc

- Prompt Tuning Tutorial  
  https://www.youtube.com/watch?v=nPbRCubUN7A

### What To Learn Specifically Today
By the end of today I should understand:
- why text-only is not enough for multimodal sarcasm
- what CLIP does
- how text and image embeddings can be combined
- what prompts are
- what cue words are
- what can realistically be implemented in 3 remaining days

### What To Implement

#### Notebooks To Create
- `sarcasm_thesis/notebooks/01_data_loading.ipynb`
- `sarcasm_thesis/notebooks/02_multimodal_baseline.ipynb`

#### Do Not Start With Full Complexity
First implement a simple multimodal baseline:
1. load image-text pairs
2. preprocess images
3. preprocess text
4. extract embeddings
5. combine embeddings
6. classify sarcastic vs non-sarcastic

#### Simple First Fusion
Use:
- concatenation of text embedding + image embedding

### What To Write
In `sarcasm_thesis/paper/draft.md` write:
- Introduction
- Problem Statement
- Related Work

### End-of-Day Deliverables
- clear understanding of CLIP and multimodal sarcasm detection
- basic multimodal data pipeline notebook written
- first thesis sections written

---

## Day 5 — Build the Sarcasm Baseline and a Cue/Prompt Variant

### Main Goal
Get a working sarcasm model and add one realistic improvement inspired by the reference paper.

### Topics To Cover
- prompt templates
- hard prompts vs soft prompts
- cue-aware classification
- comparing modalities
- limited-data experiment setup
- comparing notebook experiments cleanly

### What To Read
- sarcasm paper methodology and experiments again  
  https://www.nature.com/articles/s41598-025-94266-w?utm_source=chatgpt.com#Sec9

- CLIP docs / examples  
  https://huggingface.co/docs/transformers/model_doc/clip

### What To Watch
- Prompt Tuning Tutorial  
  https://www.youtube.com/watch?v=nPbRCubUN7A

- Deep Learning Full Course  
  https://www.youtube.com/watch?v=5HQCNAsSO-s  
  Watch only the parts needed for embeddings, neural nets, and classification.

### What To Implement

#### Notebooks To Create
- `sarcasm_thesis/notebooks/03_clip_baseline.ipynb`
- `sarcasm_thesis/notebooks/04_cue_prompt_model.ipynb`

#### Model A — Baseline
- use CLIP or separate text/image embeddings
- combine embeddings
- train a classifier head

#### Model B — Practical Thesis Variant
Use one realistic improvement:
- sarcasm-aware prompt templates
- simple cue words for sarcastic vs non-sarcastic classes
- compare text-only, image-only, and multimodal
- optionally try small-data subset experiments

#### Recommended Simple Contribution
Use this exact comparison:
1. text-only
2. image-only
3. multimodal baseline
4. multimodal + cue prompts

#### Results To Save
- validation accuracy
- validation F1
- model comparison table
- best sarcasm checkpoint

### What To Write
Add to sarcasm thesis:
- Methodology
- Model Architecture
- Training Details

### End-of-Day Deliverables
- working sarcasm baseline
- one cue/prompt-enhanced version
- methodology section drafted

---

## Day 6 — Run Final Experiments for Both Projects

### Main Goal
Finish the core experiments and collect all results for both papers.

### Topics To Cover
- result interpretation
- model comparison
- limitations
- ablation logic
- how to explain performance gaps
- turning notebook outputs into paper-ready tables

### What To Read
Re-read results/discussion sections of both papers:
- cyberbullying paper  
  https://link.springer.com/chapter/10.1007/978-3-031-12638-3_8

- sarcasm paper  
  https://www.nature.com/articles/s41598-025-94266-w?utm_source=chatgpt.com#Sec9

### What To Watch
- no long videos today
- only short targeted searches if stuck

### What To Implement

#### Cyberbullying Notebook
- update `02_baseline_model.ipynb`
- finalize `03_weighted_experiments.ipynb`
- finalize `04_error_analysis.ipynb`
- create `05_results_tables.ipynb`

#### Sarcasm Notebook
- update `02_multimodal_baseline.ipynb`
- finalize `03_clip_baseline.ipynb`
- finalize `04_cue_prompt_model.ipynb`
- create `05_results_tables.ipynb`

#### Cyberbullying Final Runs
Save:
- best model
- confusion matrix
- per-class metrics
- classification report
- sample wrong predictions

#### Sarcasm Final Runs
Save:
- text-only model results
- image-only model results
- multimodal baseline results
- multimodal + cue prompt results

### What To Write
For both papers, finish drafts of:
- Experimental Setup
- Results
- Discussion
- Limitations
- Conclusion

### End-of-Day Deliverables
- all major experiments completed
- all important tables saved
- all important figures saved
- both drafts around 85% done

---

## Day 7 — Final Writing, Cleanup, and GitHub/Submission Packaging

### Main Goal
Make everything clean, understandable, and ready to show, submit, or continue polishing.

### Topics To Cover
- abstract writing
- contribution writing
- result summarization
- proofreading
- citation cleanup
- final notebook cleanup
- final project packaging

### What To Read
Read both full drafts from top to bottom:
- check if every claim is supported by a result
- check if every metric is defined
- check if every table is readable
- check if methodology is reproducible

### What To Watch
- no major videos today
- only short help videos if needed for:
  - LaTeX
  - formatting
  - plotting
  - bibliography

### What To Implement

#### Final Cleanup
For each project:
- save final model
- save final notebooks
- restart and run all cells in final notebooks
- save requirements.txt
- save README.md
- save metrics tables
- save plots
- save confusion matrix images

#### Files To Prepare
For each project create:
- `README.md`
- `requirements.txt`
- `models/best_model/`
- `paper/final_draft.md`
- `results/metrics.csv`
- `results/confusion_matrix.png`
- `results/sample_predictions.csv`

### What To Write
Finalize for both papers:
- Title
- Abstract
- Keywords
- Introduction cleanup
- Final Discussion
- Future Work
- References
- Proofreading pass

### Final Self-Check
For each paper ask:
- Is the problem clear?
- Is the dataset clear?
- Is the method understandable?
- Are the metrics clearly shown?
- Is my contribution stated clearly?
- Can someone reproduce this from my notebooks and writing?

### End-of-Day Deliverables
- cyberbullying paper complete draft
- sarcasm thesis complete draft
- both trained models saved
- both notebook projects organized for GitHub
- enough understanding to explain every major choice

---

## Daily Time Structure

### Suggested Daily Routine
Morning:
- reading
- notes
- targeted video watching

Afternoon:
- notebook coding
- training
- debugging
- running experiments

Evening:
- paper writing
- results organization
- plotting

Night:
- save outputs
- update `progress.md`
- plan next day

### Suggested Daily Hours
- study / reading: 3 to 4 hours
- coding / experiments: 5 to 6 hours
- writing: 2 to 3 hours
- debugging / cleanup / notes: 1 to 2 hours

---

## Minimum Final Deliverables After 7 Days

### Cyberbullying Project
- one strong transformer-based text classifier
- at least 3 experiments
- metrics table
- confusion matrix
- error analysis section
- full draft paper
- clean notebooks for the full workflow

### Sarcasm Project
- one text-only baseline or quick comparison baseline
- one image-only baseline
- one multimodal baseline
- one cue/prompt-enhanced version
- metrics table
- thesis/paper draft
- clean notebooks for the full workflow

---

## What Not To Do

- do not restart full ML courses from the beginning
- do not watch whole long playlists without purpose
- do not try too many architectures
- do not delay writing until the final day
- do not chase perfect accuracy before having a working pipeline
- do not attempt to reproduce every detail of the Nature paper in this week
- do not put everything inside one giant notebook

---

## Suggested Working Titles

### Cyberbullying Paper
**Transformer-Based Cyberbullying Detection for Bangla Social Media with Comparative Error Analysis**

### Sarcasm Thesis
**Cue-Aware Multimodal Sarcasm Detection Using CLIP-Based Text-Image Representations**

---

## Final Reminder
The goal of this 7-day plan is not to master all of machine learning.

The goal is to finish this week with:
- working models
- paper drafts
- strong understanding of the exact topics needed
- enough confidence to explain and improve the work further

If I follow this properly and work consistently every day, I should end the week with real outputs instead of only partial learning.

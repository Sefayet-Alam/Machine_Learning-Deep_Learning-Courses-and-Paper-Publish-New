# 100 Days of Machine Learning (CampusX) — Interview‑Ready Revision Notes

> **Goal:** a single place to revise the full flow of ML—from problem framing → data → preprocessing → modeling → evaluation → ensembles.  
> **How to use (fast revision):**
> 1) Skim the **Interview Cheat Sheet**.  
> 2) Review the **module notes** (Foundations → Data → FE/Preprocessing → Regression → Classification → Trees/Ensembles).  
> 3) Use the **Video Checklist** to ensure you didn’t skip anything.

---

## Table of contents

- [Mental model of ML](#mental-model-of-ml)
- [Foundations](#foundations-videos-1-14)
- [Data acquisition and EDA](#data-acquisition-and-eda-videos-15-22)
- [Feature engineering and preprocessing](#feature-engineering-and-preprocessing-videos-23-46)
- [Dimensionality reduction](#dimensionality-reduction-videos-47-49)
- [Regression](#regression-videos-50-68)
- [Classification](#classification-videos-69-79)
- [Decision trees](#decision-trees-videos-80-83)
- [Ensembles](#ensembles-videos-84-100)
- [Advanced topics addendum](#advanced-topics-addendum-videos-101-134)
- [Interview cheat sheet](#interview-cheat-sheet)
- [Video checklist (1–100)](#video-checklist-1-100)

---

## Mental model of ML

### What ML is (vs conventional programming)
- **Conventional programming:** rules/logic + input → output.
- **Machine learning:** (input, output) examples + learning algorithm → learned rules/model; then input → predicted output.
- ML is valuable when:
  - The rule set is too complex/nonlinear to hardcode (vision, language).
  - The rules change frequently (spam detection, fraud).
  - You want patterns/insights beyond obvious plots (data mining).

### Core ML workflow (MLDLC, in practice)
1. **Problem framing:** task type, success metric, constraints.
2. **Data:** collect, label, validate, split (train/val/test).
3. **EDA:** understand distribution, missingness, outliers, leakage.
4. **Feature engineering + preprocessing:** clean, encode, scale, transform.
5. **Modeling:** baseline → iterate; use pipelines to avoid leakage.
6. **Evaluation:** proper metric, cross‑validation, error analysis.
7. **Deployment + monitoring:** drift, retraining, feedback loop.

---

## Foundations (Videos 1–14)

### Video 1 — What is Machine Learning?
- ML = learning patterns from data to make predictions/decisions.
- Key idea: training data teaches a mapping \(f(X)\rightarrow y\).
- Interview prompts:
  - When would you *not* use ML?
  - Supervised vs unsupervised vs reinforcement learning use cases.

### Video 2 — AI vs ML vs DL
- **AI:** broad umbrella (systems that exhibit “intelligence”).
- **ML:** subset of AI, learns from data.
- **DL:** subset of ML, neural networks with many layers (tensors + backprop).
- Practical takeaway: DL often needs more data/compute but can learn features.

### Video 3 — Types of ML (beginner → in depth)
- **Supervised:** labeled data (classification/regression).
- **Unsupervised:** no labels (clustering, dimensionality reduction).
- **Reinforcement learning:** agent learns via reward signals.
- Also common taxonomy:
  - **Batch vs online**
  - **Instance-based vs model-based**
  - **Parametric vs non‑parametric**

### Video 4 — Batch ML / Offline vs Online
- **Batch (offline):** train on entire dataset; deploy; retrain periodically.
- Pros: simpler, stable. Cons: slower to adapt, retraining cost.

### Video 5 — Online learning
- Model updates incrementally as new data arrives (streaming).
- Pros: adapts to drift; works with big/streaming data.
- Risks: feedback loops, noisy updates, model poisoning, concept drift.

### Video 6 — Instance-based vs Model-based
- **Instance-based:** store data; predict via similarity (kNN).
- **Model-based:** learn parameters/structure (linear/logistic regression, trees).
- Tradeoff: memory + inference time vs generalization + compactness.

### Video 7 — Challenges / problems in ML
- **Data quality:** missing values, outliers, noise, wrong labels.
- **Bias/variance:** underfit vs overfit.
- **Data leakage:** training sees future/target information.
- **Nonstationarity:** concept drift.
- **Interpretability, fairness, privacy, compute constraints.**

### Video 8 — Applications of ML
- Computer vision, NLP, recommendation, fraud detection, forecasting, anomaly detection, medical, manufacturing.
- Interview angle: pick **one domain** and explain full pipeline + metrics.

### Video 9 — Machine Learning Development Life Cycle (MLDLC)
- Pipeline thinking + iterations + monitoring.
- Always anchor on:
  - objective/metric
  - data strategy
  - baseline
  - validation design
  - monitoring and retraining

### Video 10 — Roles: Data Engineer vs Analyst vs Data Scientist vs ML Engineer
- **Data Engineer:** pipelines, storage, ETL/ELT, reliability.
- **Analyst:** BI, dashboards, descriptive insights.
- **Data Scientist:** experiments, modeling, insights + prototyping.
- **ML Engineer:** deploy, scale, monitor models, MLOps.

### Video 11 — Tensors
- Tensor = n‑dimensional array (scalar/1D/2D/3D…).
- In ML:
  - features as vectors/matrices
  - batches as higher‑dim tensors
- Interview: be clear about **shapes** (batch_size, features, channels, time steps).

### Video 12 — Setup: Anaconda, Jupyter, Colab
- Environments prevent dependency chaos.
- Reproducibility: `requirements.txt` / `conda env export`.
- Colab = quick GPU + shareable notebooks (but ephemeral storage).

### Video 13 — End-to-end toy project
A minimal ML project typically includes:
- Define target, load data, train/test split
- Preprocess (missing, encoding, scaling)
- Train baseline model
- Evaluate + iterate

### Video 14 — Framing an ML problem
- Identify:
  - task: classification/regression/ranking/forecasting
  - label definition + how labels are obtained
  - metric: aligned with business
  - constraints: latency, interpretability, fairness, cost
  - baseline and feasibility check

---

## Data acquisition and EDA (Videos 15–22)

### Video 15 — Working with CSV files
- Pandas essentials: `read_csv`, parsing dates, separators, encodings, chunking.
- Common issues: bad headers, NA markers, dtype inference errors.
- Tip: always start with:
  - `df.shape`, `df.head()`, `df.info()`, `df.describe(include='all')`

### Video 16 — Working with JSON / SQL
- JSON: nested structures → normalize (`json_normalize`) or custom parsing.
- SQL: push filters/aggregations to the database when possible.
- Interview: explain when you’d rather use SQL than pandas.

### Video 17 — Fetching data from an API
- Requests, auth tokens, pagination, rate limits, retries.
- Convert JSON response to DataFrame.
- Always log request params + response status for debugging.

### Video 18 — Web scraping to DataFrame
- Use `requests` + `BeautifulSoup` (static pages); Selenium for dynamic pages.
- Be mindful of robots.txt, terms of service, and rate limiting.

### Video 19 — Understanding your data
- Know:
  - feature types (num/cat/date/text)
  - missingness patterns
  - target distribution + imbalance
  - leakage risks
- Create a data dictionary (feature meaning, units, source, allowed ranges).

### Video 20 — EDA (Univariate)
- Numeric: histogram, KDE, boxplot; check skewness, heavy tails.
- Categorical: value counts, bar plots.
- Target: baseline rate + class balance.

### Video 21 — EDA (Bivariate + Multivariate)
- Numeric–numeric: scatter, correlation (Pearson/Spearman), nonlinearity.
- Cat–num: box/violin plots, group means.
- Cat–cat: contingency tables, chi-square intuition.
- Multivariate: pairplots, heatmaps, interactions.

### Video 22 — Pandas Profiling
- Automated EDA report (missingness, correlations, distributions).
- Use as a starting point, not a replacement for reasoning.

---

## Feature engineering and preprocessing (Videos 23–46)

### Video 23 — What is Feature Engineering?
- Turning raw data into signals that make learning easier.
- Typical moves:
  - transformation (log, sqrt)
  - aggregation (counts, rolling means)
  - interaction terms (ratios, products)
  - domain encodings (time since last event)

### Video 24 — Standardization
- \(z = \frac{x-\mu}{\sigma}\)
- Helps algorithms sensitive to scale (kNN, SVM, linear models, neural nets).
- Fit scaler on **train only**, apply to val/test (pipeline).

### Video 25 — Normalization / MinMax / MaxAbs / RobustScaling
- MinMax: \(x'=\frac{x-x_{min}}{x_{max}-x_{min}}\)
- Robust: uses median + IQR → resistant to outliers.
- Pick based on outlier presence and model requirements.

### Video 26 — Ordinal encoding + Label encoding
- **Ordinal**: when categories have order (low/med/high).
- **Label encoding** for target labels, not usually for nominal features.

### Video 27 — One hot encoding
- Nominal categories → binary columns.
- Watch out for:
  - high cardinality → blow-up
  - unseen categories in test → use `handle_unknown='ignore'` in sklearn

### Video 28 — ColumnTransformer
- Apply different preprocessing to different columns.
- The “correct” sklearn way to mix numeric/categorical transforms.

### Video 29 — Pipelines A–Z
- Why pipelines matter:
  - prevent leakage
  - reproducibility
  - cross-validation correctness
- Pattern:
  - `Pipeline([('preprocess', ct), ('model', clf)])`

### Video 30 — FunctionTransformer (log/reciprocal/sqrt)
- Use transforms for skewed distributions.
- Log transform example:
  - `log1p(x)` handles zeros better than `log(x)`.

### Video 31 — PowerTransformer (Box‑Cox / Yeo‑Johnson)
- Gaussianize features (help linear models).
- Box‑Cox requires positive values; Yeo‑Johnson supports zeros/negatives.

### Video 32 — Binning and Binarization
- Discretization can help:
  - handle nonlinearity
  - reduce noise
  - make models more interpretable
- Methods:
  - uniform / quantile bins
  - kmeans binning

### Video 33 — Mixed variables
- Variables that contain both numeric and categorical info (e.g., “A12”, “B03”).
- Techniques:
  - split into prefix + number
  - parse into multiple features

### Video 34 — Date and time variables
- Common features:
  - year, month, day, weekday
  - hour, is_weekend, holiday flag
  - time since event
- Beware leakage: future timestamps.

### Video 35 — Missing data (Part 1): Complete case analysis
- Drop rows with missing values.
- Works only if missingness is small and **MCAR** (missing completely at random).

### Video 36 — Missing numerical data: SimpleImputer
- Mean/median imputation.
- Median is robust to outliers.
- Always evaluate: imputation can shrink variance.

### Video 37 — Missing categorical data
- Most-frequent imputation.
- “Missing” category as its own level can carry signal.

### Video 38 — Missing indicator + Random sample imputation
- Missingness itself can be predictive.
- Add `is_missing` features.
- Random sample preserves distribution but adds variance.

### Video 39 — KNN imputer (multivariate)
- Uses similarity between rows to fill missing values.
- Needs scaling; can be slow; careful with leakage.

### Video 40 — Iterative imputer / MICE
- Model each feature with missing values as a function of other features (iterative).
- Good when relationships are learnable; more computational cost.

### Video 41 — Outliers
- Outlier ≠ error always (could be rare but valid).
- Decide using domain knowledge + impact on model/metric.

### Video 42 — Z‑score method
- Use standard deviations from mean; works for roughly normal distributions.
- Sensitive to skew/outliers (mean and std get affected).

### Video 43 — IQR method
- IQR = Q3 − Q1; typical rule: outside [Q1−1.5IQR, Q3+1.5IQR]
- More robust than z-score.

### Video 44 — Percentile / Winsorization
- Cap values at chosen percentiles (e.g., 1st and 99th).
- Good for heavy-tailed distributions.

### Video 45 — Feature construction & splitting
- Construct: combine features (ratios, interactions).
- Split: separate features into multiple (e.g., “Cabin” → deck/number/side).

### Video 46 — Curse of dimensionality
- Distance-based methods degrade in high dimensions.
- Sparsity increases; need:
  - feature selection
  - dimensionality reduction (PCA)
  - regularization

---

## Dimensionality reduction (Videos 47–49)

### Video 47 — PCA Part 1: Geometric intuition
- PCA finds orthogonal directions of maximum variance (principal components).
- Equivalent to rotating axes to capture most variance early.
- Used for:
  - compression
  - de-noising
  - speedups
  - visualization (2D/3D)

### Video 48 — PCA Part 2: Formulation
- Steps:
  1. Standardize features (usually)
  2. Compute covariance matrix
  3. Eigen decomposition → eigenvectors (components), eigenvalues (variance)
- Explained variance ratio tells how much variance each component captures.

### Video 49 — PCA Part 3: Code + visualization
- `sklearn.decomposition.PCA(n_components=...)`
- Use `explained_variance_ratio_` and cumulative variance plot.
- Interview: **PCA is unsupervised**—can drop predictive signal if variance ≠ usefulness.

---

## Regression (Videos 50–68)

### Video 50 — Simple Linear Regression (intuition + code)
- Model: \(\hat y = \beta_0 + \beta_1 x\)
- Learn line that minimizes squared error.
- Key assumptions (in classic stats view): linearity, independence, homoscedasticity, normal residuals.

### Video 51 — Simple Linear Regression (math + scratch)
- Closed form:
  - \(\beta_1 = \frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2}\)
  - \(\beta_0 = \bar y - \beta_1\bar x\)

### Video 52 — Regression metrics (MSE/MAE/RMSE/R²/Adj R²)
- MAE: robust; linear penalty.
- MSE/RMSE: penalize big errors more.
- R²: fraction of variance explained (can be negative on test).
- Adjusted R²: penalizes adding useless features.

### Video 53 — Multiple Linear Regression (intuition)
- \(\hat y = \beta_0 + \beta_1 x_1 + ... + \beta_p x_p\)
- Geometric: hyperplane in p‑dim space.
- Multicollinearity impacts coefficient stability.

### Video 54 — Multiple Linear Regression (math, from scratch)
- Normal equation:
  - \(\beta = (X^T X)^{-1}X^T y\)  (if invertible)

### Video 55 — Multiple Linear Regression (code from scratch)
- Implement normal equation + intercept handling.
- In practice, use sklearn + regularization for stability.

### Video 56 — Gradient descent from scratch (end‑to‑end)
- Iterative optimization:
  - \(\theta := \theta - \alpha \nabla_\theta J(\theta)\)
- Need learning rate \(\alpha\); watch divergence.
- Use feature scaling for faster convergence.

### Video 57 — Batch Gradient Descent
- Uses all samples per update.
- Stable but slow on large datasets.

### Video 58 — Stochastic Gradient Descent (SGD)
- Updates per single sample.
- Noisy updates; can escape shallow minima; faster.
- Needs learning rate schedule.

### Video 59 — Mini‑batch Gradient Descent
- Best of both worlds; common default in deep learning.

### Video 60 — Polynomial regression
- Add polynomial features: \(x, x^2, x^3, ...\)
- Still linear in parameters; nonlinear in inputs.
- Risk of overfitting → regularization helps.

### Video 61 — Bias‑variance trade‑off
- **High bias:** underfit; too simple.
- **High variance:** overfit; too complex.
- Fixes:
  - more data
  - regularization
  - simpler model
  - better features
  - cross‑validation

### Video 62 — Ridge regression Part 1 (intuition + code)
- L2 regularization:
  - minimize \(RSS + \lambda \sum \beta_j^2\)
- Shrinks coefficients; helps multicollinearity; reduces variance.

### Video 63 — Ridge Part 2 (math + scratch)
- Closed form:
  - \(\beta = (X^T X + \lambda I)^{-1}X^T y\)

### Video 64 — Ridge Part 3 (gradient descent)
- Update includes penalty term.
- Regularization strength \(\lambda\) is hyperparameter.

### Video 65 — Ridge Part 4 (5 key points)
- Doesn’t set coefficients exactly to zero (usually).
- Good default for linear regression with many correlated features.

### Video 66 — Lasso regression
- L1 regularization:
  - \(RSS + \lambda \sum |\beta_j|\)
- Can create sparsity (feature selection).

### Video 67 — Why Lasso creates sparsity
- L1 constraint geometry (diamond) makes optimum hit axes → exact zeros.

### Video 68 — ElasticNet
- Mix of L1 + L2.
- Useful when:
  - many correlated features
  - want sparsity but stable selection

---

## Classification (Videos 69–79)

### Video 69 — Logistic Regression Part 1 (Perceptron trick)
- Linear classifier: \(z=w^Tx+b\).
- Perceptron uses sign of \(z\); logistic uses sigmoid to get probability.

### Video 70 — Logistic Regression Part 2 (Perceptron trick code)
- Implement linear decision boundary; understand separating hyperplane.

### Video 71 — Logistic Regression Part 3 (Sigmoid)
- Sigmoid:
  - \(\sigma(z)=\frac{1}{1+e^{-z}}\)
- Converts logits to probabilities; threshold (0.5) is adjustable.

### Video 72 — Logistic Regression Part 4 (Loss / MLE / Binary Cross‑Entropy)
- Bernoulli likelihood → negative log likelihood = cross entropy.
- Loss for one sample:
  - \(L = -[y\log(p) + (1-y)\log(1-p)]\)

### Video 73 — Derivative of sigmoid
- \(\sigma'(z)=\sigma(z)(1-\sigma(z))\)
- Shows why gradients can vanish when probabilities saturate.

### Video 74 — Logistic Regression Part 5 (Gradient descent + scratch)
- Use gradient descent to fit \(w, b\).
- Scaling helps; regularization helps generalization.

### Video 75 — Classification metrics Part 1 (Accuracy + Confusion Matrix + Type I/II)
- Confusion matrix: TP/FP/TN/FN.
- Type I error = false positive; Type II = false negative.
- Accuracy can be misleading on imbalanced data.

### Video 76 — Precision, Recall, F1
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = harmonic mean of precision & recall.
- Choose metric based on cost of FP vs FN.

### Video 77 — Softmax regression (multinomial logistic)
- For K classes:
  - \(p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}\)
- Cross-entropy generalizes.

### Video 78 — Polynomial features in logistic regression (nonlinear boundaries)
- Add polynomial features to linear model → nonlinear decision boundary.
- Alternative: kernels or tree-based models.

### Video 79 — Logistic regression hyperparameters
- Regularization strength `C` (inverse of λ), penalty (L1/L2), solver, class_weight.
- Use cross-validation to tune.

---

## Decision trees (Videos 80–83)

### Video 80 — Decision trees intuition (entropy/gini/info gain)
- Tree splits feature space to create pure nodes.
- Impurity:
  - Entropy: \(-\sum p_k\log p_k\)
  - Gini: \(1-\sum p_k^2\)
- Information gain = parent impurity − weighted child impurity.

### Video 81 — Tree hyperparameters + overfitting
- Depth, min_samples_split, min_samples_leaf, max_features, etc.
- Trees overfit easily → prune / limit depth.

### Video 82 — Regression trees
- Split to minimize variance / MSE in leaves.
- Prediction = mean target in leaf.

### Video 83 — dtreeviz visualization
- Visualize splits, class distributions, feature importance.
- Helps with interpretability/debugging.

---

## Ensembles (Videos 84–100)

### Video 84 — Intro to ensemble learning
- Combine weak/varied models to improve generalization.
- Key idea: reduce variance (bagging) and/or bias (boosting).

### Video 85 — Voting ensemble Part 1
- Hard voting (majority class) vs soft voting (avg probabilities).
- Works best when base models are diverse and reasonably strong.

### Video 86 — Voting ensemble Part 2 (classification)
- Soft voting needs calibrated probabilities (or models that output good probs).

### Video 87 — Voting ensemble Part 3 (regression)
- Average predictions, possibly weighted average.

### Video 88 — Bagging Part 1
- Bootstrap samples + train many models in parallel.
- Reduces variance (esp. for high-variance learners like trees).

### Video 89 — Bagging Part 2 (bagging classifiers)
- Each model sees a bootstrap sample; aggregate predictions.
- OOB samples give internal validation.

### Video 90 — Bagging Part 3 (bagging regressor)
- Average predictions of regressors; improves stability.

### Video 91 — Random forest intro
- Bagging + feature randomness at split time.
- Strong default for tabular data.

### Video 92 — Why random forest works so well (bias-variance)
- Feature subsampling decorrelates trees → stronger variance reduction.

### Video 93 — Bagging vs Random Forest
- RF = bagging + random feature selection.
- Helps create diversity across trees.

### Video 94 — Random forest hyperparameters
- n_estimators, max_depth, max_features, min_samples_leaf, bootstrap, etc.

### Video 95 — Hyperparameter tuning RF (GridSearchCV / RandomizedSearchCV)
- Grid is exhaustive; Randomized covers space efficiently.
- Use cross-validation; keep pipeline to avoid leakage.

### Video 96 — OOB score (out-of-bag)
- Evaluate each sample on trees where it was OOB.
- Internal validation without separate holdout (still keep test set).

### Video 97 — Feature importance in trees/RF
- Impurity-based importance can be biased (toward high-cardinality features).
- Alternative: permutation importance, SHAP.

### Video 98 — AdaBoost geometric intuition
- Boosting focuses more on hard-to-classify points.
- Combines weak learners sequentially.

### Video 99 — AdaBoost step-by-step
- Maintain sample weights; misclassified points get higher weight next round.
- Final prediction is weighted vote of weak learners.

### Video 100 — AdaBoost code from scratch
- Implement:
  - weighted error
  - learner weight (alpha)
  - update sample weights
  - final ensemble prediction

---

## Advanced topics addendum (Videos 101–134)

> **Important note:** I could only retrieve the official playlist listing up to **Video 100** within this session’s accessible sources.  
> The playlist itself contains **134 videos**, so there are **34 more videos (101–134)** beyond AdaBoost.  
> I’m including this addendum so you still have interview-ready notes for the most likely “next topics” that typically follow AdaBoost in this course family (boosting extensions, SVM, clustering, etc.).  
> If you paste the titles of videos 101–134, I can map this addendum precisely to each remaining video.

### Gradient Boosting (GBM)
- Boosting but instead of reweighting points, you add new learners that fit the **residuals** (errors) of the current model.
- For squared loss:
  - Start with constant model \(F_0(x)=\bar y\)
  - Residuals \(r_i = y_i - F_{m-1}(x_i)\)
  - Fit weak learner \(h_m(x)\) to residuals
  - Update \(F_m(x)=F_{m-1}(x)+\eta h_m(x)\)
- Hyperparameters:
  - n_estimators, learning_rate \(\eta\), max_depth (base tree), subsample
- Interview: explain why small learning_rate + more trees often generalize better.

### XGBoost (Extremely Gradient Boosting)
- Optimized gradient boosting with:
  - regularization
  - efficient tree growth
  - handling missing values
  - parallelization + shrinkage
- Intuition:
  - each tree corrects errors of prior trees
  - objective = loss + regularization (controls complexity)
- Key knobs:
  - `eta`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`

### Stacking / Blending
- Train multiple base models, then train a **meta-learner** on their predictions.
- Must avoid leakage:
  - use out-of-fold predictions for meta model
- Works best when base models are diverse.

### Support Vector Machines (SVM)
- Goal: find hyperplane with **maximum margin**.
- Hard margin vs soft margin (C controls tolerance to misclassification).
- Kernel trick:
  - implicitly map to high dimensional feature space.
  - common kernels: linear, RBF, polynomial.
- Scaling is critical; SVM sensitive to feature scale.

### k-Nearest Neighbors (kNN)
- Instance-based, non-parametric.
- Choose `k`, distance metric; normalize features.
- Pros: simple. Cons: slow prediction, curse of dimensionality.

### Naive Bayes
- Bayes rule: \(P(y|x)\propto P(x|y)P(y)\)
- Naive assumption: features conditionally independent given class.
- Variants:
  - Gaussian NB (continuous)
  - Multinomial NB (counts/text)
  - Bernoulli NB (binary)
- Strong baseline for text classification.

### Clustering: KMeans, Hierarchical, DBSCAN
- **KMeans:**
  - iterative assign → update centroids
  - choose K (elbow, silhouette)
  - sensitive to scaling and initialization
- **Hierarchical:**
  - agglomerative clustering; linkage (single/complete/average/ward)
  - dendrogram helps choose number of clusters
- **DBSCAN:**
  - density-based; finds arbitrary shapes; identifies noise points
  - parameters: eps, min_samples

### Imbalanced data handling
- Problems: accuracy lies, minority recall suffers.
- Fixes:
  - better metrics: F1, PR-AUC, balanced accuracy
  - resampling: oversample (SMOTE), undersample
  - class weights
  - threshold tuning, calibrated probabilities

### Hyperparameter tuning (general)
- Use:
  - train/val split or cross-val
  - `RandomizedSearchCV` / Bayesian optimization (Optuna)
- Always tune **with the same pipeline** as training.
- Treat tuning as an experiment with tracked configs + seeds.

---

## Interview cheat sheet

### 1) Problem framing questions (always start here)
- What is the target? How is it defined & collected?
- What type of problem (classification/regression/ranking/forecasting)?
- What metric matches business cost? What are FP/FN costs?
- What is the baseline?

### 2) Leakage checklist
- Any feature derived from future info?
- Did you fit scalers/encoders on full data?
- Did you impute using info from the test set?
- Time series split needed?

### 3) Preprocessing defaults (tabular)
- Numeric: impute (median) → scale (standardize or robust)
- Categorical: impute (“missing”) → one-hot (or target encoding if high-cardinality)
- Use ColumnTransformer + Pipeline.

### 4) Model selection heuristics (tabular)
- Start simple:
  - Logistic/linear regression for interpretability + baseline
  - Tree-based models for nonlinearity + interactions
  - Random forest / gradient boosting for strong performance on tabular
- If many features / high dimension:
  - regularization (ridge/lasso)
  - PCA if needed for compression (but validate performance)

### 5) Metrics quick map
- Regression: MAE (robust), RMSE (big errors), R² (explained variance)
- Classification:
  - balanced? accuracy ok
  - imbalanced? precision/recall/F1, PR-AUC, ROC-AUC
  - calibrate thresholds based on cost.

### 6) Common “tell me about X” answers
- **Bias vs variance:** underfit vs overfit; fixes for each.
- **Regularization:** add penalty to control complexity → better generalization.
- **Ensembles:** diversity + aggregation reduces error; bagging reduces variance; boosting reduces bias.

---

## Video checklist (1–100)

> Tick as you revise.

- [ ] 1. What is Machine Learning? | 100 Days of Machine Learning
- [ ] 2. AI Vs ML Vs DL for Beginners in Hindi
- [ ] 3. Types of Machine Learning for Beginners | Types of ML in Depth
- [ ] 4. Batch Machine Learning | Offline Vs Online Learning | Machine Learning Types
- [ ] 5. Online Machine Learning | Online Vs Offline Machine Learning
- [ ] 6. Instance-Based Vs Model-Based Learning | Types of Machine Learning
- [ ] 7. Challenges in Machine Learning | Problems in Machine Learning
- [ ] 8. Application of Machine Learning | Real Life Machine Learning Applications
- [ ] 9. Machine Learning Development Life Cycle | MLDLC in Data Science
- [ ] 10. Data Engineer Vs Data Analyst Vs Data Scientist Vs ML Engineer | Data Science Job Roles
- [ ] 11. What are Tensors | Tensor In-depth Explanation | Tensor in Machine Learning
- [ ] 12. Installing Anaconda | Jupyter Notebook | Google Colab for ML
- [ ] 13. End to End Toy Project | Day 13 | 100 Days of Machine Learning
- [ ] 14. How to Frame a Machine Learning Problem | Plan a Data Science Project
- [ ] 15. Working with CSV files | Day 15
- [ ] 16. Working with JSON/SQL | Day 16
- [ ] 17. Fetching Data From an API | Day 17
- [ ] 18. Fetching data using Web Scraping | Day 18
- [ ] 19. Understanding Your Data | Day 19
- [ ] 20. EDA using Univariate Analysis | Day 20
- [ ] 21. EDA using Bivariate and Multivariate Analysis | Day 21
- [ ] 22. Pandas Profiling | Day 22
- [ ] 23. What is Feature Engineering | Day 23
- [ ] 24. Feature Scaling - Standardization | Day 24
- [ ] 25. Feature Scaling - Normalization | MinMaxScaling | MaxAbsScaling | RobustScaling
- [ ] 26. Encoding Categorical Data | Ordinal Encoding | Label Encoding
- [ ] 27. One Hot Encoding | Handling Categorical Data | Day 27
- [ ] 28. Column Transformer in Machine Learning | ColumnTransformer in Sklearn
- [ ] 29. Machine Learning Pipelines A-Z | Day 29
- [ ] 30. Function Transformer | Log / Reciprocal / Square Root Transform
- [ ] 31. Power Transformer | Box-Cox | Yeo-Johnson
- [ ] 32. Binning and Binarization | Discretization | Quantile / KMeans Binning
- [ ] 33. Handling Mixed Variables | Feature Engineering
- [ ] 34. Handling Date and Time Variables | Day 34
- [ ] 35. Handling Missing Data | Part 1 | Complete Case Analysis
- [ ] 36. Handling missing data | Numerical Data | Simple Imputer
- [ ] 37. Handling Missing Categorical Data | Most Frequent | Missing Category
- [ ] 38. Missing Indicator | Random Sample Imputation | Part 4
- [ ] 39. KNN Imputer | Multivariate Imputation | Part 5
- [ ] 40. MICE / Iterative Imputer | Multivariate Imputation by Chained Equations
- [ ] 41. What are Outliers | Outliers in Machine Learning
- [ ] 42. Outlier Detection/Removal using Z-score | Part 2
- [ ] 43. Outlier Detection/Removal using IQR | Part 3
- [ ] 44. Outlier Detection using Percentiles | Winsorization
- [ ] 45. Feature Construction | Feature Splitting
- [ ] 46. Curse of Dimensionality
- [ ] 47. PCA Part 1 | Geometric Intuition
- [ ] 48. PCA Part 2 | Formulation + Step-by-step
- [ ] 49. PCA Part 3 | Code Example + Visualization
- [ ] 50. Simple Linear Regression | Code + Intuition
- [ ] 51. Simple Linear Regression | Mathematical Formulation | Scratch
- [ ] 52. Regression Metrics | MSE, MAE, RMSE, R2, Adjusted R2
- [ ] 53. Multiple Linear Regression | Geometric Intuition & Code
- [ ] 54. Multiple Linear Regression | Part 2 | Mathematical Formulation
- [ ] 55. Multiple Linear Regression | Part 3 | Code From Scratch
- [ ] 56. Gradient Descent From Scratch | End-to-end + Animation
- [ ] 57. Batch Gradient Descent | Code demo
- [ ] 58. Stochastic Gradient Descent
- [ ] 59. Mini-Batch Gradient Descent
- [ ] 60. Polynomial Regression
- [ ] 61. Bias Variance Trade-off | Overfitting vs Underfitting
- [ ] 62. Ridge Regression Part 1 | Intuition + Code
- [ ] 63. Ridge Regression Part 2 | Math + Scratch
- [ ] 64. Ridge Regression Part 3 | Gradient Descent
- [ ] 65. Ridge Regression Part 4 | 5 Key Points
- [ ] 66. Lasso Regression | Intuition + Code
- [ ] 67. Why Lasso creates sparsity?
- [ ] 68. ElasticNet Regression | Intuition + Code
- [ ] 69. Logistic Regression Part 1 | Perceptron Trick
- [ ] 70. Logistic Regression Part 2 | Perceptron Trick Code
- [ ] 71. Logistic Regression Part 3 | Sigmoid Function
- [ ] 72. Logistic Regression Part 4 | Loss Function | MLE | Binary Cross Entropy
- [ ] 73. Derivative of Sigmoid Function
- [ ] 74. Logistic Regression Part 5 | Gradient Descent | Code From Scratch
- [ ] 75. Accuracy + Confusion Matrix | Type 1 & Type 2 Errors | Metrics Part 1
- [ ] 76. Precision, Recall and F1 Score | Metrics Part 2
- [ ] 77. Softmax Regression | Multinomial Logistic Regression
- [ ] 78. Polynomial Features in Logistic Regression | Nonlinear Logistic Regression
- [ ] 79. Logistic Regression Hyperparameters
- [ ] 80. Decision Trees | Entropy | Gini | Information Gain
- [ ] 81. Decision Trees Hyperparameters | Overfitting/Underfitting
- [ ] 82. Regression Trees
- [ ] 83. Decision Tree Visualization using dtreeviz
- [ ] 84. Introduction to Ensemble Learning
- [ ] 85. Voting Ensemble Part 1 | Core idea
- [ ] 86. Voting Ensemble Part 2 | Classification | Hard vs Soft Voting
- [ ] 87. Voting Ensemble Part 3 | Regression
- [ ] 88. Bagging Part 1 | Intro
- [ ] 89. Bagging Part 2 | Bagging Classifiers
- [ ] 90. Bagging Part 3 | Bagging Regressor
- [ ] 91. Random Forest | Intuition
- [ ] 92. Why Random Forest performs so well? Bias/Variance
- [ ] 93. Bagging vs Random Forest
- [ ] 94. Random Forest Hyperparameters
- [ ] 95. RF Hyperparameter Tuning (GridSearchCV & RandomizedSearchCV)
- [ ] 96. OOB Score | Out of Bag Evaluation
- [ ] 97. Feature Importance in RF & Trees
- [ ] 98. AdaBoost Classifier | Geometric Intuition
- [ ] 99. AdaBoost | Step-by-step Explanation
- [ ] 100. AdaBoost Algorithm | Code from Scratch
 
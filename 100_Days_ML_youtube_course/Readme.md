# 100 Days of Machine Learning (CampusX) — Concise Notes  
**Part 1: Day 01 → Day 05**  

> Goal: quick-review notes in **Q/A format** + a few small code demos so you can revise fast and answer interview-style questions.

---

## Table of contents
- [Day 01 — What is Machine Learning?](#day-01--what-is-machine-learning)
- [Day 02 — AI vs ML vs DL](#day-02--ai-vs-ml-vs-dl)
- [Day 03 — Types of Machine Learning](#day-03--types-of-machine-learning)
- [Day 04 — Batch (Offline) Learning](#day-04--batch-offline-learning)
- [Day 05 — Online Learning](#day-05--online-learning)
- [Quick Revision Cheatsheet (Day 01–05)](#quick-revision-cheatsheet-day-0105)

---

## Day 01 — What is Machine Learning?
**Video:** *What is Machine Learning? | 100 Days of Machine Learning*  
Link: https://www.youtube.com/watch?v=ZftI2fEz0Fw  

### 1) What is Machine Learning?
**Q: What is Machine Learning (ML)?**  
**A:** ML is a way to build software where the system **learns patterns from data** (experience) instead of you writing every rule manually. The “output” of learning is typically a **model** (a mathematical function) that can make predictions/decisions on new data.

**Q: What does “learn from data” actually mean?**  
**A:** The algorithm adjusts its internal parameters so that it performs better on a task (classification/prediction) based on examples.

---

### 2) ML vs Traditional Programming
**Q: How is ML different from traditional programming?**  
**A:**  
- **Traditional:** `Rules + Input → Output` (you hard-code rules)  
- **ML:** `Data (Input+Output examples) → Model`, then `Model + New Input → Predicted Output`  

**Q: Why can’t we always use if-else rules?**  
**A:** Because many real problems have:
- too many rules (rule explosion)
- rules keep changing (need continuous adaptation)
- patterns are hard for humans to explicitly define

---

### 3) When should you use ML?
**Q: When is ML a good choice?**  
**A:** Common “ML-worthy” cases:
1. **Complex rule-based tasks** (e.g., spam detection)  
2. **Perception tasks** (image/audio/text understanding)  
3. **Data mining** (discover hidden patterns in large datasets)

**Q: When is ML a *bad* choice?**  
**A:** If a task can be solved by a few stable rules, or you have **no data**, or the cost of wrong predictions is unacceptable without strong controls.

---

### 4) What will this course focus on?
**Q: What’s the focus of this “100 Days” playlist?**  
**A:** More on **end-to-end ML workflow / life cycle** (preprocessing, feature selection, model selection, bias–variance, etc.) rather than only “algorithm theory”.

---

### Mini demo (Supervised ML): tiny spam classifier
> This is NOT meant to be a perfect spam filter—just to show the “rules come from data” idea.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "Win money now", "Limited time offer win cash",
    "Hello how are you", "Let's meet for lunch",
    "Cheap meds available", "Your invoice is attached",
    "Claim your free prize", "Are we still on for tomorrow?"
]
y = ["spam","spam","ham","ham","spam","ham","spam","ham"]

vec = CountVectorizer()
X = vec.fit_transform(texts)

clf = MultinomialNB().fit(X, y)

test_texts = ["free money prize", "see you at lunch tomorrow"]
print(clf.predict(vec.transform(test_texts)))
```

**Output (example):**
```text
['spam' 'ham']
```

---

### Interview-style flash Q/A (Day 01)
**Q: Define ML in one line.**  
**A:** “ML builds models from data to make predictions/decisions without explicit rule coding.”

**Q: What are 3 signs a problem needs ML?**  
**A:** Too many rules, rules change, pattern recognition needed.

---

## Day 02 — AI vs ML vs DL
**Video:** *AI Vs ML Vs DL for Beginners in Hindi*  
Link: https://www.youtube.com/watch?v=1v3_AQ26jZ0  

### 1) Relationship (the “circles” idea)
**Q: How are AI, ML, and DL related?**  
**A:**  
- **AI (Artificial Intelligence)** = the broad umbrella: “machines showing intelligent behavior.”  
- **ML (Machine Learning)** ⊂ AI: systems learn from data.  
- **DL (Deep Learning)** ⊂ ML: learning using **neural networks with many layers**, great for complex patterns.

A simple mental model:

```
AI
└── ML
    └── DL
```

---

### 2) What is AI?
**Q: What counts as AI?**  
**A:** Anything that tries to make computers perform tasks requiring “intelligence” (search, planning, reasoning, perception, language, etc.). AI can include:
- **rule-based systems** (classic AI)
- **learning-based systems** (ML/DL)

---

### 3) What is ML (in this context)?
**Q: What makes an ML system “learn”?**  
**A:** It uses data to find patterns and make better predictions over time.

**Q: Why is data so important for ML?**  
**A:** Model performance often depends heavily on **data quality + quantity** (garbage in → garbage out).

---

### 4) What is DL and why is it “special”?
**Q: Why do we need Deep Learning if ML already exists?**  
**A:** DL is especially effective when:
- data is **unstructured** (images, audio, text)
- feature design is hard for humans  
DL can learn representations/features automatically, but usually needs **more data** and **stronger hardware** (GPUs).

---

### 5) Typical applications
**Q: Give examples of where AI/ML/DL show up.**  
**A:**  
- ML: spam detection, recommendations, fraud detection  
- DL: image recognition, speech recognition, autonomous driving components  
- AI (umbrella): includes both, plus classic planning/search systems

---

### Interview-style flash Q/A (Day 02)
**Q: “Is all AI ML?”**  
**A:** No. ML is a subset of AI; AI also includes non-learning approaches.

**Q: “Is all ML Deep Learning?”**  
**A:** No. DL is a subset of ML.

---

## Day 03 — Types of Machine Learning
**Video:** *Types of Machine Learning for Beginners | Types of ML in Depth*  
Link: https://www.youtube.com/watch?v=81ymPYEtFOw  

### 1) Main types (based on supervision)
**Q: What does “supervision” mean in ML?**  
**A:** Whether the training data includes correct answers (**labels**).

**Q: What are the main ML types discussed?**  
**A:**  
1. **Supervised Learning** (labeled data)  
2. **Unsupervised Learning** (no labels)  
3. **Semi-Supervised Learning** (few labels + lots of unlabeled data)  
4. **Reinforcement Learning** (learn via rewards/penalties)

---

### 2) Supervised Learning
**Q: What is supervised learning?**  
**A:** Learn a mapping: **inputs (X) → outputs (y)** from labeled examples.

**Q: What are the two subtypes?**  
**A:**  
- **Regression:** output is **numeric/continuous** (e.g., salary prediction)  
- **Classification:** output is **categorical** (e.g., spam vs ham, placed vs not placed)

---

### 3) Unsupervised Learning
**Q: What is unsupervised learning?**  
**A:** You only have **X (inputs)**. The goal is to discover structure/patterns.

**Q: Common unsupervised tasks mentioned?**  
**A:**  
- **Clustering:** group similar items (customer segmentation)  
- **Dimensionality reduction:** reduce features for visualization/compression  
- (Often also includes association rules in many curricula)

---

### 4) Semi-Supervised Learning
**Q: What is semi-supervised learning?**  
**A:** Training with **a small labeled set + a large unlabeled set**, useful when labeling is expensive (e.g., image labeling).

**Q: Example intuition?**  
**A:** Like giving a few “named” photos to an app and letting it generalize to many unlabeled photos.

---

### 5) Reinforcement Learning (RL)
**Q: What is reinforcement learning?**  
**A:** An **agent** interacts with an **environment**, takes **actions**, receives **rewards/penalties**, and learns a strategy (policy) to maximize long-term reward.

**Key RL words to remember:** agent, environment, state, action, reward, policy.

---

### Mini demo (Unsupervised ML): KMeans clustering
```python
import numpy as np
from sklearn.cluster import KMeans

X = np.array([
    [0.0, 0.1], [0.2, -0.1], [-0.1, 0.0],
    [3.0, 3.1], [2.8, 2.9], [3.2, 2.7]
])

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
print("labels:", labels.tolist())
print("centers:", kmeans.cluster_centers_)
```

**Output (example):**
```text
labels: [0, 0, 0, 1, 1, 1]
centers: [[0.0333... 0.0...]
          [3.0...   2.9...]]
```

---

### Interview-style flash Q/A (Day 03)
**Q: Supervised vs Unsupervised in one line?**  
**A:** Supervised learns with labels; unsupervised finds structure without labels.

**Q: Regression vs Classification?**  
**A:** Regression predicts numbers; classification predicts classes.

---

## Day 04 — Batch (Offline) Learning
**Video:** *Batch Machine Learning | Offline Vs Online Learning*  
Link: https://www.youtube.com/watch?v=nPrhFxEuTYU  

### 1) What is “production”?
**Q: What does “production” mean in ML?**  
**A:** The real environment where the trained model is deployed (e.g., on servers) and used by users/customers.

---

### 2) What is Batch / Offline learning?
**Q: What is batch learning (offline learning)?**  
**A:** Train the model **using the full dataset at once** (offline), then deploy it. Updates happen by **retraining** later.

**Typical flow:**
1. Collect data → 2. Train model offline → 3. Deploy → 4. Model gets stale → 5. Retrain on new data → redeploy

---

### 3) Why batch learning exists
**Q: Why do companies still use batch learning?**  
**A:** It’s simpler to implement and often works well when:
- data changes slowly
- you can retrain periodically (daily/weekly/monthly)
- real-time adaptation is not required

---

### 4) Limitation of batch learning
**Q: What’s the biggest limitation?**  
**A:** **No automatic adaptation** to new patterns unless you retrain.  
This becomes a big deal in systems like recommendations or fraud detection where patterns evolve quickly.

---

### Interview-style flash Q/A (Day 04)
**Q: What is “model staleness”?**  
**A:** When the real world changes but the model still reflects old data.

---

## Day 05 — Online Learning
**Video:** *Online Machine Learning | Online Vs Offline Machine Learning*  
Link: https://www.youtube.com/watch?v=3oOipgCbLIk  

### 1) What is online learning?
**Q: What is online ML?**  
**A:** Training **incrementally** (step-by-step) as new data arrives, instead of training once on a static dataset.

---

### 2) How does online learning work?
**Q: What’s the process?**  
**A:**  
- start with a small trained model  
- as new data arrives, update the model continuously (often in mini-batches)  
- monitor performance to ensure updates are improving things

---

### 3) When should you use online learning?
**Q: When is online learning useful?**  
**A:**  
- when data arrives as a **stream** (real-time user events)  
- when data is too big for memory (**out-of-core learning**)  
- when the pattern changes over time (**concept drift**)

---

### 4) Tools & terms mentioned
**Q: What tools/libraries are mentioned for online learning?**  
**A:** Common names mentioned include:
- **River** (Python, streaming ML)
- **Vowpal Wabbit** (fast online learning system)

**Q: What is “online learning rate”?**  
**A:** The step size controlling how strongly new data updates the model. Too high → unstable; too low → slow adaptation.

**Q: What is out-of-core learning?**  
**A:** Training when the dataset doesn’t fit in RAM, by processing it in chunks/streams.

---

### 5) Downsides / risks
**Q: Why is online learning harder than offline?**  
**A:** Because you must handle:
- noisy/bad incoming data
- bias introduced over time
- monitoring + anomaly detection
- infrastructure complexity

---

### Mini demo (Online learning): incremental training with `partial_fit`
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=600, n_features=10, n_informative=5, n_redundant=2,
    random_state=42
)

X_stream, y_stream = X[:400], y[:400]      # streaming training data
X_test, y_test = X[400:], y[400:]          # holdout test set

clf = SGDClassifier(loss="log_loss", random_state=42)

for i in range(0, 400, 100):
    X_batch = X_stream[i:i+100]
    y_batch = y_stream[i:i+100]
    if i == 0:
        clf.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
    else:
        clf.partial_fit(X_batch, y_batch)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"After batch {(i//100)+1}: test accuracy={acc:.3f}")
```

**Output (example):**
```text
After batch 1: test accuracy=0.680
After batch 2: test accuracy=0.815
After batch 3: test accuracy=0.710
After batch 4: test accuracy=0.805
```

> Note: online learning accuracy can fluctuate because the model keeps updating; that’s why monitoring matters.

---

### Interview-style flash Q/A (Day 05)
**Q: Online vs batch learning in one line?**  
**A:** Batch trains on all data at once (periodic retrain), online updates continuously with new data.

**Q: Give 2 real examples where online learning helps.**  
**A:** Recommendations (YouTube/Netflix) and fraud detection.

---

## Quick Revision Cheatsheet (Day 01–05)

### 1) Quick definitions (memorize-ready)
- **AI:** umbrella for making machines act intelligently.  
- **ML:** learn patterns from data to make predictions.  
- **DL:** neural networks (many layers), strong for unstructured data.  
- **Supervised:** labeled data (regression/classification).  
- **Unsupervised:** no labels (clustering, dimensionality reduction).  
- **Semi-supervised:** few labels + many unlabeled.  
- **Reinforcement:** agent learns via reward.  
- **Batch learning:** train offline on full dataset, redeploy periodically.  
- **Online learning:** update incrementally as new data arrives.

### 2) Decision cheat: “Which learning type is it?”
- **Do you have labels?**  
  - Yes → supervised  
  - No → unsupervised  
  - Few labels + many unlabeled → semi-supervised  
- **Is there an agent interacting with an environment + rewards?**  
  - Yes → reinforcement learning  

### 3) Decision cheat: “Batch vs Online”
- Use **Batch** when: data changes slowly, simpler infra, periodic retrain is fine.  
- Use **Online** when: data streams in, concept drift exists, or data is too big for RAM.

---

### Sources (videos)
- Playlist (Day 01–05 titles): https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH
- Day 01: https://www.youtube.com/watch?v=ZftI2fEz0Fw  
- Day 02: https://www.youtube.com/watch?v=1v3_AQ26jZ0  
- Day 03: https://www.youtube.com/watch?v=81ymPYEtFOw  
- Day 04: https://www.youtube.com/watch?v=nPrhFxEuTYU  
- Day 05: https://www.youtube.com/watch?v=3oOipgCbLIk  

# What this assignment is asking you to do — explained clearly

Rough instructions are a compact roadmap for a complete NLP mini-project. The goal is to take raw text datasets, clean and explore them, build a few models that solve a defined text task, compare their performance using sensible metrics and visuals, and produce polished results (graphs, tables, write-up). Below I break down each step, explain *why* it matters, and give a concrete, step-by-step plan you can follow (including concrete dataset choices, preprocessing recipes, model suggestions, evaluation methods, and visualization ideas).

---

# 1) Decide the task & pick two datasets

**What it means:** Choose what kind of NLP problem you’ll solve (e.g., sentiment analysis, spam detection, topic classification, intent classification). The two datasets can be two separate tasks or two datasets for the *same* task (good for showing generalization).

**Why:** The methods and metrics depend on the task. Choosing two datasets gives you more material to compare model robustness.

**Concrete suggestions (easy to use, well-known):**

* **Sentiment analysis:** IMDB movie reviews (binary, long reviews) and Yelp reviews (multi-star / can be converted to binary).
* **Topic / news classification:** 20 Newsgroups (multi-class) or AG News (4-class).
* **Spam detection:** SMS Spam Collection (binary).
* **Short social-text sentiment:** Sentiment140 (tweets).

**Which two to pick (good combo for an assignment):**

* **IMDB (sentiment)** + **20 Newsgroups (topic classification)** — shows both sentiment and topic classification.
  Or, if you want to keep the task the same,
* **IMDB + Yelp** (both sentiment) — shows model transfer/generalization.

---

# 2) Preprocessing & EDA — what to do and why

## Preprocessing steps (pipeline)

1. **Inspect raw fields**: text, label, metadata (length, date).
2. **Lowercasing** (optional for some models).
3. **Remove or normalize URLs, emails, mentions** (for social media datasets).
4. **Tokenization** — word tokenizer for classical models; subword (BPE) for transformers.
5. **Remove or keep punctuation** depending on model (transformers keep; some classical pipelines remove).
6. **Handle stopwords**: remove for BoW/TF-IDF baselines; keep for neural/transformer models.
7. **Stemming / Lemmatization**: helpful for classical models sometimes — try both.
8. **Deal with numbers**: map to special token or keep.
9. **Handle class imbalance**: resample or use class weights.
10. **Train/validation/test split**: stratified splits (e.g., 80/10/10 or 70/15/15).

## Exploratory Data Analysis (EDA) — what to include

* **Class balance** (bar chart of label counts).
* **Text length distribution** (histogram of token counts).
* **Most frequent words** per class (top-N lists).
* **n-grams** (unigrams, bigrams) bar charts per label.
* **Word clouds** per class (visual but optional).
* **Outliers / weird cases** (very long or short texts).
* **Correlation of length vs label** (boxplot).
* **Percentage of missing/empty texts**.
* **Vocabulary size and OOV rate** (how many tokens are unseen in embedding or train set).
* **Sample examples** (handful of positive/negative texts to show why problem is tricky).

**Tip:** Make EDA part of the story — use visuals as evidence when you later interpret model errors.

---

# 3) Models to apply (2–3 models) — recommended set

You want a **baseline**, a **strong classical model**, and a **modern neural / transformer** model (if compute allows).

### Simple baseline (must-have)

* **Majority class** and **TF-IDF + Logistic Regression** or **TF-IDF + Multinomial Naive Bayes**.

  * Why: fast, interpretable, strong baseline for many text tasks.

### Classical / stronger baseline

* **SVM (LinearSVC) with TF-IDF** or **Logistic Regression with n-grams and char-n-grams**.

  * Tune regularization (C), n-gram range, max features.

### Neural / modern

* **BiLSTM** with pre-trained word embeddings (GloVe / FastText) — if you want an RNN.
* **CNN for text** (Kim CNN) — lighter than LSTM and good for short texts.
* **BERT / DistilBERT fine-tuning** (transformer): usually the best performing for many tasks.

  * DistilBERT if you have limited compute; full BERT if you have GPU/time.

**If you must pick only 2–3:**

* TF-IDF + Logistic Regression, LinearSVC, and BERT (or DistilBERT).

---

# 4) Training details & validation

* **Stratified train/val/test splits.**
* **Cross-validation** (5-fold stratified CV) for baselines to report stable scores.
* **Hyperparameter search:** grid or random search for LR/SVM (C, n-gram range, max_features). For transformers, try 2–3 learning rates (e.g., 2e-5, 3e-5), batch sizes, and epochs (2–4).
* **Early stopping** on validation loss / metric.
* **Seed everything** for reproducibility.
* **Use class weights** or oversampling if labels are imbalanced.

---

# 5) Evaluation & comparison — what metrics to report

**Classification metrics:**

* **Accuracy** (useful if balanced).
* **Precision / Recall / F1** (report per class and macro-average).
* **Confusion matrix** (visual).
* **ROC AUC / PR AUC** (for binary problems or one-vs-rest in multiclass).

**Model comparison approaches:**

* **Numeric table** with metrics (accuracy, macro F1, macro precision, macro recall, inference time, train time, model size).
* **Statistical significance test** (e.g., McNemar’s test) when comparing two classifiers on the same test set — shows if improvement is significant.
* **Error analysis**: sample where model A is correct and model B fails; categorize error types (sarcasm, negation, domain terms).
* **Calibration / probability reliability** (optional).

---

# 6) Graphs and visual outputs you should produce

(These make your report look strong — your friend already said “Get good graphs.”)

**Essential visuals**

1. **Class distribution bar chart.**
2. **Text length histogram.**
3. **Top 20 words / n-grams per class** (bar charts).
4. **Confusion matrix heatmaps** for each model.
5. **Precision–Recall curves** or **ROC curves** (binary).
6. **Model comparison bar chart** (e.g., macro F1 for each model side-by-side).
7. **Training/validation loss & accuracy curves** (for neural/transformers).
8. **Error breakdown pie chart** (types of errors).
9. **Feature importance** for linear models (top positive/negative coefficients).
10. **Examples of model predictions** (table: text, true label, pred label, confidence).

**Make visuals clean:** label axes, add captions, and keep fonts readable. Use colorblind-friendly palettes if possible.

---

# 7) Write-up / report structure

1. **Title & objective** — one line describing the problem and datasets.
2. **Datasets** — short description + sample rows + dataset stats.
3. **Preprocessing & EDA** — visuals and insights.
4. **Models & experimental setup** — hyperparams, train/val/test split, compute used.
5. **Results** — metric tables, graphs, significance tests.
6. **Error analysis** — qualitative examples and failure modes.
7. **Conclusion & next steps** — what worked, tradeoffs, suggested improvements.
8. **Appendix** — code links, full hyperparam tables.

---

# 8) Deliverables to hand in

* **Jupyter notebook(s)** (well-organized, runnable).
* **Report (PDF)** with all key visuals and explanations.
* **Code folder** with README and instructions to run.
* **Optional demo**: Streamlit app showing model predictions on input text.
* **Presentation slides** (if required by assignment).

---

# 9) Practical tips & pitfalls

* **Start simple:** get TF-IDF + Logistic Regression working before touching transformers.
* **Reproducibility:** set random seeds and log package versions.
* **Compute limits:** fine-tuning BERT needs GPU; use DistilBERT if no GPU.
* **Small datasets:** fine-tuning big transformers can overfit — use regularization and data augmentation.
* **Data leakage:** ensure no test data is used during preprocessing steps like vocabulary building.
* **Interpretability:** linear models + coefficient lists give easy human-readable features. Use SHAP for deeper interpretability if time permits.

---

# 10) Concrete mini plan you can follow (actionable)

1. **Pick task & datasets** (IMDB + Yelp or IMDB + 20 Newsgroups).
2. **Explore & preprocess** (EDA visuals + clean pipeline).
3. **Baseline:** TF-IDF + Logistic Regression. Report metrics & confusion matrix.
4. **Stronger classical:** LinearSVC with n-grams and char-n-grams. CV to tune.
5. **Modern model:** Fine-tune DistilBERT for 2–4 epochs (or use BiLSTM + GloVe if no GPU).
6. **Compare:** metric table, per-class F1, confusion matrices, PR/ROC curves. Run McNemar if comparing two classifiers.
7. **Error analysis & write-up.**
8. **Deliverables:** notebook, report, README, optional Streamlit demo.

---

# Quick checklist to get started (copyable)

* [ ] Choose task and two datasets.
* [ ] Create train/val/test splits (stratified).
* [ ] Implement preprocessing functions (tokenize, clean, lemmatize optional).
* [ ] Create EDA notebook with visuals.
* [ ] Implement TF-IDF + Logistic Regression baseline.
* [ ] Implement SVM baseline and hyperparameter tuning.
* [ ] Fine-tune DistilBERT (or BiLSTM if no GPU).
* [ ] Evaluate & compare, produce graphs + tables.
* [ ] Do error analysis and write report.

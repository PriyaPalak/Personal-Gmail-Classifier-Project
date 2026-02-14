# Personal Gmail Classifier Project
A Classification System to classify my emails as Important, Promotional and Spam as per my preferences.

---
## 🎯 Problem Statement
Modern inboxes are heavily cluttered with promotional and social emails, while genuinely important emails are fewer but critical. Default Gmail categorization is not fully reliable for personal relevance because:

- Promotions often appear in Primary or Updates
- Important emails may be mixed with marketing content
- Spam detection is optimized for general users, not individual preferences

The goal of this project is to build a personalized email classifier that predicts whether an email is:

- **Important** → I would open and act on it  
- **Promotional** → Legitimate but ignorable  
- **Spam** → I do not want to see it  

The classifier should reflect *my personal priorities*, not generic rules.

---

## 🧭 Approach
This project follows a classical supervised machine learning pipeline:

1. Extract structured data from Gmail
2. Intelligently sample emails for training
3. Manually label emails based on personal relevance
4. Convert text to numerical features (TF-IDF)
5. Train a Logistic Regression classifier
6. Evaluate using class-wise metrics and probabilistic loss

The design philosophy is:

> **High-quality labels + smart sampling > large noisy dataset**

---

## 📥 Data Extraction
Email data is extracted using the Gmail API in three steps:

1. Enable Gmail API in Google Cloud Console  
2. Authenticate Python access  
3. Extract messages  

To reduce noise, only high-signal fields are collected:

- `message_id`
- `subject`
- `snippet`
- `sender`
- `sender_domain`
- `internal_date`
- `labels`
- `has_attachment`

These fields provide enough semantic and contextual information without unnecessary payload.

---

## 🧪 Sampling and Labelling

### Why Random Sampling Fails
The mailbox contains ~70k emails, but distribution is highly skewed:

- 80–90% → promotional/social
- Few → important
- Very few → spam

Random sampling would produce:

- Almost no spam examples
- Very few important emails
- Highly biased dataset

This leads to a weak classifier that never learns minority classes.

---

### Stratified Sampling Strategy
Instead of random sampling, **stratified sampling** is used:

- Split emails into meaningful groups
- Sample from each group deliberately
- Ensure representation of all email types

Sampling diversity dimensions:

- Gmail label
- Sender domain
- Time
- Email intent

---

### Sampling Buckets (Pre-Labelling)
Based on Gmail categories and personal usage patterns:

| Bucket | Count | Source Categories | Rationale |
|------|------|----------------|---------|
Important | 500 | Primary / Updates | Rare but critical → high precision needed |
Promotions | 300 | Promotions | Common but unimportant |
Spam | 400 | Social | Most frequent and unwanted |

Total sample ≈ **1200 emails**

---

### Manual Labelling
After sampling, each email is manually labelled using personal criteria:

| Label | Meaning |
|------|--------|
Important | I would open & act |
Promotional | Legit but ignorable |
Spam | I don’t want this |

Final dataset distribution:

| Class | Count |
|------|------|
Important | 290 |
Promotional | 510 |
Spam | 399 |

This dataset is not perfectly balanced but **healthy**:

- No tiny classes
- No dominating class
- All classes sufficiently represented

---

### Train/Test Split
To preserve class proportions during evaluation:

> **Stratified Train-Test Split** is used

This ensures:
- Fair evaluation
- All classes present in both sets
- Reliable metrics

---

## 🤖 Model Training

### Feature Engineering
Text fields are converted into numerical vectors using:

> **TF-IDF Vectorization**

This captures word importance relative to document frequency and works well for classical NLP classification.

---

### Model Choice
A **Logistic Regression classifier** is used as the baseline model because it is:

- Simple
- Interpretable
- Strong for sparse text features
- Fast to train

---

### Evaluation Philosophy
Accuracy alone is misleading for multi-class classification, especially with imbalanced data. Therefore evaluation includes:

- Precision → trustworthiness of predictions
- Recall → ability to catch real cases
- F1 Score → balance of precision and recall
- Confusion Matrix → exact mistake patterns
- Log Loss → probability calibration quality

---

### Model Results

**Classification Report**

| Class | Precision | Recall | F1 |
|------|----------|------|----|
Important | 0.92 | 0.59 | 0.72 |
Promotional | 0.73 | 0.97 | 0.84 |
Spam | 1.00 | 0.85 | 0.92 |

Accuracy: **0.84**

**Key Takeaways**

- Spam detection is excellent (perfect precision)
- Important predictions are highly trustworthy
- Promotions are well identified
- Model is conservative with “Important” predictions (low recall)

This is desirable behavior because false “Important” predictions are more harmful than missed ones.

---

### Log Loss Evaluation (Andrew Ng Method)

| Model | Train | Test |
|------|------|------|
Random Guess | ~1.10 | ~1.10 |
Majority Class | ~0.9–1.2 | ~0.9–1.2 |
**TF-IDF + Logistic Regression** | **0.31** | **0.46** |

Insights:

- Model clearly learns real patterns
- Strong generalization
- Small train–test gap → good bias-variance tradeoff

---

### Final Summary
The TF-IDF + Logistic Regression classifier achieves strong performance with limited labeled data, significantly outperforming naive baselines and demonstrating reliable probabilistic predictions tailored to personal email relevance.


## 🚀 Model Improvement

The baseline model performed well overall but showed one key limitation:

> **Low recall for Important emails**

This meant the classifier was too conservative and missed many emails that were actually important. Since missing important emails is more costly than false alarms, the goal was to **increase Important recall while preserving precision and spam safety**.

---

### Approach — Cross-Validation + Class Weights
To ensure improvements generalize, we used:

> **Stratified K-Fold Cross-Validation with Log Loss**

We then adjusted **class weights** so the model penalizes mistakes on Important emails more heavily.

Tested weights:

| Important Weight | Result |
|-----------------|--------|
1.5 | Minor recall gain |
2.0 | Strong recall gain + stable metrics |

Selected configuration:
```yaml
class_weight:
  important: 2
  promotional: 1
  spam: 1
````
---

### Improved Performance

| Metric | Before | After |
|------|-------|------|
Important Recall | 0.59 | **0.78** |
Spam Precision | 1.00 | **1.00** |

Key Outcomes:

- Most important emails are now detected  
- Predictions remain trustworthy  
- No legitimate emails marked as spam  
- Promotions remain well separated  

---

### Generalization Check

| Metric | Value |
|------|------|
Train Log Loss | 0.304 |
Test Log Loss | 0.458 |

> Small gap indicates stable generalization and no overfitting.

---

### Summary
By increasing the class weight of Important emails and validating via cross-validation, the model significantly improved detection of important emails while maintaining strong overall performance and reliability.

## 🧠 Challenges and Learnings

### Challenges

**1. Highly Imbalanced Real-World Data**  
The mailbox distribution was extremely skewed (majority promotional/social, very few important or spam). Random sampling produced biased datasets that prevented the model from learning minority classes effectively.

**2. Gmail Labels ≠ Ground Truth**  
Default Gmail categories were inconsistent with personal relevance. Important emails appeared in Promotions or Updates, while some promotional content appeared in Primary. This meant labels could not be trusted directly and required manual correction.

**3. Limited Labeled Data**  
Although ~70k emails existed, only ~1200 could realistically be labeled. This required maximizing information from a small, carefully curated dataset.

**4. Metric Selection Complexity**  
Accuracy alone was misleading for multi-class classification. Understanding model behavior required interpreting precision, recall, F1-score, confusion matrix, and log loss together.

**5. Behavior Optimization vs Accuracy Optimization**  
Improving recall for Important emails required intentionally trading off precision. The challenge was controlling this trade-off without degrading spam detection or overall stability.

---

### Key Learnings

**Stronger Data > Bigger Data**  
High-quality labels and smart sampling matter more than dataset size.

**Sampling Strategy Shapes Model Intelligence**  
Stratified sampling ensured the model learned all email types instead of overfitting to dominant classes.

**Model Behavior Can Be Controlled**  
Adjusting class weights demonstrated that model predictions can be intentionally guided based on real-world priorities.

**Evaluation Must Match Real Goals**  
Different metrics answer different questions:

- Precision → Can I trust predictions?
- Recall → Am I missing important cases?
- Log Loss → Are probabilities calibrated?

**Cross-Validation Prevents False Confidence**  
Using stratified K-fold validation ensured improvements were real and not due to lucky splits.

---

### Final Insight
> Building real-world ML systems is less about choosing complex algorithms and more about designing data, evaluation, and objectives that reflect actual user needs.


## 🔮 Next Steps

**1. Expand Labeled Dataset**  
Gradually label more emails to improve model robustness and help it learn rarer patterns, especially edge cases for Important and Spam.

**2. Tune Decision Thresholds**  
Adjust class probability thresholds to further optimize the recall–precision trade-off for Important emails without compromising spam safety.

**3. Build Feedback + Retraining Loop**  
Create a system where user corrections are logged and periodically used to retrain the model so it continuously adapts to evolving inbox behavior.

---

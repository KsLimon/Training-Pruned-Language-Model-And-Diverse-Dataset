# Training Pruned Language Model

**Research Paper Implementation**

[![DOI](https://img.shields.io/badge/DOI-Digital%20Object%20Identifier-blue)]()

## Authors

- **MD KAMRUS SAMAD** - Department of Electrical and Computer Engineering, North South University  
  📧 kamrus.samad@northsouth.edu

- **ANAN GHOSH** - Department of Electrical and Computer Engineering, North South University  
  📧 anan.ghosh@northsouth.edu

- **SAJIB HOSSAIN** - Department of Electrical and Computer Engineering, North South University  
  📧 sajib.hossain03@northsouth.edu

- **DR. NABEEL MOHAMMED** - Department of Electrical and Computer Engineering, North South University  
  📧 nabeel.mohammed@northsouth.edu

**Corresponding Author:** Md Kamrus Samad

---

## Abstract

Natural language processing using large pre-trained models like BERT is expensive, time-consuming, has a large carbon footprint, and is nearly difficult to implement on machines with minimal CPU capability. This research demonstrates that by reducing storage requirements and increasing inference computational efficiency without sacrificing comparative performance in different downstream tasks, a much greener, resource-saving, and comparatively smaller model can achieve performance comparable to larger models.

**Key Findings:**
- Models can perform comparably well in various downstream tasks after losing 90% of their weight
- Tiny models can be few-shot learners when pruned up to 90% sparsity rate
- Few-shot learning models can be trimmed using Lottery Ticket Hypothesis without affecting comparative performance on Bangla NLP tasks

---

## Table of Contents

- [Introduction](#introduction)
- [Literature Review](#literature-review)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [References](#references)

---

## Introduction

Natural language processing research faces significant challenges with massive models like BERT that use Transformer architectures and require huge computational resources. Large models:

- Leave enormous carbon footprints
- Consume massive amounts of energy
- Require prohibitively expensive computational power
- Are nearly impossible to implement in resource-constrained systems

This research explores **pruning techniques** to eliminate more than 90% of unnecessary weights from neural networks without harming accuracy. We combine:

1. **Iterative Pattern Exploiting Training (iPET)** for extremely small pre-trained models
2. **Lottery Ticket Hypothesis (LTH)** for pruning BERT models with 90% sparsity
3. Performance measurement on Bangla language downstream tasks

### Key Contributions

- Iterative pattern exploiting training on Bangla Language makes the model a few-shot learner
- Pruning using Lottery Ticket Hypothesis to assess trainable, transferrable subnetworks in pre-trained BERT models with 90% sparsity
- Competitive performance on downstream tasks comparable to bigger models while being greener and more resource-efficient

---

## Literature Review

### Key Research Areas Explored

#### 1. Small Language Models as Few-Shot Learners
Pattern Exploiting Training (PET) combines reformulating tasks as cloze questions with gradient-based fine-tuning, enabling models with 0.1% of GPT-3's parameters to achieve comparable performance.

#### 2. Efficient BERT Training via Early-Bird Lottery Tickets
EarlyBERT identifies structured winning tickets in early training stages by slimming self-attention and fully connected sub-layers, consisting of three stages:
- Searching Stage
- Ticket Drawing Stage  
- Efficient-Training Stage

#### 3. Lottery Ticket Hypothesis for Pre-trained BERT
Research shows that pre-trained BERT models contain trainable, transferable subnetworks at 40-90% sparsity, discoverable at initialization rather than after training.

#### 4. Bangla NLP Review
Bangla, despite being the world's sixth most spoken language with 230 million native speakers, remains a low-resource language in NLP. Transformer-based models show promising results but involve computational trade-offs.

---

## Methodology

![Methodology Flowchart](https://github.com/KsLimon/Training-Pruned-Language-Model-And-Diverse-Dataset/blob/master/A%20Final%20Paper/img1.png)
*Figure 1: Complete methodology workflow showing the process from base model to pruned model*

Our approach consists of three main phases:

### A. Iterative Pattern Exploiting Training (iPET)

For a masked language model M with vocabulary T:

**Pattern-Verbalizer Pairs (PVPs):** Each PVP p = (P, v) consists of:
- Pattern P: X → T* that maps inputs to cloze questions with a single mask
- Verbalizer v: Y → T that maps outputs to tokens representing task-specific meaning

**Probability Distribution:**

```
q_p(y|x) = exp(s_p(y|x)) / Σ_y' exp(s_p(y'|x))
```

Where `s_p(y|x) = s¹_M(v(y)|p(x))` is the raw score of v(y) at the masked position.

**Ensemble Approach:**

```
q_p(y|x) ∝ exp(Σ_p∈P w_p · s_p(y|x))
```

Where w_p is proportional to accuracy achieved with p on the training set.

### B. Pruning Using Lottery Ticket Hypothesis

**Network Definition:**  
For a network f(x; θ, ·), a subnetwork f(x; m ⊙ θ, ·) with pruning mask m ∈ {0,1}^d

**Matching Subnetwork:**  
A subnetwork is matching if:

```
ε_T(A^T_t(f(x; m⊙Φ, θ, γ))) ≥ ε_T(A^T_t(f(x; θ_0, γ)))
```

**Winning Ticket:**  
A subnetwork is a winning ticket if it's a matching subnetwork and θ = θ_0

**Universal Subnetwork:**  
A subnetwork f(x; m ⊙ θ, γ^T_0) is universal for tasks {T_i}^N_i=1 if it matches each A^T_i_t_i

### C. Implementation Strategy

1. **Fine-tune** a tiny BERT model (ckiplab/albert-tiny-chinese) from scratch using diverse Bangla dataset
2. **Apply iPET** for few-shot learning capability
3. **Prune the model** using Lottery Ticket Hypothesis:
   - Fine-tune on Masked Language Modeling
   - Drop 10% of network weights per iteration
   - Rewind from last checkpoint
   - Continue until 90% sparsity rate achieved
4. **Evaluate** on downstream tasks

**Training Parameters:**
- Batch size: 32
- Logging interval: Every 5093 steps
- Pruning rate: 10% per iteration
- Final sparsity: 90%

---

## Experiments

### Dataset Construction

![Dataset Construction Process](https://github.com/KsLimon/Training-Pruned-Language-Model-And-Diverse-Dataset/blob/master/A%20Final%20Paper/img2.png)
*Figure 2: Procedure for constructing the diverse Bangla dataset*

**Pretraining Corpus:**
- **Total Size:** 80 MB
- **Data Split:** 80% training, 20% testing
- **Sources:** 
  - Social media text
  - Bangla newspapers
  - Bangla Shadhu text (classical literature)

**Data Processing:**
1. Collected PDF files
2. Extracted Bangla text using OCR
3. Replaced contaminated text with white-space
4. Preprocessed to BERT format (one sentence per line)

### Downstream Task Datasets

#### 1. Emotion Classification
- **Classes:** 5 emotions (fear/surprise, joy, sadness, anger/disgust, none)
- **Source:** 2,890 YouTube comments
- **Languages:** Bangla, English, romanized Bangla
- **Split:** 80% train, 20% dev, 10% test

#### 2. Authorship Classification
- **Authors:** 14 distinct writers
- **Document Size:** 750 words per document
- **Total:** 14,047 train / 3,511 dev / 750 test

#### 3. News Categorization
- **Classes:** 6 categories
- **Total:** 11,109 train / 1,408 dev / 1,407 test

#### 4. Sentiment Classification

**Multiple Datasets Combined:**

| Dataset | Train | Dev | Test | Classes |
|---------|-------|-----|------|---------|
| YouTube Comments | 1,660 | 297 | 273 | 3 |
| BengFastText | 5,253 | 1,228 | 1,295 | Multiple |
| SAIL | 697 | 99 | 204 | 3 |
| ABSA Cricket | 1,943 | 373 | 372 | 3 |
| ABSA Restaurant | 1,188 | 225 | 209 | 3 |
| CogniSenti | 4,599 | 985 | 986 | Multiple |
| **Combined** | 4,807 | 1,031 | 1,031 | 3 |

#### 5. POS Tagging

Three progressive datasets:
- **LDC Corpus:** 7,393 sentences / 102,937 tokens
- **LDC+IITKGP:** Additional 5,473 sentences / 72,400 tokens
- **LDC+IITKGP+CRBLP:** Additional 1,176 sentences / 20,000 tokens

#### 6. Punctuation Restoration
- **Train:** 1,379,986 tokens
- **Dev:** 179,371 tokens
- **Test Sets:** News (87,721), Ref (6,821), ASR (6,417)
- **Labels:** 4 (comma, period, question mark, O token)

### Model Configurations

| Model | Attention Heads | Hidden Layers | Hidden Size | Vocab Size | Parameters |
|-------|----------------|---------------|-------------|------------|------------|
| **Bnbert (ours)** | 12 | 4 | 312 | 30,522 | ~4M |
| **Bnbert iPET (ours)** | 12 | 4 | 312 | 30,522 | ~4M |
| **Bnbert iPET Pruned (ours)** | 12 | 4 | 312 | 30,522 | ~0.4M |
| BanglaBert | 12 | 12 | 768 | 32,000 | ~110M |
| Bangla-Electra | 4 | 12 | 256 | 29,898 | ~14M |
| Indic-BERT | 12 | 12 | 768 | 100,000 | ~110M |
| BERT-bn | 12 | 12 | 768 | 102,025 | ~110M |
| XLM-RoBERTa | 16 | 24 | 1,024 | 250,002 | ~550M |

### Evaluation Metrics

**Perplexity (PP):** Measures how well the probability model predicts samples

```
PP(W) = ᴺ√(∏ᴺᵢ₌₁ 1/P(Wᵢ|Wᵢ₋₁))
```

**Classification Metrics:**
- Accuracy (Acc)
- Precision (P) - weighted average
- Recall (R) - weighted average  
- F1 Score - weighted average

---

## Results

### Perplexity Comparison

#### Dataset-wise Perplexity

| Dataset | Eval Loss | Perplexity |
|---------|-----------|------------|
| Full Dataset | 2.28 | 9.43 |
| Sadhu | 2.29 | 9.88 |
| Social Media | 2.36 | 10.52 |
| Newspaper | 2.24 | 9.29 |

#### Model-wise Perplexity

| Model | Perplexity |
|-------|------------|
| Base Model | 3.8722 |
| **iPET Model** | **1.0460** ⭐ |
| Pruned Model | 46.8558 |

*iPET methodology significantly improves perplexity compared to base model*

### Downstream Task Performance

#### 1. Emotion Classification

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Bnbert (ours) | 45.51 | 43.06 | 45.51 | 41.21 |
| Bnbert iPET (ours) | 43.71 | 41.08 | 43.71 | 39.84 |
| **Bnbert iPET Pruned (ours)** | **50.30** | **46.86** | **50.30** | **48.10** |
| Bangla Electra | 43.8 | 38.3 | 43.8 | 36.3 |
| Indic BERT | 50.6 | 52.1 | 50.6 | 49.1 |
| BERT-bn | 49.1 | 46.7 | 49.1 | 46.9 |
| BanglaBert | 71.05 | 66.46 | 71.05 | 68.62 |

**Key Finding:** Our pruned model outperforms Bangla Electra and performs comparably to Indic-BERT

#### 2. Authorship Classification

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Bnbert (ours) | 97.11 | 97.14 | 97.11 | 97.11 |
| **Bnbert iPET (ours)** | **97.72** | **97.76** | **97.72** | **97.73** |
| Bnbert iPET Pruned (ours) | 97.67 | 97.69 | 97.67 | 97.68 |
| XLM-RoBERTa | 93.8 | 94.1 | 93.8 | 93.8 |
| Indic BERT | 95.2 | 95.3 | 95.2 | 95.2 |
| BERT-bn | 90.2 | 90.3 | 90.2 | 90.2 |
| BanglaBert | 98.85 | 97.71 | 97.88 | 98.71 |

**Key Finding:** Our models achieve results comparable to BanglaBert while being significantly smaller

#### 3. News Categorization

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Bnbert (ours) | 82.51 | 83.03 | 82.51 | 81.50 |
| Bnbert iPET (ours) | 84.29 | 84.35 | 84.29 | 84.08 |
| **Bnbert iPET Pruned (ours)** | **88.48** | **88.24** | **88.48** | **88.19** |
| Bangla Electra | 80.4 | 78.5 | 80.4 | 79.2 |
| Indic-DistilBERT | 89.0 | 90.2 | 89.0 | 89.4 |
| DistilBERT-m | 79.5 | 79.4 | 79.5 | 79.0 |
| BanglaBert | 94.52 | 94.55 | 94.52 | 94.53 |

**Key Finding:** Pruned model performs nearly as well as Indic-DistilBERT

#### 4. Sentiment Classification (Combined Dataset)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Bnbert (ours) | 77.91 | 79.21 | 77.91 | 77.62 |
| **Bnbert iPET (ours)** | **79.52** | **79.75** | **79.52** | **79.10** |
| Bnbert iPET Pruned (ours) | 78.52 | 78.49 | 78.52 | 78.41 |
| Bangla Electra | 72.0 | 72.4 | 72.0 | 71.8 |
| DistilBERT-m | 74.3 | 74.8 | 74.3 | 74.2 |
| BERT-bn | 79.3 | 79.3 | 79.3 | 79.2 |
| BanglaBert | 95.43 | 95.76 | 95.43 | 95.50 |

**Key Finding:** All our models outperform Bangla Electra and DistilBERT-m

#### 5. POS Tagging (LDC+IITKGP+CRBLP)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Bnbert (ours) | 78.08 | 70.91 | 69.38 | 69.30 |
| Bnbert iPET (ours) | 77.73 | 70.36 | 69.15 | 69.24 |
| **Bnbert iPET Pruned (ours)** | **82.01** | **75.51** | **74.78** | **82.01** |
| Bangla Electra | 80.4 | 75.1 | 73.0 | 74.7 |
| DistilBERT-m | 83.8 | 78.4 | 78.0 | 78.20 |
| BERT-bn | 85.4 | 81.0 | 80.2 | 80.6 |
| BanglaBert | 92.08 | 88.97 | 89.59 | 89.22 |

**Key Finding:** Pruned model performs comparably to Bangla Electra

#### 6. Punctuation Restoration

**News Dataset:**

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Bnbert (ours) | 94.32 | 98.38 | 96.31 |
| Bnbert iPET (ours) | 87.94 | 95.20 | 93.76 |
| **Bnbert iPET Pruned (ours)** | **94.55** | **98.23** | **96.36** |
| XLM-RoBERTa | 87.8 | 86.2 | 87.0 |
| Indic-BERT | 73.9 | 70.5 | 72.2 |
| Bangla Electra | 64.8 | 49.7 | 56.3 |
| BanglaBert | 94.77 | 98.21 | 96.46 |

**Key Finding:** Our models significantly outperform XLM-RoBERTa, Indic-BERT, and Bangla Electra

### Inference Time Comparison

| Task | Training Time | Inference Time |
|------|---------------|----------------|
| **Sentiment (Combined)** | 1h 40m 39s | 15s |
| **Authorship** | 1h 8m 34s | 16s |
| **Emotion** | 3m 14s | ~4s |
| **News** | 58m 8s | 5s |
| **SVM (baseline)** | 2h 32m 30s | 4s |
| **XLM-RoBERTa** | 3h 42m 41s | 1m 21s |

**Key Findings:**
- Our model inference time is **5.4× faster** than XLM-RoBERTa
- Our model inference time is 3.75× slower than SVM but offers better accuracy
- Training time is competitive with large models while offering better efficiency

---

## Key Advantages

### Environmental Benefits
- ✅ **90% reduction** in model parameters
- ✅ **Significantly lower** carbon footprint
- ✅ **Reduced energy** consumption during training and inference

### Computational Efficiency
- ✅ **5.4× faster inference** compared to XLM-RoBERTa
- ✅ Can run on **resource-constrained devices**
- ✅ **Lower memory requirements** for deployment

### Performance
- ✅ **Competitive accuracy** with much larger models
- ✅ **Few-shot learning** capability
- ✅ **Comparable F1 scores** across multiple tasks

---

## Conclusion

This research demonstrates that:

1. **Smaller models can be few-shot learners** through iterative pattern exploitation training
2. **iPET significantly improves perplexity** (from 3.87 to 1.05)
3. **90% pruning using Lottery Ticket Hypothesis** maintains competitive performance
4. **Pruned models perform comparably** to base models in downstream tasks
5. The model is **lighter, greener, and more efficient** for training and deployment

### Future Work

- Extend methodology to other low-resource languages
- Investigate optimal pruning rates for different tasks
- Explore dynamic pruning strategies
- Develop automated pattern generation for iPET
- Optimize for edge device deployment

---

## Citation

If you use this work, please cite:

```bibtex
@article{samad2024training,
  title={Training Pruned Language Model},
  author={Samad, MD Kamrus and Ghosh, Anan and Hossain, Sajib and Mohammed, Nabeel},
  journal={IEEE Transactions},
  year={2024},
  publisher={IEEE}
}
```
---

## Acknowledgments

- North South University Department of Electrical and Computer Engineering
- The Bangla NLP research community
- Contributors to the datasets used in this research

---

## Contact

For questions and feedback:

- 📧 Email: kamrus.samad@northsouth.edu
- 🏛️ Institution: North South University, Bangladesh

---

**Keywords:** Bangla Language Processing, Benchmarks, Dataset, iPET, Pruning, Text Classification, Token Classification, Transformer Models, BERT, Lottery Ticket Hypothesis, Few-Shot Learning, Low-Resource Languages

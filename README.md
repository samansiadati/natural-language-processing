# Natural Language Processing: Classical to Modern

## Companion Repository

This repository is a **study companion and practical implementation guide** for the book ***Natural Language Processing: Classical to Modern***.

The goal of this project is to bridge the gap between **NLP theory** and **hands-on implementation** by providing:

* Clear explanations of classical and modern NLP concepts
* Step-by-step derivations and algorithm walkthroughs
* Python implementations from first principles
* Visualizations to understand text processing and model behavior
* Original exercises with worked solutions

> 📌 **Important note**: This repository does **not** contain the book itself, nor does it reproduce copyrighted content. All explanations, code, and exercises are original and written as a learning aid.

---

## 🎯 Who This Repository Is For

This repo is designed for:

* Students learning **NLP, machine learning, or AI**
* Practitioners who want to **strengthen NLP foundations**
* Engineers preparing for **NLP-focused interviews**
* Researchers who want runnable examples of classical and modern NLP techniques

If you have ever thought *“I know NLP libraries, but I want to understand the algorithms behind them”*, this repository is for you.

---

## 🧠 Core Topics Covered

The repository follows a structure aligned with standard NLP curricula:

* **Text Preprocessing & Tokenization** (cleaning, stemming, lemmatization)
* **Classical NLP Algorithms** (n-grams, TF-IDF, bag-of-words, naive Bayes)
* **Syntactic & Semantic Analysis** (POS tagging, parsing, word embeddings)
* **Sequence Modeling** (RNNs, LSTMs, GRUs)
* **Attention Mechanisms & Transformers** (self-attention, BERT, GPT)
* **NLP Tasks** (text classification, sentiment analysis, named entity recognition)
* **Evaluation & Metrics** (BLEU, ROUGE, perplexity)

Each topic is treated with:

* Mathematical and computational rigor
* Hands-on Python examples
* Practical relevance for real-world NLP applications

---

## 📂 Repository Structure

```text
natural-language-processing/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── 01-text-preprocessing/
│   ├── README.md
│   ├── tokenization.ipynb
│   ├── stemming_lemmatization.ipynb
│   └── exercises.md
│
├── 02-classical-nlp/
│   ├── ngrams_tfidf.ipynb
│   ├── bag_of_words.ipynb
│   └── naive_bayes.ipynb
│
├── 03-syntactic-semantic-analysis/
│   ├── pos_tagging.ipynb
│   ├── parsing.ipynb
│   └── word_embeddings.ipynb
│
├── 04-sequence-modeling/
│   ├── rnn_basics.ipynb
│   ├── lstm.ipynb
│   └── gru.ipynb
│
├── 05-attention-transformers/
│   ├── self_attention.ipynb
│   ├── transformer_basics.ipynb
│   └── bert_gpt.ipynb
│
├── 06-nlp-tasks/
│   ├── text_classification.ipynb
│   ├── sentiment_analysis.ipynb
│   └── named_entity_recognition.ipynb
│
├── 07-evaluation-metrics/
│   ├── bleu_rouge.ipynb
│   └── perplexity.ipynb
│
└── utils/
    └── plotting.py

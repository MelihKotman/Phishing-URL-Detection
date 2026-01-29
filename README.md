# 🛡️ Phishing URL Detection: Classical ML vs. Deep Learning

> **Project Goal:** To develop a robust cybersecurity system capable of detecting malicious URLs by comparing **Classical Machine Learning (Random Forest)** and **State-of-the-Art Deep Learning (CNN)** approaches, ultimately culminating in a **Hybrid Production Engine**.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Random_Forest-yellow?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/Data-SQLite-green?style=for-the-badge&logo=sqlite&logoColor=white)

---

## 📖 Project Overview

Phishing attacks are evolving, and static blocklists are no longer sufficient. This project explores two distinct paradigms to solve this problem using a dataset of **~550,000 URLs**:

1.  **Phase 1: Feature-Based Engineering (Random Forest):** Extracting mathematical features (entropy, length, symbol count) to classify URLs based on statistical properties.
2.  **Phase 2: Character-Level NLP (Deep Learning):** Treating URLs as raw text sequences and using **1D-CNNs** to learn morphological patterns automatically.
3.  **Final Product:** A **Hybrid Engine** that combines a Rule-Based Whitelist with the Deep Learning model for maximum efficiency, speed, and security.

---

## ⚔️ Comparative Analysis: Which Approach Won?

I implemented and tested both methodologies to find the best solution. Here is the technical breakdown:

| Feature | 🌲 Approach 1: Random Forest | 🧠 Approach 2: Deep Learning (CNN) |
| :--- | :--- | :--- |
| **Methodology** | Manual Feature Extraction | Character-Level Embeddings |
| **Input Data** | Numerical (Length, Entropy, etc.) | Raw Text Sequence (Tokenized) |
| **Data Structure** | CSV (Static File) | SQLite Database (Dynamic) |
| **Accuracy** | **~86.26%** | **~96.50%** 🏆 |
| **Key Strength** | Interpretable (We know *why*) | Pattern Recognition (High Accuracy) |
| **Weakness** | Manual feature selection is tedious | Requires more computational power |

> **Conclusion:** While Random Forest provided solid baselines, the **Deep Learning model outperformed it significantly** by capturing hidden sequential patterns that manual features missed.

---

## 🏗️ System Architecture

### 1. The Classical Approach (Random Forest)
Located in `src_csv/`, this module relies on statistical properties of URLs:
* **Shannon Entropy:** Calculates randomness (e.g., `xf3a-9b.com` has high entropy compared to `google.com`).
* **Structural Features:** URL length, count of dots (.), dashes (-), and special characters.
* **Lexical Features:** Frequency of sensitive words (`secure`, `login`, `bank`, `update`).

### 2. The Modern Approach (Deep Learning)
Located in `src_db/`, this module uses **Natural Language Processing (NLP)**:
* **Tokenizer:** Converts URLs into sequences of integers based on character frequency.
* **Embedding Layer:** Maps characters to dense vectors.
* **Conv1D Layer:** Scans the URL like an image to find local correlations (e.g., `.exe` inside a path or DGA patterns).

### 3. The Final Hybrid Engine (Production Ready)
The `predict_deep_model.py` script implements a real-world logic:
1.  **Whitelist Filter:** Checks if the domain is a known giant (e.g., `google.com`, `ibu.edu.tr`). If yes, bypass AI (Zero Latency).
2.  **AI Analysis:** If the domain is unknown, the CNN model scans the URL for threats.

---

## 📊 Exploratory Data Analysis (EDA) Insights

Before modeling, extensive analysis revealed key phishing behaviors:
* **Length Anomaly:** Phishing URLs are on average **40% longer** than safe URLs to hide the real domain.
* **Keyword Hijacking:** Attackers heavily abuse words like `security`, `update`, and `verify`.
* **Obfuscation:** High usage of `@` symbols and hyphens (`-`) to trick users.

---

## 📂 Project Structure

```text
Phishing-URL-Detection/
├── data/
│   ├── phishing_final.db        # SQLite Database (Used for Deep Learning)
│   └── phishing_site_urls.csv   # Raw Data (Used for Random Forest)
├── experiments/                 # Execution Scripts
│   ├── main_RF.py               # Runs the Random Forest Training
│   ├── main_deep_model.py       # Runs the Deep Learning Training
│   └── predict_deep_model.py    # Real-Time Hybrid Prediction Tool
├── models/                      # Saved AI Models (.keras & .pickle)
├── src_csv/                     # CLASSICAL ML Source Code
│   ├── feature_extraction.py    # Math/Regex logic for RF
│   └── model/classical.py       # Scikit-Learn implementation
├── src_db/                      # DEEP LEARNING Source Code
│   ├── database_connection/     # SQLite Managers
│   └── model/deep_model_db.py   # Keras/TensorFlow Architecture
└── visualization/               # EDA Charts & Graphs
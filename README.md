# 🛡️ Phishing URL Detection System (Hybrid AI Engine)

> **Objective:** To build a real-time, hybrid machine learning system capable of detecting malicious (phishing) URLs by analyzing their lexical, structural, and behavioral properties.

This project focuses on **Cybersecurity Data Science**. Using a dataset of over **500,000 URLs**, I developed a system that combines **Random Forest Classification** with a **Rule-Based Whitelist** engine. The model analyzes URL patterns, entropy, and linguistic features to distinguish between safe and dangerous websites with high accuracy.

---

## 📊 Key Findings from Exploratory Data Analysis (EDA)

Before modeling, extensive visual analysis was performed on the dataset. Key insights that shaped the model include:

### 1. The "Length" Anomaly
* **Insight:** Phishing URLs are significantly longer than safe ones. Attackers often use long subdomains to hide the actual destination.
    * *Safe Average Length:* ~45 chars
    * *Phishing Average Length:* ~63 chars

### 2. Lexical Hijacking (Word Frequency)
* **Insight:** Phishing URLs rely heavily on "urgency" and "authority" keywords to trick users.
    * **Phishing Keywords:** `login`, `secure`, `update`, `account`, `paypal`, `verify`.
    * **Safe Keywords:** `wiki`, `search`, `news`, `images`, `blog`.

### 3. Structural Obfuscation
* **Insight:** Attackers use an excessive amount of special characters (`-`, `.`, `@`) and deep file paths (`/`) to mimic legitimate domains.

---

## ⚙️ Technical Architecture & Features

The system uses a **Hybrid Approach** to minimize false positives (e.g., blocking university sites) while maximizing detection rates.

### 1. Feature Engineering (The "Brain" of the Model)
Instead of treating URLs as simple text, the system extracts **mathematical features**:
* **Shannon Entropy:** Calculates the randomness of the URL to detect algorithmically generated domains (DGA).
* **Digit Ratio:** Measures the density of numbers in the URL (higher in phishing).
* **Suspicious Keywords:** Checks for the presence of high-risk words found during EDA.
* **IP Address Check:** Detects raw IP usage (e.g., `http://192.168.1.1`).
* **Shortening Services:** Identifies hidden URLs using services like `bit.ly` or `tinyurl`.

### 2. The Hybrid Engine (AI + Whitelist)
* **Layer 1 (Whitelist):** Checks if the domain is in a trusted list (e.g., `google.com`, `ibu.edu.tr`) to prevent false alarms.
* **Layer 2 (Random Forest Model):** If not whitelisted, the AI model analyzes the features and calculates a probability score.

---

## 📈 Model Performance

The model was trained on **549,346 URLs** using a Random Forest Classifier with class balancing.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **86.26%** | Overall correctness of the model. |
| **Recall (Bad)** | **78.00%** | The ability to catch actual phishing attacks. |
| **Precision (Good)**| **91.00%** | Trustworthiness when labeling a site as safe. |

---

## 🛠 Technologies Used

* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
* **Feature Extraction:** Regex (Re), Math, Collections

---

## 📂 Project Structure

```text
Phishing-URL-Detection/
├── data/                   # Dataset (excluded via .gitignore)
├── visualization/          # Generated EDA plots (PNGs)
├── src/
│   ├── feature_extraction.py   # Extracts Entropy, IP, Length, etc.
│   ├── visualization.py        # Plotting functions
│   └── model.py                # Model training & evaluation logic
├── main.py                 # Main execution script (CLI Tool)
└── README.md               # Project documentation
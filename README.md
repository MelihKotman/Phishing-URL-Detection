# 🛡️ Phishing URL Detection System

> **Objective:** To build a machine learning model capable of detecting malicious (phishing) URLs by analyzing their lexical and structural properties.

This project focuses on **Cybersecurity Data Science**. Using a dataset of over **500,000 URLs**, we apply Exploratory Data Analysis (EDA) and Natural Language Processing (NLP) to uncover patterns used by attackers. The ultimate goal is to extract these patterns as "features" and train a classification model (e.g., Random Forest) to predict whether a URL is safe or dangerous in real-time.

---

## 📊 Key Insights from Exploratory Data Analysis (EDA)

Before modeling, we performed extensive visual analysis to understand the behavior of phishing URLs. Here are the **5 major findings**:

### 1. Class Distribution & Imbalance
* **Analysis:** We checked the ratio of "Good" vs. "Bad" URLs.
* **Insight:** Real-world data is often imbalanced. Understanding this helps in choosing the right evaluation metrics (F1-Score vs. Accuracy) later in the project.

### 2. URL Length Distribution
* **Analysis:** Comparison of character counts in Safe vs. Phishing URLs.
* **Insight:** Phishing URLs tend to be **significantly longer**. Attackers often use long subdomains or complex file paths to hide the actual destination and trick users.

### 3. Special Character Analysis (`-`, `.`, `/`, `@`)
* **Analysis:** We counted the frequency of special characters.
* **Insight:**
    * **Hyphens (`-`):** Frequently used in phishing to mimic legitimate domains (e.g., `paypal-secure-login`).
    * **Slashes (`/`):** Higher count in phishing URLs due to deep file paths used to obfuscate the domain.
    * **Dots (`.`):** Excessive use of dots often indicates multiple subdomains (e.g., `secure.login.bank.com.fake.site`).

### 4. Top-Level Domain (TLD) Breakdown
* **Analysis:** Which extensions are most dangerous? (`.com` vs `.xyz`)
* **Insight:** While `.com` is common in both, attackers often leverage cheap or less regulated TLDs (like `.xyz`, `.top`, `.site`) for disposable phishing campaigns.

### 5. Word Frequency (NLP Analysis)
* **Analysis:** Tokenized URLs to find the most common words.
* **Insight (The "Human Factor"):**
    * **Phishing URLs:** Heavily rely on *urgency* and *authority*. Top words: `login`, `secure`, `account`, `update`, `verify`, `paypal`, `battle`.
    * **Safe URLs:** Contain content-specific terms. Top words: `wiki`, `search`, `news`, `images`, `en`.

---

## 🛠 Technologies & Tools

* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
* **Feature Engineering:** Re (Regex), Collections
* **Machine Learning (Planned):** Scikit-Learn (Random Forest / XGBoost)

---

## 🚀 Future Roadmap (To-Do List)

To further improve the detection accuracy, the following advanced features will be analyzed and implemented:

- [ ] **IP Address Detection:** Check if the URL uses a raw IP address (e.g., `http://192.168.1.1`) instead of a domain name.
- [ ] **Shannon Entropy Analysis:** Calculate the randomness of the URL string to detect algorithms (DGA) used by botnets.
- [ ] **Shortening Service Detection:** Identify URLs wrapped in shorteners like `bit.ly` or `tinyurl`.
- [ ] **Model Training:** Train Random Forest and Logistic Regression models using the extracted features.
- [ ] **Feature Importance:** Visualize which feature (Length vs. Keywords) contributes most to the detection.

---

## 📂 Project Structure

```text
Phishing-URL-Detection/
├── data/                  # Dataset files (excluded from repo via .gitignore)
├── visualization/         # Saved plot images (PNGs)
├── src/
│   ├── visualization.py   # EDA functions (Plots & Graphs)
│   └── feature_extraction.py # Feature engineering modules (Regex & Math)
├── main.py                # Main execution script
└── README.md              # Project documentation
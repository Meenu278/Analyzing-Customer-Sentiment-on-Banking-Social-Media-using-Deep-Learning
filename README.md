# 📊 Master’s Thesis: Analyzing Customer Sentiment on Banking Social Media using Deep Learning

## 📌 Project Overview

This repository contains the research work and implementation developed for my Master’s Thesis:
**“Analyzing Customer Sentiment on Banking Social Media using Deep Learning”**.

The study investigates how **machine learning (ML)** and **deep learning (DL)** models can be applied to classify sentiments in banking-related Twitter data. The aim is to provide actionable insights for banks to enhance **customer satisfaction**, **service delivery**, and **reputation management**.


## 🎯 Research Objectives

1. Develop a framework to classify customer sentiment (positive, negative, neutral) from Twitter data.
2. Compare traditional ML methods (Naive Bayes, SVM) with advanced DL approaches (LSTM, CNN, BERT).
3. Identify recurring **themes and trends** in banking-related conversations.
4. Offer recommendations for banks to leverage sentiment insights for improved customer engagement.


## 📊 Key Findings

* **BERT** significantly outperformed all other models in terms of accuracy and F1-score.
* **Negative sentiment** dominates, primarily around *service quality, transactions, and customer support*.
* Misclassifications were mainly caused by **sarcasm, slang, and ambiguous language**.
* Explainable AI techniques (e.g., word importance visualization) proved valuable for interpreting model decisions.



## 🏗️ Repository Structure

```
├── Twitter.Meenakshi.Assisment.Final-GH1037473.ipynb   # Jupyter Notebook with implementation
├── README.md                                           # Project documentation (this file)
```



## ⚙️ Methodology

* **Data Source**: Twitter API (bank-related hashtags and keywords, English only).
* **Preprocessing**: Text normalization, stopword removal, handling slang/hashtags, anonymization.
* **Class Imbalance**: Addressed using **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Models Compared**:

  * Traditional ML: Naive Bayes, Support Vector Machine (SVM)
  * Deep Learning: LSTM, CNN
  * Transformers: BERT (Bidirectional Encoder Representations from Transformers)
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix


## 📂 Installation & Usage
## Data set link :
https://www.kaggle.com/datasets/pypiahmad/twitter-sentiment-analysis

### Requirements

* Python 3.8+
* Jupyter Notebook
* Key libraries:
  `numpy`, `pandas`, `scikit-learn`, `tensorflow`/`pytorch`, `transformers`, `imbalanced-learn`, `matplotlib`, `seaborn`

### Steps

1. Clone this repository:

   ```bash
 git clone https://github.com/Meenu278/Analyzing-Customer-Sentiment-on-Banking-Social-Media-using-Deep-Learning.git
cd Analyzing-Customer-Sentiment-on-Banking-Social-Media-using-Deep-Learning

   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:

   ```bash
   jupyter notebook Twitter.Meenakshi.Assisment.Final-GH1037473.ipynb
   ```
4. Run the cells sequentially to reproduce the experiments and results.



## 🔒 Ethical Considerations

* Only **publicly available tweets** were used.
* User identities were anonymized or pseudonymized.
* Research conducted in compliance with **ethical guidelines** and **Twitter’s terms of service**.



## 📌 Future Directions

* Extend to **multilingual sentiment analysis**.
* Explore **multimodal analysis** (text + images + audio).
* Develop a **real-time sentiment monitoring dashboard** for banks.
* Investigate **hybrid ML + DL models** tailored for domain-specific financial language.


## 👩‍💻 Author

**Meenakshi Ashok**
Master’s Thesis in Analyzing Customer Sentiment on Banking Social Media using Deep Learning

GISMA UNIVERSITY OF APPLIED SCIENCE ,POTSDAM , GERMANY


## 📚 Citation

If you use this work, please cite it as:

```
Meenakshi Ashok, "Analyzing Customer Sentiment on Banking Social Media using Deep Learning." Master’s Thesis, [GISMA UNIVERSITY OF APPLIED SCIENCE], [2025].
```


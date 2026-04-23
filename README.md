
---

# 📄 Resume Classification System

### Machine Learning Project using TF-IDF & NLP

---

## 🚀 Overview

This project implements a machine learning pipeline for automatically classifying resumes into job categories.

Using Natural Language Processing (NLP), the system transforms raw text into numerical features and applies supervised learning to predict the category of each resume.

---

## 🎯 Objectives

* Convert unstructured text into numerical features
* Train a classification model
* Evaluate performance on unseen data
* Analyze strengths and limitations

---

## 📊 Dataset

| Feature      | Description               |
| ------------ | ------------------------- |
| Clean_Resume | Processed resume text     |
| Category     | Target job category label |

---

## ⚙️ Methodology

### 1️⃣ Data Splitting

We split the dataset into training and testing sets (80/20):

```
X = df['Clean_Resume']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### 2️⃣ Feature Engineering (TF-IDF)

Text data is transformed into numerical vectors using TF-IDF:

```
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

🔹 Keeps the 1000 most important words

🔹 Removes common words like "the", "and", "is"

Example:

```
print(vectorizer.get_feature_names_out()[:20])
```

---

### 3️⃣ Model Training

After vectorization, classification models can be applied:

* Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)

---

### 4️⃣ Evaluation Metrics

| Metric    | Description                      |
| --------- | -------------------------------- |
| Accuracy  | Overall correctness              |
| Precision | Quality of predictions           |
| Recall    | Ability to find true positives   |
| F1-score  | Balance between precision/recall |

---

## 📈 Results

* Baseline Accuracy: ~58%
* The model captures general patterns but still needs improvement

---

## ⚠️ Challenges

* Multi-class classification
* Imbalanced dataset
* Similar categories with overlapping vocabulary

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Use n-grams (bi-grams, tri-grams)
* Try advanced models (XGBoost, Neural Networks)
* Improve preprocessing

---

## 📂 Project Structure

```
project/
│── data/
│── notebook.ipynb
│── model_training.py
│── README.md
```

---

## ▶️ How to Run

Install dependencies:

```
pip install pandas scikit-learn  
```

Run the project:

1. Load dataset
2. Split data
3. Apply TF-IDF
4. Train model
5. Evaluate results

---

## 🧠 Key Insights

* TF-IDF is a strong baseline for NLP
* Feature quality is critical
* Preprocessing has a huge impact on results

---

## 👩‍💻 Author
 
• Neta Gooldzad  
• Eden Zarbian

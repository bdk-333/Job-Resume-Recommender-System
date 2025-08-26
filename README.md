
# 📄 Job Resume Analysis & Recommender System

## 🚀 Project Overview

Recruiters often spend only a few seconds scanning resumes, making manual screening time-consuming and inconsistent. This project, **Job Resume Analysis & Recommender System**, leverages **Natural Language Processing (NLP)**, **feature engineering**, and **machine learning** to automate resume-job matching and provide better recommendations for candidates.

We worked with a **Kaggle Resume Dataset** containing **344 resumes** and **28 job descriptions**, producing **9,544 unique resume-job combinations**. Our project examined **content-based filtering, collaborative filtering, and regression models** to improve job recommendations.

---

## 📊 Dataset

* Source: [Kaggle Resume Dataset](https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset)
* **Resumes (344 unique)** → Candidate education, skills, experiences, certifications, activities.
* **Jobs (28 unique)** → Titles, required skills, education, and experience.
* **Final dataset size**: 9,544 resume-job combinations.

---

## ⚙️ Methodology

### 🔹 Data Preprocessing

* Text cleaning: lowercasing, punctuation removal, stopword removal, lemmatization.
* Handling missing values.
* Synonym & acronym normalization (e.g., *ML → Machine Learning*).
* Feature engineering:

  * Skill overlap ratio
  * Experience match flag
  * Highest education level
  * Overlapping words count

### 🔹 NLP & Vectorization

* **TF-IDF with cosine similarity** for textual similarity.
* **Word clouds** & **N-gram analysis** for skill/keyword exploration.
* **PCA** for dimensionality reduction.

### 🔹 Machine Learning Models

* **Regression Models**: Linear, Ridge, KNN, Random Forest, XGBoost, Gradient Boosting.
* **Clustering**: K-means to group resumes into 3 main clusters:

  * Machine Learning & Data Science
  * Engineering & Management
  * Accounting & Finance
* **Recommendation Systems**:

  * Content-based (TF-IDF + engineered features)
  * Collaborative filtering (Cosine similarity & SVD-based factorization)

---

## 📈 Results

* **Content-Based Filtering**:

  * Top-5 Accuracy → **35.5% suitable match**
  * Top-10 Accuracy → **51.7% suitable match**

* **Regression Models**:

  * Best model: **Gradient Boosting**
  * R²: **0.64** | RMSE: **0.10** | MAE: **0.08**

* **Collaborative Filtering**:

  * Cosine similarity → RMSE: **0.1407**
  * SVD-based CF → RMSE: **0.1033** (best performance)

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy, Scikit-learn**
* **Matplotlib, Seaborn, WordCloud**
* **XGBoost, Gradient Boosting**
* **TF-IDF, PCA, Cosine Similarity, SVD**

---

## 📂 Repository Structure

```markdown
├── data/                  # Raw and processed data files
├── notebooks/             # Jupyter notebooks with preprocessing, EDA, modeling
├── models/                # Trained models and embeddings
├── plots/                 # Word clouds, PCA plots, clustering results
├── README.md              # Project report (this file)
```

---

## 🎯 Conclusion

This project demonstrates how combining **NLP techniques**, **structured feature engineering**, and **recommendation algorithms** can create a scalable and efficient **AI-powered resume-job matching system**. It improves recruiter efficiency, reduces bias, and can be extended to real-world recruitment platforms.

---

✨ *Developed by Team Visionaries (Akanksha Singh, Kirtankumar Parekh, Krishna Kalakonda, Raj Patel)*

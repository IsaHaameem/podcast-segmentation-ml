# 🎙️ Podcast Listener Segmentation Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

An end-to-end machine learning system that segments podcast listeners into actionable behavioral cohorts using **Ward’s Hierarchical Clustering**.

The platform combines clustering, evaluation, and business interpretation to transform raw listening data into insights that can drive **personalization, retention, and content strategy**.

---

## ✨ Key Highlights

* **End-to-End ML Pipeline**
  From raw CSV → preprocessing → clustering → insights → visualization

* **Hierarchical Clustering (Ward’s Method)**
  Minimizes intra-cluster variance to produce compact, meaningful segments

* **Automated Cluster Evaluation**
  Uses **Silhouette Score** to determine optimal cluster count (K ∈ [2,10])

* **Business-Aware Segmentation**
  Converts clusters into interpretable personas like:

  * *Binge Listeners*
  * *Casual Users*
  * *Low Engagement Users*

* **Interactive Analytics Dashboard**
  Built with Streamlit + Plotly for real-time exploration

---

## 🧱 Tech Stack

| Layer                | Technology         |
| -------------------- | ------------------ |
| Frontend             | Streamlit          |
| Data Processing      | Pandas, NumPy      |
| Machine Learning     | Scikit-learn       |
| Scientific Computing | SciPy              |
| Visualization        | Plotly, Matplotlib |

---

## 📂 Project Structure

```text
project/
│
├── app.py                     # Streamlit dashboard (UI + orchestration)
├── generate_sample_data.py    # Synthetic dataset generator
├── requirements.txt           # Dependencies
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Data cleaning & feature engineering
│   ├── clustering.py          # Ward clustering implementation
│   ├── evaluation.py          # Silhouette scoring & optimal K
│   ├── insights.py            # Business logic & segmentation
│   ├── visualization.py       # Charts & dendrograms
│
└── data/
    └── sample.csv
```

---

## 📊 Data Schema

| Feature             | Type        | Description              |
| ------------------- | ----------- | ------------------------ |
| user_id             | String      | Unique listener ID       |
| listening_time      | Numeric     | Minutes listened         |
| episode_length_pref | Categorical | short / medium / long    |
| genre               | Categorical | Preferred genre          |
| skip_rate           | Numeric     | Fraction skipped (0–1)   |
| completion_rate     | Numeric     | Fraction completed (0–1) |
| frequency           | Categorical | daily / weekly / monthly |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/podcast-segmentation-ml.git
cd podcast-segmentation-ml
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Dataset

```bash
python generate_sample_data.py
```

### 5. Run the App

```bash
streamlit run app.py
```

---

## 📈 How It Works

### 1. Data Preprocessing

* Missing value imputation
* One-hot encoding for categorical variables
* Feature scaling using StandardScaler

### 2. Clustering

* Uses **Agglomerative Clustering (Ward linkage)**
* Minimizes variance within clusters

### 3. Evaluation

* Computes **Silhouette Score**
* Automatically identifies optimal K
* Visualizes score vs K

### 4. Insight Generation

* Aggregates cluster centroids
* Applies rule-based labeling for interpretability

---

## 📊 Dashboard Features

* 📌 Cluster size distribution
* 📌 Listening behavior scatter plots
* 📌 Dendrogram visualization
* 📌 Silhouette score analysis
* 📌 Segment-level business insights

---

## 🌐 Deployment

This app can be deployed easily using **Streamlit Community Cloud**:

1. Push repository to GitHub
2. Connect GitHub to Streamlit Cloud
3. Deploy `app.py`

---

## 💡 Use Cases

* Podcast platforms → audience segmentation
* Content creators → targeted content strategy
* Marketing teams → personalized campaigns
* Product teams → retention optimization

---

## 🧠 Key Learning Outcomes

* Applied hierarchical clustering in a real-world scenario
* Bridged ML outputs with business interpretation
* Built an interactive ML-powered analytics dashboard
* Designed a modular, scalable ML pipeline

---

## 👤 Author

**Muhammad Isa Haameem**
🔗 LinkedIn: https://linkedin.com/in/muhammad-isa-haameem-ba420834a
💻 GitHub: https://github.com/IsaHaaMEEM

---

> This project demonstrates the integration of machine learning, data engineering, and product thinking to deliver actionable insights from user behavior data.

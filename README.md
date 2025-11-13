
# 🧠 Performance Comparison of Traditional Machine Learning and Ensemble Models in Network Intrusion Detection

### 📘 Minor Project - 2 (Even Semester 2025)  
**Department of Computer Science and Information Technology**  
**Jaypee Institute of Information Technology, Noida**  
**Supervisor:** Dr. Meenal Jain  
**Contributors:** Ravi Raushan Vishwakarma, Viyom Shukla, Abhijeet Kumar  

---

## 📄 Project Overview

This project presents a **comparative study** of traditional machine learning models and ensemble learning algorithms for **Network Intrusion Detection Systems (NIDS)** using the **NSL-KDD dataset**.  
The main goal is to improve the detection of both **common** and **rare** cyberattacks by applying:
- **Recursive Feature Elimination (RFE)** for feature selection, and  
- **Conditional Generative Adversarial Networks (cGANs)** for synthetic data generation.

---

## 🎯 Objectives
- Evaluate and compare traditional ML models (SVM, Decision Tree, Logistic Regression, Random Forest) with ensemble models (AdaBoost, Gradient Boosting, XGBoost).  
- Handle data imbalance for rare attacks (R2L, U2R) using **synthetic data generation**.  
- Improve model robustness and accuracy through **feature selection** and **cross-validation**.  
- Measure model performance using **accuracy**, **precision**, **recall**, **F1-score**, and **AUC-ROC**.

---

## 🧩 Dataset
**Dataset Used:** [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)  
- 41 features per record  
- 4 main attack categories:
  - 🧨 Denial of Service (DoS)  
  - 🔍 Probe  
  - 🐚 Remote to Local (R2L)  
  - 🔑 User to Root (U2R)  
- Addressed **class imbalance** by augmenting minority classes using **Conditional GANs**.  

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Handled missing values and outliers  
- Encoded categorical features (`protocol_type`, `service`, `flag`)  
- Normalized numerical features  

### 2. Feature Selection
- Applied **Recursive Feature Elimination (RFE)** to remove irrelevant or redundant attributes.  

### 3. Synthetic Data Generation
- Used **Conditional GANs (cGANs)** to balance dataset by generating realistic attack samples.  

### 4. Model Training
- Implemented multiple models:
  - **Traditional:** Logistic Regression, Decision Tree, Random Forest, SVM  
  - **Ensemble:** AdaBoost, Gradient Boosting, XGBoost  

### 5. Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUC-ROC  

---

## 📊 Results Summary

| Attack Type | Best Model | Accuracy | AUC-ROC | Key Insight |
|--------------|-------------|-----------|----------|--------------|
| **DoS** | XGBoost | 0.93 | 0.93 | Strong detection due to pattern consistency |
| **Probe** | AdaBoost | 0.8946 | 0.94 | Effective at identifying probing activities |
| **U2R** | Gradient Boosting | 0.995 | 0.92 | High recall, handles rare patterns well |
| **R2L** | XGBoost | 0.78 | 0.93 | Hardest to detect due to class imbalance |

🧩 Ensemble models consistently **outperformed traditional ML** methods, particularly for rare attacks.

---

## 🧠 Key Findings
- **XGBoost** showed the highest robustness and recall for multiple attack types.  
- **Feature selection (RFE)** improved training efficiency and interpretability.  
- **Synthetic data (cGAN)** helped balance rare classes, improving F1-scores.  
- **Cross-validation** reduced overfitting and improved model generalization.

---

## 🚀 Future Enhancements
- Real-time intrusion detection using **streaming frameworks** (e.g., Apache Kafka, Spark Streaming).  
- Advanced imbalance handling using **SMOTE** or **ADASYN**.  
- Integration of **deep learning** (CNN, LSTM, Transformer) with ensemble models.  
- Deployment on **cloud-based environments** (AWS / Azure) for real-time monitoring.  
- Implementation of **Explainable AI (XAI)** for model interpretability.

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Deep Learning (cGAN)** | TensorFlow / Keras |
| **Visualization** | Matplotlib, Seaborn |
| **IDE** | Jupyter Notebook, VS Code |

---

## 📈 Performance Visualization
- Confusion Matrices  
- ROC Curves  
- Feature Importance Graphs  

---

---

## 🧑‍💻 Authors

| Name  | Role |
|------|------|
| **Ravi Raushan Vishwakarma** |  Data Preprocessing & Model Development |
| **Viyom Shukla** |  Data Augmentation & Visualization |
| **Abhijeet Kumar** | Feature Selection, Evaluation & Documentation |

---

## 🏁 Conclusion
This project demonstrates that **ensemble learning models**, particularly **XGBoost**, provide superior accuracy, robustness, and recall for **Network Intrusion Detection Systems**. The combination of **feature selection** and **synthetic data generation** significantly enhances detection capability for rare attack types, offering a strong foundation for future real-time and scalable intrusion detection frameworks.

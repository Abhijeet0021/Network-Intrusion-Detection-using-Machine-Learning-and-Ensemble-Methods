# <h1>Network-Intrusion-Detection-using-Machine-Learning-and-Ensemble-Methods</h1>
# Project Overview
This project presents a comparative study of traditional machine learning models and ensemble learning algorithms for Network Intrusion Detection Systems (NIDS) using the NSL-KDD dataset.
The main goal is to improve the detection of both common and rare cyberattacks by applying:
 1. Recursive Feature Elimination (RFE) for feature selection, and
 2. Conditional Generative Adversarial Networks (cGANs) for synthetic data generation.
# Objectives
1. Evaluate and compare traditional ML models (SVM, Decision Tree, Logistic Regression, Random Forest) with ensemble models (AdaBoost, Gradient Boosting, XGBoost).
2. Handle data imbalance for rare attacks (R2L, U2R) using synthetic data generation.
3. Improve model robustness and accuracy through feature selection and cross-validation.
4. Measure model performance using accuracy, precision, recall, F1-score, and AUC-ROC.
# Dataset
Dataset Used: NSL-KDD Dataset
<ul> 41 features per record</ul>
<ul><li>4 main attack categories:
 1.🧨 Denial of Service (DoS)
 2. 🔍 Probe
 3. 🐚 Remote to Local (R2L)
 4. 🔑 User to Root (U2R)</li></ul>
<li>Addressed class imbalance by augmenting minority classes using Conditional GANs.</li>

# ⚙️ Methodology
1. Data Preprocessing
Handled missing values and outliers
Encoded categorical features (protocol_type, service, flag)
Normalized numerical features
2. Feature Selection
Applied Recursive Feature Elimination (RFE) to remove irrelevant or redundant attributes.
3. Synthetic Data Generation
Used Conditional GANs (cGANs) to balance dataset by generating realistic attack samples.
4. Model Training
Implemented multiple models:
Traditional: Logistic Regression, Decision Tree, Random Forest, SVM
Ensemble: AdaBoost, Gradient Boosting, XGBoost
5. Evaluation Metrics
Accuracy
Precision
Recall
F1-score
AUC-ROC

# 🧠 Key Findings
1. XGBoost showed the highest robustness and recall for multiple attack types.
2. Feature selection (RFE) improved training efficiency and interpretability.
3. Synthetic data (cGAN) helped balance rare classes, improving F1-scores.
4. Cross-validation reduced overfitting and improved model generalization.

# 🚀 Future Enhancements
Real-time intrusion detection using streaming frameworks (e.g., Apache Kafka, Spark Streaming).
Advanced imbalance handling using SMOTE or ADASYN.
Integration of deep learning (CNN, LSTM, Transformer) with ensemble models.
Deployment on cloud-based environments (AWS / Azure) for real-time monitoring.
Implementation of Explainable AI (XAI) for model interpretability.

# 🧰 Tech Stack
Category	   Tools / Libraries
Language	     Python 3.x
Data Processing	 Pandas, NumPy
Machine Learning	 Scikit-learn
Deep Learning (cGAN)	  TensorFlow / Keras
Visualization	     Matplotlib, Seaborn
IDE	           Jupyter Notebook, VS Code

# 📈 Performance Visualization
 <li>Confusion Matrices</li>
  <li>ROC Curves</li>
<li>Feature Importance Graphs</li>

# 🏁 Conclusion
This project demonstrates that ensemble learning models, particularly XGBoost, provide superior accuracy, robustness, and recall for Network Intrusion Detection Systems. 
The combination of feature selection and synthetic data generation significantly enhances detection capability for rare attack types, offering a strong foundation for future real-time and scalable intrusion detection frameworks.

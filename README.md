# ECG Heartbeat Classification (Decision Tree vs Random Forest)

## Project Overview
This project focuses on classifying ECG (electrocardiogram) heartbeats into **normal (1)** and **abnormal (0)** classes using machine learning techniques.  

The goal is to explore the full data science pipeline, including:
- data preprocessing  
- feature engineering  
- dimensionality reduction (PCA)  
- model training and evaluation  

A key focus is comparing a **single model (Decision Tree)** with an **ensemble model (Random Forest)**.



## Dataset
- Curated ECG heartbeat dataset  
- Each row represents a single heartbeat signal  
- Time-series data stored in tabular format  
- Variable-length signals with missing values  

**Binary classification:**
- `1 → Normal`
- `0 → Abnormal`



## Methods

###  Preprocessing & Feature Engineering
- Extracted labels from variable-length signals  
- Removed missing values  
- Converted signals into fixed-length representations  
- Generated features:
  - Statistical (mean, std, min, max, range)  
  - Signal-based (energy, peak count)  



### Dimensionality Reduction (PCA)
- Reduced feature redundancy  
- First 3 components captured ~86% variance  
- Original features retained for modelling (better interpretability and suitability for tree-based models)



### Models Used
- Decision Tree  
- Random Forest  



## Results
- **Decision Tree F1-score:** 0.956  
- **Random Forest F1-score:** 0.964  



### Key Findings:
- Random Forest achieved better performance  
- Reduced false negatives (critical in ECG classification)  
- Demonstrated improved generalisation  



## Cross-Validation
- 10-fold cross-validation applied  
- Random Forest slightly outperformed Decision Tree  
- Difference **not statistically significant** (p-value = 0.23)  



## Key Insight
Ensemble methods like Random Forest reduce variance and produce more stable predictions, but improvements may not always be statistically significant.



## Tools & Technologies
- Python  
- scikit-learn  
- pandas  
- numpy  
- VS Code  



## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt

2. Run the project
python src/assignment.py


## Project Structure

ecg-heartbeat-classification/
│
├── data/
│   └── ecg_curated.csv
├── src/
│   └── assignment.py
├── README.md
├── requirements.txt
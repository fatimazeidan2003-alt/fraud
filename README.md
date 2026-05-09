# Cost-Sensitive Learning for Insurance Fraud Detection

**Team:** Leen ElSakka & Fatima Zeidan  
**Course:** Applied Machine Learning - Lebanese American University  
**Instructor:** Dr. Seifedine Kadry

## Project Overview
This project investigates cost-sensitive machine learning for insurance fraud 
detection on the Insurance Fraud Oracle dataset (15,420 records, ~6% fraud rate).
We compare 5 algorithms and demonstrate that cost-sensitive Logistic Regression 
reduces operational cost by 26.4% while achieving 86% fraud recall.

## Files
- `ml-fraud-final.ipynb` — Main notebook (full code, models, results)
- `app.py` — Streamlit web application for fraud detection
- `fraud_oracle.csv` — Dataset
- `requirements.txt` — Python dependencies
- `model.pkl`, `scaler.pkl`, `features.pkl`, `threshold.pkl` — Saved model artifacts

## Live Demo
🛡️ [Fraud Detection App](https://fraudgit-uckx2kjzatv6ax8bbczvam.streamlit.app/)


## Key Results
| Model | Fraud Recall | Cost |
|---|---|---|
| Logistic Regression (baseline) | 0.000 | 1,851 |
| Gradient Boosting (baseline) | 0.032 | 1,790 |
| Weighted LR (cost-sensitive) | 0.860 | 1,318 |
| Weighted RF (cost-sensitive) | 0.860 | 1,232 |


**please refer to the report for full analysis**

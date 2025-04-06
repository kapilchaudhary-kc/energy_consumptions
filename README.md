# ⚡ Energy Consumption Prediction Pipeline

This project builds and tracks a **Linear Regression model** to predict energy consumption based on temporal and weather features.

---

## 📁 Project Structure

```plaintext
info/ 
├── data/ 
│   ├── energy_consumption.csv 
│   └── processed_test_data.csv 
├── models/ 
│   └── lin_reg.pkl 
├── scripts/ 
│   ├── train_model.py 
│   └── run_assumption_checks.py 
├── .github/ 
│   └── workflows/ 
│   └── ml-pipeline.yml 
├── requirements.txt 
└── README.md

```

---

## 🧠 Features

- 📊 Linear Regression using Scikit-learn  
- 📁 Data preprocessing, train-test split, model training  
- 🧪 Assumption checks with saved test data  
- 📈 MLflow autologging & artifact tracking  
- 🛠️ CI/CD pipeline using **GitHub Actions**

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/kapilchaudhary-kc/energy_consumptions.git
cd energy-prediction-pipeline

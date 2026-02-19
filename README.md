# Causal Uplift Modeling for Telco Churn Reduction 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Executive Summary
This project implements a **Heterogeneous Treatment Effect (HTE)** framework to optimize customer retention strategies. By simulating a Randomized Controlled Trial (RCT) environment, I developed an **X-Learner** meta-algorithm from scratch to identify "Persuadable" customers—those who are most likely to be retained *only if* they receive a treatment (e.g., a discount).

Unlike traditional churn prediction (which targets high-risk users regardless of intervention effect), this Uplift Model focuses on **incremental ROI**, achieving a **$10k+ expected profit increase** per 7,000 users compared to random targeting.

## 📊 Key Results & Visualizations

### 1. Business Impact: Profit Curve (The "Money" Shot)
The model identifies the optimal targeting threshold. By targeting the top **59%** of users based on Uplift Score, we maximize expected profit to **~$10k** (per 7k users), avoiding waste on users who would stay anyway or leave regardless of the offer.
![Profit Curve](images/profit_curve.png)

### 2. Model Interpretability: SHAP Analysis (The "Why")
Using SHAP values to interpret the CATE (Conditional Average Treatment Effect), we discovered that **Month-to-month contract** users with high tenure are the most responsive segment. This confirms our hypothesis that long-term users on flexible contracts are the most "Persuadable."
![SHAP Summary](images/shap_summary.png)

### 3. Model Performance: Qini Curve (The "Proof")
The X-Learner (Pink line) significantly outperforms the T-Learner (Baseline) and Random selection. The Area Under Uplift Curve (AUUC) is **0.68**, indicating strong ranking ability.
![Qini Curve](images/qini_curve.png)

### 4. Model Reliability: Calibration Plot (The "Trust")
The model is well-calibrated. Users predicted to have high uplift (Decile 1) actually showed the highest true uplift in the validation set, proving the model's reliability for real-world deployment.
![Calibration Plot](images/calibration_plot.png)

### 5. Data Context: Churn by Contract (The "Context")
EDA reveals that Month-to-month users have a significantly higher baseline churn rate, making them the primary target pool for retention experiments.
![Churn EDA](images/churn_by_contract.png)

---

## 🛠️ Technical Implementation

### Core Components
* **Causal Inference Framework**: Implemented **X-Learner** and **T-Learner** meta-algorithms.
* **Base Learner**: Utilized **XGBoost Regressor** and **Classifier** for outcome prediction.
* **Evaluation Metrics**: AUUC (Area Under Uplift Curve), Qini Coefficient, and Decile Calibration.
* **Simulation**: Generated synthetic ground-truth uplift labels to validate model accuracy (Spearman Correlation: **0.87**).

### Code Structure
The `XLearner` class is implemented with a Scikit-Learn compatible API:
```python
class XLearner(BaseEstimator, RegressorMixin):
    def fit(self, X, y, t):
        # Stage 1: Base Outcome Models (Control vs Treatment)
        # Stage 2: Pseudo-Outcome Regression (Estimating CATE)
        ...

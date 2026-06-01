# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops an end-to-end credit risk scoring system for Bati Bank using transaction data from an eCommerce platform. The objective is to build a proxy-based credit scoring model capable of estimating customer risk probabilities and supporting Buy-Now-Pay-Later (BNPL) lending decisions.

## Project Structure

```text
credit-risk-model/
├── .github/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

## Credit Scoring Business Understanding

### 1. Basel II and Interpretability

### 2. Why a Proxy Target Variable is Necessary

### 3. Interpretable vs High Performance Models


# Credit Risk Probability Model for Alternative Data

## Project Overview

...

## Project Structure

...

## Credit Scoring Business Understanding

### 1. Basel II and Interpretability

### 2. Why a Proxy Target Variable is Necessary

### 3. Interpretable vs High Performance Models

### 1. Basel II and Interpretability
The Basel II Capital Accord emphasizes effective risk measurement, transparency, and regulatory compliance in financial institutions. Since credit decisions directly affect access to financing and influence a bank’s exposure to risk, models used for credit scoring must be interpretable, well-documented, and auditable.

An interpretable model allows risk managers, auditors, and regulators to understand how predictions are generated and verify that lending decisions are based on objective and measurable factors. Proper documentation also supports model validation, monitoring, and governance throughout the model lifecycle.

For Bati Bank, Basel II considerations influence the selection of modeling techniques, feature engineering processes, performance evaluation methods, and reporting standards. Even when advanced machine learning models are used, the rationale behind predictions should remain explainable and defensible to both internal stakeholders and external regulators.

### 2. Why a Proxy Target Variable is Necessary
The transaction dataset provided by the eCommerce platform does not contain a direct indicator of customer default. Since supervised machine learning models require a target variable, a proxy variable must be created to represent credit risk.

In this project, customer transaction behavior will be analyzed using Recency, Frequency, and Monetary (RFM) metrics. Customers exhibiting low engagement, infrequent transactions, and low monetary activity may be considered higher-risk segments. Clustering techniques will be used to identify these behavioral groups and create a binary target variable for model training.

While proxy variables enable the development of predictive models in the absence of actual default data, they also introduce business risks. The proxy may not perfectly represent true credit default behavior, potentially leading to misclassification of customers. As a result, model predictions should be interpreted as estimates of risk rather than definitive measures of creditworthiness.

### 3. Interpretable vs High Performance Models
Credit risk modeling often involves balancing model interpretability and predictive performance. Traditional models such as Logistic Regression, especially when combined with Weight of Evidence (WoE) transformation, are widely used because they are transparent, easy to explain, and align well with regulatory requirements.

On the other hand, advanced machine learning models such as Random Forests, Gradient Boosting, XGBoost, and LightGBM can capture complex patterns in data and often achieve higher predictive accuracy. However, these models are generally more difficult to interpret and may require additional explainability techniques to satisfy regulatory and business requirements.

In a regulated financial environment, model selection should consider not only predictive performance but also transparency, fairness, maintainability, and compliance. Therefore, this project will evaluate multiple models and select the most appropriate solution based on both business and technical considerations.
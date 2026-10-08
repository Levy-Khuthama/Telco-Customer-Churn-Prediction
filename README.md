# Telco-Customer-Churn-Prediction
Predicting which telecom customers are likely to leave (churn) using logistic regression and a random forest classifier, with a comparison of the two models beyond plain accuracy.

This project was completed as a machine learning capstone for the Artificial Intelligence online bootcamp (Stellenbosch University in partnership with HyperionDev).

What's inside
File	Description
telco_customer_churn.ipynb	Full notebook: data cleaning, visualisation, both models, and evaluation
Dataset

The Telco Customer Churn dataset: one row per customer, with account details (tenure, contract type, payment method), services signed up for (internet, phone, streaming, tech support) and charges, plus whether the customer churned. The CSV is not included in this repo; add Telco-Customer-Churn.csv next to the notebook to run it.

Approach

Preprocessing

Converted TotalCharges from text to numeric and removed the 11 rows with missing values, leaving 7,032 customers.
Dropped customerID (a unique identifier with no predictive value).
Encoded the Churn target as 1/0.
One-hot encoded the 15 categorical columns with drop_first=True to avoid the dummy variable trap, giving 30 features.

Exploration

Correlation of every feature with churn, a tenure histogram, monthly vs. total charges coloured by churn, and a tenure box plot for churned vs. retained customers.
Churned customers tend to have shorter tenure, and month-to-month contracts correlate more strongly with churn.

Modelling

75/25 train/test split (5,274 / 1,758 rows), with Min-Max scaling fitted on the training data only to avoid data leakage.
Logistic regression (max_iter=1000).
Random forest: 2,000 trees, max_features="sqrt", max_leaf_nodes=50, bootstrapping with out-of-bag scoring.
Results (test set)
Model	Accuracy	Precision	Recall
Logistic Regression	0.791	0.619	0.517
Random Forest	0.794	0.648	0.454

The random forest's out-of-bag score was 0.807 (OOB error 0.193), which is close to its test accuracy and suggests it generalises reasonably well.

Key takeaways
Accuracy alone is misleading here. Non-churners outnumber churners, so both models look decent on accuracy while still missing a large share of customers who actually leave.
Both models have higher precision than recall. Of the customers flagged as likely to churn, most really do, but roughly half or more of the true churners are missed.
Logistic regression finds more churners (higher recall), while the random forest makes fewer false alarms (higher precision). If missing a churner costs more than a wasted retention offer, the higher recall favours logistic regression, which is also easier to explain to non-technical stakeholders.
Ideas for improvement
Handle the class imbalance (class weights, resampling, or adjusting the decision threshold) to raise recall.
Add feature importance and logistic regression coefficients to show what drives churn.
Tune hyperparameters with cross-validation instead of a single train/test split.
Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

How to run
Clone the repo.
Install dependencies:
bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
Place Telco-Customer-Churn.csv in the same folder as the notebook.
Run all cells in telco_customer_churn.ipynb.

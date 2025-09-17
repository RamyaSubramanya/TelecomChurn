Goal: "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs."

The Orange Telecom's Churn Dataset, which consists of cleaned customer activity data (features), along with a churn label specifying whether a customer canceled the subscription, will be used to develop predictive models. Two datasets are made available here: The churn-80 and churn-20 datasets can be downloaded.

The two sets are from the same batch, but have been split by an 80/20 ratio. As more data is often desirable for developing ML models, let's use the larger set (that is, churn-80) for training and cross-validation purposes, and the smaller set (that is, churn-20) for final testing and model performance evaluation.

Methodology: 
1. Data pre-processing: clean and format datasets, handle missing values, split into train-test
2. Modelling: LogisticRegression, RandomForest, GradientBoosting, Neural network models
3. To check the uncertainity/risk analysis for future: MonteCarlo simulation 
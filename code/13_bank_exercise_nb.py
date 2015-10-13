# # Exercise with bank marketing data

# ## Introduction
# 
# - Data from the UCI Machine Learning Repository: [data](https://github.com/justmarkham/DAT8/blob/master/data/bank-additional.csv), [data dictionary](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
# - **Goal:** Predict whether a customer will purchase a bank product marketed over the phone
# - `bank-additional.csv` is already in our repo, so there is no need to download the data from the UCI website

# ## Step 1: Read the data into Pandas

# ## Step 2: Prepare at least three features
# 
# - Include both numeric and categorical features
# - Choose features that you think might be related to the response (based on intuition or exploration)
# - Think about how to handle missing values (encoded as "unknown")

# ## Step 3: Model building
# 
# - Use cross-validation to evaluate the AUC of a logistic regression model with your chosen features
# - Try to increase the AUC by selecting different sets of features


import pandas as pd

bank = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bank-additional.csv", sep = ';')
bank.columns


features = ['marital', 'education', 'age']
x = bank[features]
y = bank.y

from sklearn.cross_validation import cross_val_score

marital_dummies= pd.get_dummies(x.marital, prefix = 'marital')
marital_dummies.drop(marital_dummies.columns[3], inplace = True, axis = 1)


x.drop(x.columns[0], inplace = True, axis = 1)
x = pd.concat([x, marital_dummies], axis = 1)

x.educationmap()





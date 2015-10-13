# # Logistic regression exercise with Titanic data

# ## Introduction
# 
# - Data from Kaggle's Titanic competition: [data](https://github.com/justmarkham/DAT8/blob/master/data/titanic.csv), [data dictionary](https://www.kaggle.com/c/titanic/data)
# - **Goal**: Predict survival based on passenger characteristics
# - `titanic.csv` is already in our repo, so there is no need to download the data from the Kaggle website

# ## Step 1: Read the data into Pandas

# ## Step 2: Create X and y
# 
# Define **Pclass** and **Parch** as the features, and **Survived** as the response.

# ## Step 3: Split the data into training and testing sets

# ## Step 4: Fit a logistic regression model and examine the coefficients
# 
# Confirm that the coefficients make intuitive sense.

# ## Step 5: Make predictions on the testing set and calculate the accuracy

# ## Step 6: Compare your testing accuracy to the null accuracy

import pandas as pd
titanic = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/titanic.csv')

feature_cols = ['Pclass', 'Parch']
y = titanic.Survived
x = titanic[feature_cols]

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
linreg = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=123)
logreg = LogisticRegression(C=1e9)

logreg.fit(X_train, y_train)
print logreg.coef_
zip(feature_cols, logreg.coef_)

S_pred = logreg.predict(x)
titanic['Survival_Pred'] = S_pred

S_pred_class  = logreg.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_test, S_pred_class)

import matplotlib.pyplot as mtp
import seaborn as sns
mtp.scatter(titanic.Pclass, titanic.Survived)
mtp.xlabel('Class')
mtp.ylabel('Died')












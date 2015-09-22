# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 19:38:41 2015

@author: William
"""

# read the NBA data into a DataFrame
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT4-students/master/kerry/Final/NBA_players_2015.csv'
nba = pd.read_csv(url, index_col=0)

# map positions to numbers
nba['pos_num'] = nba.pos.map({'C':0, 'F':1, 'G':2})

# create feature matrix (X)
feature_cols = ['ast', 'stl', 'blk', 'tov', 'pf']
X = nba[feature_cols]

# create response vector (y)
y = nba.pos_num

# import the class
from sklearn.neighbors import KNeighborsClassifier

# instantiate the model
knn = KNeighborsClassifier(n_neighbors=50)

# train the model on the entire dataset
knn.fit(X, y)

# predict the response values for the observations in X ("test the model")
knn.predict(X)

# store the predicted response values
y_pred_class = knn.predict(X)

# compute classification accuracy
from sklearn import metrics
print metrics.accuracy_score(y, y_pred_class)

#KNN (K=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred_class = knn.predict(X)
print metrics.accuracy_score(y, y_pred_class) # will print an accuracy of 100% but only 
#because it is most similar to itself.  this is training accuracy, which is not an indicator of out of sample performance

# Understanding Unpacking
def min_max(nums):
    smallest = min(nums)
    largest = max(nums)
    return [smallest, largest]

min_and_max = min_max([1, 2, 3])
print min_and_max
print type(min_and_max)

the_min, the_max = min_max([1, 2, 3])
print the_min
print type(the_min)
print the_max
print type(the_max)

#Understanding the train_test_split function (split the sample into two sets and compare the model results generated on one subset to the actual data from the other subset)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# before splitting
print X.shape

# after splitting
print X_train.shape
print X_test.shape

# before splitting
print y.shape

# after splitting
print y_train.shape
print y_test.shape

# WITHOUT a random_state parameter
X_train, X_test, y_train, y_test = train_test_split(X, y)

# print the first element of each object
print X_train.head(1)
print X_test.head(1)
print y_train.head(1)
print y_test.head(1)

# WITH a random_state parameter
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=95)

# print the first element of each object
print X_train.head(1)
print X_test.head(1)
print y_train.head(1)
print y_test.head(1)

# Use stratified sampling to overcome possible oversampling of any one segment 
#(ie. too many Centers and no Guards or Forwards of the dataset.

# STEP 1: split X and y into training and testing sets (using random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

# STEP 2: train the model on the training set (using K=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


# STEP 3: test the model on the testing set, and check the accuracy
y_pred_class = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class) #check accuracy by comparing the y_test data to trh y_pred_class, which is the result set of the model trained on the original data

import matplotlib as plt
import pandas as pd
prob_list= []
for x in range(1,35):
    knn = KNeighborsClassifier(n_neighbors=x)
    knn.fit(X_train, y_train)
    y_pred_class = knn.predict(X_test)
    prob_list.append((x,metrics.accuracy_score(y_test, y_pred_class)))

probs = pd.DataFrame(prob_list, columns = ['K', 'probs'])
pr_plot = probs.plot(kind = 'line', x = 'K', y = 'probs')
pr_plot.show()
# examine the class distribution
y_test.value_counts()

# examine the class distribution
y_test.value_counts()

# calculate TRAINING ERROR and TESTING ERROR for K=1 through 100

k_range = range(1, 101)
training_error = []
testing_error = []

for k in k_range:

    # instantiate the model with the current K value
    knn = KNeighborsClassifier(n_neighbors=k)

    # calculate training error
    knn.fit(X, y)
    y_pred_class = knn.predict(X)
    training_accuracy = metrics.accuracy_score(y, y_pred_class)
    training_error.append(1 - training_accuracy)
    
    # calculate testing error
    knn.fit(X_train, y_train)
    y_pred_class = knn.predict(X_test)
    testing_accuracy = metrics.accuracy_score(y_test, y_pred_class)
    testing_error.append(1 - testing_accuracy)
    
    
    
# allow plots to appear in the notebook
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# create a DataFrame of K, training error, and testing error
column_dict = {'K': k_range, 'training error':training_error, 'testing error':testing_error}
df = pd.DataFrame(column_dict).set_index('K').sort_index(ascending=False)
df.head()



# plot the relationship between K (HIGH TO LOW) and TESTING ERROR
df.plot(y='testing error')
plt.xlabel('Value of K for KNN')
plt.ylabel('Error (lower is better)')


# find the minimum testing error and the associated K value
df.sort('testing error').head()

# alternative method
min(zip(testing_error, k_range))


# plot the relationship between K (HIGH TO LOW) and both TRAINING ERROR and TESTING ERROR
df.plot()
plt.xlabel('Value of K for KNN')
plt.ylabel('Error (lower is better)')

# MAKING PREDICTIONS ON OUT OF SAMPLE DATA

# instantiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=14)

# re-train the model with X and y (not X_train and y_train) - why?
knn.fit(X, y)

# make a prediction for an out-of-sample observation
knn.predict([1, 1, 0, 1, 2])


# try different values for random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=98)
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)












# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:32:30 2015

@author: William
"""

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=col_names)

%matplotlib inline
import matplotlib.pyplot as plt

# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14

# create a custom colormap
from matplotlib.colors import ListedColormap
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# map each iris species to a number
iris['species_num'] = iris.species.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# create a scatter plot of PETAL LENGTH versus PETAL WIDTH and color by SPECIES
iris.plot(kind='scatter', x='petal_length', y='petal_width', c='species_num', colormap=cmap_bold)

# create a scatter plot of SEPAL LENGTH versus SEPAL WIDTH and color by SPECIES
iris.plot(kind='scatter', x='sepal_length', y='sepal_width', c='species_num', colormap=cmap_bold)


# store feature matrix in "X"
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris[feature_cols]

# alternative ways to create "X"
X = iris.drop(['species', 'species_num'], axis=1)
X = iris.loc[:, 'sepal_length':'petal_width']
X = iris.iloc[:, 0:4]


# store response vector in "y"
#this must be a series not a data frame
y = iris.species_num


# make an instance of a KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors=1)
type(knn)

# make an instance of a list object
mylist = [1, 2, 3]
type(mylist)

# Created an object that "knows" how to do K-nearest neighbors classification, and is just waiting for data
#Name of the object does not matter
#Can specify tuning parameters (aka "hyperparameters") during this step
#All parameters not specified are set to their defaults

print knn

knn.fit(X, y)

knn.predict([3, 5, 4, 2])

X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn.predict(X_new)

#  Refining the model----------------------

# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X, y)

# predict the response for new observations
knn.predict(X_new)

# show probability associated with each prediction
knn.predict_proba(X_new)

knn.predict











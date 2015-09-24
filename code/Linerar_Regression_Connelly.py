# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:23:06 2015

@author: William
"""
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv'
yelp_df = pd.read_csv(url)


yelp_df.plot(kind = 'scatter', x = 'cool', y = 'stars')
yelp_df.plot(kind = 'scatter', x = 'useful', y = 'stars')
yelp_df.plot(kind = 'scatter', x = 'funny', y = 'stars')

yelp_df.columns
yelp_reg = LinearRegression()
features = ['funny', 'cool', 'useful']

y = yelp_df.stars
x = yelp_df[features]

yelp_reg.fit(x, y)
zip(features, yelp_reg.coef_)


from sklearn.cross_validation import train_test_split


X_train, X_test, y_train, y_test = train_test_split (x,y, random_state = 99)
yelp_reg_tts = LinearRegression()

yelp_reg_tts.fit(X_train, y_train)
yelp_star_pred = yelp_reg_tts.predict(X_test)

from sklearn import metrics

root_mean_square_error = np.sqrt(metrics.mean_squared_error(y_test, yelp_star_pred))
root_mean_square_error # = 1.179679274035353


# Removing features
yelp_reg_reduced = LinearRegression()
feature_cols = ['useful', 'funny']
x2 = yelp_df[feature_cols]


yelp_reg_reduced.fit(x2,y)
zip(feature_cols, yelp_reg_reduced.coef_)

X_train, X_test, y_train, y_test = train_test_split (x2,y2, random_state = 99)
yelp_reg_reduced.fit(X_train, y_train)

star_pred = yelp_reg_reduced.predict(X_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, star_pred))

rmse #= 1.1926202449959369 with ['funny', 'cool']
rmse #= 1.1926202449959369 with ['useful', 'funny']
# the best result was with all three features

















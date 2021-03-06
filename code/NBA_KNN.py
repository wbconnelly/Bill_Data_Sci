# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:34:58 2015

@author: William
"""
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

nba = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT4-students/master/kerry/Final/NBA_players_2015.csv")

feature_cols = ['ast', 'stl', 'blk', 'tov', 'pf']
x = nba[feature_cols]

nba['pos_num'] = nba.pos.map({'F':0, 'G':1, 'C':2})
y = nba['pos_num']


num_iter = 100
knn = KNeighborsClassifier(n_neighbors= num_iter)

knn.fit(x,y)

knn.predict([1,1,0,1,2])


nba.plot(kind = 'scatter', x = 'tov', y = 'ast', c = 'pos_num', colormap = cmap_bold)
plt.xlabel('tov')
plt.ylabel('ast')


iris.plot(kind='scatter', x='petal_length', y='petal_width', c='species_num', colormap=cmap_bold)
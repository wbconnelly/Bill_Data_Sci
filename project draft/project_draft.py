# -*- coding: utf-8 -*-
"""
Created on Wed Oct 07 21:05:16 2015

@author: William
"""

from bs4 import BeautifulSoup
import urllib2 as ul
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

# Pull down yearly financial reporting data from the Kimono API

data = []

for i in range(0, len(sym_list)-1):
    results = urllib.urlopen(
    "http://sec.kimonolabs.com/companies/"+ sym_list[i]+"/forms/10-K/ANN/2015,2014,2013,2012,2011/Q4?apikey=yaimV7I2x4hOIMnVTpdOOmz4etlnIxdn")
    results = pd.read_json(results.read())
    data.append(results)

dataset = pd.concat(data)
#write data to csv
dataset.to_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/fin_data.csv")

# Web scraper to pull down the industry classification of all companies in the S&P 500
sym_file = open("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/symbol_list.txt")
sym_list = sym_file.read()
sym_list = sym_list.split(',/n')

def print_page(Base):
    
    target = Base
    page = ul.urlopen(target)
    bs = BeautifulSoup(page.read())
    #start = '<a id=sector href=[^.]*>(.+?)</a>' 
    #pattern = re.compile(start)
    #industry = re.findall(pattern, bs)
    try:
        industry =  bs.find(name = 'a', attrs = {'id':'sector'}).text
        return industry
    except: 
        return "None Found"
        
industry_list= []

for symb in sym_list:
    Base_URL = 'https://www.google.com/finance?q='+symb +'&ei=q8gVVqn1GoKjmAHX7oWwCg'
 
    #print (symb, print_page(Base_URL))
    industry_list.append([symb, print_page(Base_URL)])

pd.concat(industry_list)
company_sectors = pd.DataFrame(industry_list, columns = ['Symbol', 'Sector'])

#write the data to a csv
company_sectors.to_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/company_sectors.csv")

#---------------------------------------------------------# End Web Scraper


fin_data = pd.read_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/fin_data.csv")
company_sectors = pd.read_csv("C:\Users\William\Desktop\Git_Repos\Bill_Data_Sci\project\company_sectors.csv")
company_sectors.rename(columns={'Symbol':'company_symbol'}, inplace = True)

# Company data is the unaveraged dataset
company_data = pd.merge(fin_data, company_sectors, on = 'company_symbol')

#company_data.to_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/final_company_data.csv")


company_data.company_symbol.value_counts()

#get average values for each company across the years in the sample
company_data.groupby('company_symbol').mean().shape
company_avg = company_data.groupby('company_symbol').mean()

company_avg['company_symbol'] = company_avg.index

# reattch the symbols since they were dropped when getting averages
company_avg = pd.merge(company_avg, company_sectors, on = 'company_symbol')

# create all dummy columns for the sectors
y_vals = pd.get_dummies(company_avg.Sector)
y_vals['ind_val'] = y_vals.index
company_avg['ind_val'] = company_avg.index

# get final dataset by reattaching the dummy values
company_avg_sector = pd.merge(company_avg, y_vals, on = 'ind_val')


# find the number of companies in each sector
company_avg.Sector.value_counts()

# find average number of missing values
company_avg.isnull().sum().mean()

# find columns with few missing values for each sector grouping
null_list = {}

# add each to  data frame to se which columns for each sector are good candidates as predictors
for sector in company_data.Sector.unique():
    null_df = pd.DataFrame(company_data[company_data.Sector == sector].isnull().sum())
    null_list[sector] = null_df

company_avg_sector.columns
null_list['Financials']
# use logistic regression to try and predict the industry classifiaction based on well populated columns

    #look at the financials sector list and choose some suitable columns that have few missing values
null_list['Financials']

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
feature_cols = ['retainedearnings', 'totalassets', 'totalrevenue']
x = company_avg_sector[feature_cols]
#fill NaNs with the average of each column
x = x.fillna(x.mean())
y = company_avg_sector.Financials
logreg.fit(x, y)
#make predictions
company_avg_sector['Financials_predicted'] = logreg.predict(x)
#predict probabilities
company_avg_sector['Financials_pred_prob'] = logreg.predict_proba(x)[:, 1]

#plot the probabilities
probs_sorted = company_avg_sector.Financials_pred_prob.copy()
probs_sorted.sort()
plt.plot(probs_sorted)

# print the accuracy
print metrics.accuracy_score(y, company_avg_sector.Financials_predicted)
    # 0.933884297521


# Create the confusion matrix

print metrics.confusion_matrix(y, company_avg_sector.Financials_predicted)


# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y, company_avg_sector.Financials_predicted)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]

print 'True Positives:', TP
print 'True Negatives:', TN
print 'False Positives:', FP
print 'False Negatives:', FN

#True Positives: 56
#True Negatives: 396
#False Positives: 3
#False Negatives: 29

# I need a more systematic way of selecting features and a

















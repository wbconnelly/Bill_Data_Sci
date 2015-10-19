from bs4 import BeautifulSoup
import urllib2 as ul
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

fin_data = pd.read_csv("https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/fin_data.csv")
company_sectors = pd.read_csv("https://raw.githubusercontent.com/wbconnelly/Bill_Data_Sci/master/project%20draft/company_sectors.csv")
company_sectors.rename(columns={'Symbol':'company_symbol'}, inplace = True)

# Company data is the unaveraged dataset
company_data = pd.merge(fin_data, company_sectors, on = 'company_symbol')

#company_data.to_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/final_company_data.csv")


#get average values for each company across the years in the sample
company_data.groupby('company_symbol').mean().shape
company_avg = company_data.groupby('company_symbol').mean()

company_avg['company_symbol'] = company_avg.index

# reattch the symbols since they were dropped when getting averages
company_avg = pd.merge(company_avg, company_sectors, on = 'company_symbol')

# find the number of companies in each sector
company_avg.Sector.value_counts()

# find average number of missing values
null_sum = pd.DataFrame(company_avg.isnull().sum()).reset_index()
null_sum.rename(columns = {0:'null_count', 'index':'col_title'}, inplace = True)
company_avg.isnull().sum().mean()

# find columns with few missing values for each sector grouping
null_list = {}

# add each to  data frame to se which columns for each sector are good candidates as predictors
for sector in company_data.Sector.unique():
    null_df = pd.DataFrame(company_data[company_data.Sector == sector].isnull().sum()).reset_index()
    null_df.rename(columns = {0:'null_count', 'index':'col_title'}, inplace = True)
    null_list[sector] = null_df



# use logistic regression to try and predict the industry classifiaction based on well populated columns

    #look at the financials sector list and choose some suitable columns that have few missing values
null_list['Technology']

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)


feature_cols = list(company_avg.loc[:, company_avg.dtypes == np.float64].columns)

col_list_delete = ['extraordinaryitems',
'deferredcharges',
'accountingchange',
'amended',
'audited',
'Unnamed: 0_y',
'Unnamed: 0_x',
 'year',
 'quarter',
 'restated',
'company_cik', 
'usdconversionrate',
'periodlength',
'original']

for col in col_list_delete:
    try:    
        feature_cols.remove(col)    
    except:
        pass
feature_cols.append('Sector')


#fill NaNs with the average of each column
sector_list = list(company_avg_sector.Sector.unique())

# impute all NaN values with the mean from it's respective sector
feature_dfs = []
for sector in sector_list:
    x = company_avg[company_avg.Sector == sector][feature_cols]
    x = x.fillna(x.mean())
    feature_dfs.append(x)
    
x = pd.concat(feature_dfs)
x.reset_index(inplace = True)
x = x.fillna(x.mean())
x['researchdevelopmentexpense'].fillna(x['researchdevelopmentexpense'].mean(), inplace = True)
#company_avg.loc[:, company_avg.dtypes == np.float64]

# Get dummy values for the Sectors
y_vals = pd.get_dummies(x.Sector)


# get final dataset by reattaching the dummy values
#x = pd.merge(x, y_vals, on = 'ind_val')
x.drop('Sector', axis = 1, inplace= True)
feature_cols.remove('Sector')

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics

# testing accuracy of Multinomial Naive Bayes
mnb = MultinomialNB()
#mnb.fit(X_, y)

# testing accuracy of Gaussian Naive Bayes
#gnb = GaussianNB()
#gnb.fit(x, y)


for sect in sector_list:
    x = x[feature_cols]
    y = y_vals[sect]
    mnb.fit(x**2, y)
    x['predicted'] = mnb.predict(x)
    print sector, '---',metrics.accuracy_score(y, x.predicted)
    confusion = metrics.confusion_matrix(y, x.predicted)
    TP = confusion[1][1]
    TN = confusion[0][0]
    FP = confusion[0][1]
    FN = confusion[1][0]
    print 'True Positives:', TP
    print 'True Negatives:', TN
    print 'False Positives:', FP
    print 'False Negatives:', FN




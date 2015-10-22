from bs4 import BeautifulSoup
import urllib2 as ul
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
plt.style.use('fivethirtyeight')

# Pull down yearly financial reporting data from the Kimono API
sym_file = open("https://github.com/wbconnelly/Bill_Data_Sci/blob/master/project%20draft/symbol_list.txt")
sym_list = sym_file.read()
sym_list = sym_list.split(',')
data = []

for i in range(0, len(sym_list)-1):
    results = urllib.urlopen(
    "http://sec.kimonolabs.com/companies/"+ sym_list[i]+"/forms/10-K/ANN/2015,2014,2013,2012,2011/Q4?apikey=yaimV7I2x4hOIMnVTpdOOmz4etlnIxdn")
    results = pd.read_json(results.read())
    data.append(results)

dataset = pd.concat(data)
#write data to csv
dataset.to_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/fin_data.csv")
#-----------------------------End API call

# Web scraper to pull down the industry classification of all companies in the S&P 500


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

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)


feature_cols = list(company_avg.loc[:, company_avg.dtypes == np.float64].columns)

col_list_delete = ['extraordinaryitems',
'deferredcharges',
'accountingchange',
'amended',
'audited',
'Unnamed: 0_y', 'preliminary',
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
sector_list = list(company_avg.Sector.unique())

# impute all NaN values with the mean from it's respective sector
feature_dfs = []
for sector in sector_list:  
    x = company_avg[company_avg.Sector == sector][feature_cols]
    industry = pd.DataFrame(x.Sector, columns = ['Sector'])
    x.drop('Sector', axis = 1, inplace= True)
    x = x.fillna(x.mean())
    #x = (x - x.mean())/ np.std(x)
    x = x.join(industry)
    feature_dfs.append(x)
    
x = pd.concat(feature_dfs)
x.reset_index(inplace = True)
#x = x.fillna(x.mean())
x['researchdevelopmentexpense'].fillna(x['researchdevelopmentexpense'].mean(), inplace = True)

#company_avg.loc[:, company_avg.dtypes == np.float64]

# get final dataset by reattaching the dummy values
#x = pd.merge(x, y_vals, on = 'ind_val')


#x.drop('Sector', axis = 1, inplace = True)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)

"""x = x[feature_cols]
y = y_vals.Technology
logreg.fit(x, y)"""

"""#make predictions
company_avg_sector['predicted'] = logreg.predict(x)
#predict probabilities
company_avg_sector['pred_prob'] = logreg.predict_proba(x)[:, 1]
#plot the probabilities
probs_sorted = company_avg_sector.Financials_pred_prob.copy()
probs_sorted.sort()
plt.plot(probs_sorted)
# print the accuracy
print metrics.accuracy_score(y, company_avg_sector.Financials_predicted)
    # 0.933884297521"""
#np.isfinite(x).sum()
# Create the confusion matrix

var_list = {}
for sector in sector_list:
    var_tbl = pd.DataFrame(x[feature_cols][x.Sector == sector].var().reset_index())
    var_tbl.rename(columns = {0:'variance', 'index':'col_title'}, inplace = True)
    var_tbl.sort('variance', ascending = False, inplace = True)
    var_tbl['total_var']= var_tbl['variance'].cumsum(skipna = True)/var_tbl.variance.sum()
    var_list[sector] = var_tbl

# Get dummy values for the Sectors
y_vals = pd.get_dummies(x.Sector)

 
x.drop('Sector', axis = 1, inplace= True)
feature_cols.remove('Sector')

# use only the features that account for 99% of the variance
def frange(init, end, step):
    steps = []    
    while init < end:
        steps.append(init)
        init += step
    return steps


false_pred_dict = {}
num_feat_dict = {}
var_limit_dict = {}

for sector in sector_list:
    false_pred = []
    num_feat = []
    var_limit = []
    for var in frange(0.5,1,.01):
        x = x[feature_cols]
        x2 = x[var_list[sector]['col_title'][var_list[sector].total_var <= var]]
        y = y_vals[sector]
        logreg.fit(x2, y)
        x2['predicted'] = logreg.predict(x2)
        accuracy = metrics.accuracy_score(y, x2.predicted)
        confusion = metrics.confusion_matrix(y, x2.predicted)
        TP = confusion[1][1]
        TN = confusion[0][0]
        FP = confusion[0][1]
        FN = confusion[1][0]
        total_false = FP + FN
        false_pred.append(total_false)
        var_limit.append(var)
        num_feat.append(len(x2.columns))
        
        false_pred_dict[sector] = false_pred
        num_feat_dict[sector] = num_feat
        var_limit_dict[sector] = var_limit
       # print 'True Positives:', TP
        #print 'True Negatives:', TN
        #print 'False Positives:', FP
        #print 'False Negatives:', FN


'Cyclical Consumer Goods & Services',
 'Telecommunications Services',
 'Industrials',
 'Energy',
 'None Found',
 'Utilities',
 'Financials',
 'Healthcare',
 'Basic Materials',
 'Technology',
 'Non-Cyclical Consumer Goods & Services'

sector = 'Energy'
plt.scatter(x = var_limit_dict[sector], y = false_pred_dict[sector])

#explore the variance





#-------------------------#
for sector in sector_list:
    var_tbl = pd.DataFrame(company_avg[feature_cols][company_avg.Sector == sector].var().reset_index())
    var_tbl.rename(columns = {0:'variance', 'index':'col_title'}, inplace = True)
    var_tbl.sort('variance', ascending = False, inplace = True)
    var_tbl['total_var']= var_tbl['variance'].cumsum(skipna = True)/var_tbl.variance.sum()
    var_list[sector] = var_tbl

var_list['Utilities'].plot(kind = 'bar', x = 'col_title', y = 'variance')
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

# take only first three column titles from var_list['Utilities']
list(var_list['Utilities']['col_title'][var_list['Utilities'].total_var <.99])


sample = var_list['Utilities']










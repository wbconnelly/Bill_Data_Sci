# -*- coding: utf-8 -*-
"""
Created on Wed Oct 07 21:05:16 2015

@author: William
"""

from bs4 import BeautifulSoup
import urllib2 as ul
import re
import pandas as pd

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

company_sectors.to_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/company_sectors.csv")
company_sectors.rename(columns={'Symbol':'company_symbol'}, inplace = True)

fin_data = pd.read_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/fin_data.csv")

company_data = pd.merge(fin_data, company_sectors, on = 'company_symbol')

company_data.to_csv("C:/Users/William/Desktop/Git_Repos/Bill_Data_Sci/project/final_company_data.csv")


company_data.company_symbol.value_counts()

company_data.groupby('company_symbol').mean().shape

company_avg = company_data.groupby('company_symbol').mean()
company_avg['company_symbol'] = company_avg.index
company_avg = pd.merge(company_avg, company_sectors, on = 'company_symbol')

# create all dummy columns for the sectors
features = pd.get_dummies(company_avg.Sector)
features['ind_val'] = features.index
company_avg['ind_val'] = company_avg.index
company_avg = pd.merge(company_avg, features, on = 'ind_val')













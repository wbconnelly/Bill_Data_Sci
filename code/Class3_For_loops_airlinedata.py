# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:51:03 2015

@author: William
"""

#open("C:/Users/William/Desktop/Git_Repos/DAT8/data/airlines.csv", mode = "rU")

import csv
with open("airlines.csv", mode = "rU") as f:
    file_nested_list = [row for row in csv.reader(f)]
    
header =file_nested_list[0]
data = file_nested_list[1:]

def star(row):
    if row[0][-1] == '*':
        return 1
    else:
        return 0  

for row in data:
    print star(row)


avg_list = []
x= 0
for row in data:
    x= [row[0].strip('*'), star(row), (float(row[2]) + float(row[5]))/30]
    avg_list.append(x)

print avg_list



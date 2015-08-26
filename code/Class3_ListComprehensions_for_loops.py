# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:15:26 2015

@author: William
"""

movies = ['tt0111161', 'tt1856010', 'tt0096694', 'tt0088763', 'tt1375666']
numbers= []

for m in movies:
    numbers.append(m[2:])
print numbers
    
"""using a list comprehension"""

movies = ['tt0111161', 'tt1856010', 'tt0096694', 'tt0088763', 'tt1375666']
numbers= []

numbers = [m[2:] for m in movies]
numbers

print type(numbers[1])

"""coerce and sum all the strings"""

movies = ['tt0111161', 'tt1856010', 'tt0096694', 'tt0088763', 'tt1375666']
umbers= []

sum(int(i) for i in numbers)

s = 0
for i in numbers:
    s = s + int(i)
print s
    












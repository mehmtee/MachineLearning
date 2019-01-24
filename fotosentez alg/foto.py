#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:14:54 2019

@author: root1
"""


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("foto.csv")
y = df.hiz.values.reshape(-1,1)
x = df.sicaklik.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("Sicaklik")
plt.xlabel("Fotosentez Hizi")




from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)
#%%

y_head = lr.predict(x)
plt.plot(x,y_head,color="red",label="linear")
plt.show()

#%%
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree=10)

x_polynomial = polynomial_regression.fit_transform(x)
linear_regression2=LinearRegression()
linear_regression2.fit(x_polynomial,y)
y_head2=linear_regression2.predict(x_polynomial)
plt.plot(x,y_head2,color="black",label="Poly")
plt.legend()
plt.show()

#%%


for i in range(0,100):
    
    i = i+1
    y_eksen = (linear_regression2.predict(polynomial_regression.fit_transform(i)))
    print("""
       x Değeri : {}
       y Değeri : {}
    """.format(i,y_eksen))
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 01 17:25:50 2018

@author: Janco
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

path = os.path.join('.','Wrangling Code','CAD_and_Angina_cases_2013_to_2014.csv')
path2 = os.path.join('.','Wrangling Code','CAD_and_Angina_cases_2011_to_2012.csv')

df=pd.read_csv(path)
df2=pd.read_csv(path2)

#Delete all the non relevant columns
df=df[['SEQN', 'Angina', 'LBDLDLSI', 'RIDAGEYR']]
df=df.dropna()

df2=df2[['SEQN', 'Angina', 'LBDLDLSI', 'RIDAGEYR']]
df2=df2.dropna()

#Merge the dataframes for 2011-2012 and 2013-2014
df = df.merge(df2, left_on=['SEQN', 'Angina', 'LBDLDLSI', 'RIDAGEYR'], right_on=['SEQN', 'Angina', 'LBDLDLSI', 'RIDAGEYR'], how='outer')
df = df.reset_index()

df['Angint']=np.nan #Adding Anginaint column to dataframe

#Using One-Hot Encoding to convert class Yes/No to 1.0/0.0
for i in range(len(df)):
    if df.ix[i,'Angina']=="Yes":
        df.ix[i,'Angint']=1.0
    else:
        df.ix[i,'Angint']=0.0

#Binning data so that the average Angint can be obtained which will indicate the risk of developing CHD symptoms
age_bin_size=2.5
LDL_bin_size=0.3

Age_list=np.arange(0, 85, age_bin_size).tolist()
Age_list=[float("{:.2f}".format(x)) for x in Age_list]
Age_bin_names=[x for x in Age_list[0:((len(Age_list)-1))]]

for i in range(len(Age_bin_names)):
    Age_bin_names[i]=Age_bin_names[i]+(age_bin_size/2.0)

Age_bin_names= [float("{:.2f}".format(x)) for x in Age_bin_names]
df['bin_Age'] = pd.cut(df['RIDAGEYR'], Age_list, labels=Age_bin_names)


LDL_list=np.arange(0, 10, LDL_bin_size).tolist()
LDL_list= [float("{:.2f}".format(x)) for x in LDL_list]
LDL_bin_names=[x for x in LDL_list[0:((len(LDL_list)-1))]]

for i in range(len(LDL_bin_names)):
    LDL_bin_names[i]=LDL_bin_names[i]+(LDL_bin_size/2.0)

LDL_bin_names= [float("{:.2f}".format(x)) for x in LDL_bin_names]
df['bin_LDL'] = pd.cut(df['LBDLDLSI'], LDL_list, labels=LDL_bin_names)
     
df = df.groupby(['bin_LDL', 'bin_Age']).mean().dropna().reset_index()

#Plotting the data on a 3 dimensional axis
x_vals = df['bin_LDL']
z_vals = df['Angint']
y_vals = df['bin_Age']

fig = plot.figure(1)
axis = Axes3D(fig, axisbg='g')

axis.scatter(x_vals, y_vals, z_vals, c=z_vals, marker='o', cmap='Reds')
axis.set_xlabel("LDL-C (mmol/L)")
axis.set_ylabel("Age (Years)")
axis.set_zlabel("Risk")
plot.show()

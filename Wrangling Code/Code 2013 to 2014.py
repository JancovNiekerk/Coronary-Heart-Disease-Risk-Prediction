# -*- coding: utf-8 -*-
"""
Created on Sun Apr 01 18:08:29 2018

@author: Janco
"""
#Revision 3 of Project 2
#DATA WRANGLING FOR 2013 TO 2014

import pandas as pd
import os
import numpy as np

path_RXQ_RX_H = os.path.join('.', 'Datasets', '2013 to 2014', 'RXQ_RX_H.XPT')
path_CDQ_H = os.path.join('.', 'Datasets', '2013 to 2014', 'CDQ_H.XPT')
path_TRIGLY_H = os.path.join('.', 'Datasets', '2013 to 2014', 'TRIGLY_H.XPT')
path_MCQ_H = os.path.join('.', 'Datasets', '2013 to 2014', 'MCQ_H.XPT')
path_DEMO_H = os.path.join('.', 'Datasets', '2013 to 2014', 'DEMO_H.XPT')

RXQ_RX_H = pd.read_sas(path_RXQ_RX_H)
CDQ_H = pd.read_sas(path_CDQ_H)
TRIGLY_H = pd.read_sas(path_TRIGLY_H)
MCQ_H = pd.read_sas(path_MCQ_H)
DEMO_H = pd.read_sas(path_DEMO_H)

df = RXQ_RX_H.merge(CDQ_H, left_on='SEQN', right_on='SEQN', how='outer')
df = df.merge(TRIGLY_H, left_on='SEQN', right_on='SEQN', how='outer')
df = df.merge(MCQ_H, left_on='SEQN', right_on='SEQN', how='outer')
df = df.merge(DEMO_H, left_on='SEQN', right_on='SEQN', how='outer')

#REMOVING HYPERCHOLESTEROLEMIA MEDICATION USERS, HEART ATTACK VICTIMS & DIAGNOSED CHD PATIENTS
#THE FOLLOWING OPERATION WILL BE UNIQUE FOR EACH YEAR
###################################################################################
DLTSEQN=df[df.RXDRSD1 == 'Pure hypercholesterolemia']
DLTSEQN = DLTSEQN.reset_index()
statin_users=DLTSEQN
DLTSEQN=DLTSEQN[['SEQN']]

df2=df.drop_duplicates(subset='SEQN', keep="last")

for i in range(len(DLTSEQN.SEQN)):
    sqn=DLTSEQN.SEQN[i]
    df=df[df.SEQN != sqn]

df=df.drop_duplicates(subset='SEQN', keep="last")

df=df.dropna(subset = ['MCQ160E'])
df=df[df.MCQ160E!=1]
###################################################################################

df['Angina']=np.nan #Adding Angina column to dataframe
df = df.reset_index()

for i in range(len(df.SEQN)):
    if df.MCQ160C[i]==1 or  (df.CDQ001[i]==1 and df.CDQ002[i]==1 and df.CDQ004[i]==1 and df.CDQ005[i]==1 and df.CDQ006[i]==1 and ((df.CDQ009D[i]==4 or df.CDQ009E[i]==5) or (df.CDQ009F[i]==6 and df.CDQ009G[i]==7))):
        df.Angina[i]="Yes"
    else:
        df.Angina[i]="No"

df.to_csv('CAD_and_Angina_cases_2013_to_2014.csv')
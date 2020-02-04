# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 00:28:41 2018

@author: Janco
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

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
        
df_temp=df #Saving a temporary dataframe which will be used as the training data for the logistic regression model

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

df=df[df.bin_LDL<6.15] #REMOVING OUTLIERS (this wont affect training data and is only used to improve visualisation)
df=df[df.Angint<0.3]   #REMOVING OUTLIERS (this wont affect training data and is only used to improve visualisation)
#saving the binned data in lists for plotting
x_vals = df['bin_LDL']
z_vals = df['Angint']
y_vals = df['bin_Age']

##################################################################################
## LOGISTIC REGRESSION MODELLING ##
################################################################################
df=df_temp #Resetting dataframe to non-binned format
clf = linear_model.LogisticRegression(C=1e5)


X=[]
for i in range(len(df)):
    LDL=df.ix[i, 'LBDLDLSI']
    Age=df.ix[i, 'RIDAGEYR']
    X+=[[LDL, Age]]

X=np.array(X)

#Splitting data into training and testing
X_train, X_test_actual, y_train, y_test_actual = train_test_split(X, df['Angint'].tolist(), test_size=0.33, random_state=42)

#clf.fit(X, df['Angint'].tolist())
clf.fit(X_train, y_train)
clf1=clf.coef_

for i in range(len(X)-1):
    clf1=np.vstack([clf1, clf.coef_])


def model(x):
    return 1 / (1 + np.exp(-x))

X_test=[[x,y]
        for x in LDL_bin_names[0:21]
        for y in Age_bin_names]

loss=[]
for i in range(len(X_test)):
    val=model(np.dot(X_test[i], clf1[i]) + clf.intercept_)
    loss +=[val[0]]

X1=[] #LDLvalues
X2=[] #Agevalues

for i in range(len(X_test)):
    X1 += [X_test[i][0]]
    X2 += [X_test[i][1]]


fig = plot.figure(1)
axis = Axes3D(fig, axisbg='g')

axis.scatter(x_vals, y_vals, z_vals, c=z_vals, marker='o', cmap='Reds')
axis.scatter(X1, X2, loss, c=loss, marker='*', cmap='Purples_r')
axis.set_xlabel("LDL-C (mmol/L)")
axis.set_ylabel("Age (Years)")
axis.set_zlabel("Risk")

axis.set_xlim(0, 6.15)

plot.show()

# Working out Precision
# Predict on X for model
loss2=[]
for i in range(len(X)):
    val=model(np.dot(X[i], clf1[i]) + clf.intercept_)
    loss2 +=[val[0]]
# Create list of Angint for df
angint = []
for i in range(len(X)):
    angint += [df.ix[i, 'Angint']]
# Create fucntion to convert for a certain threshold to 1 or 0
def convert_angint(thresh, losslist):
    converted_list = []
    for i in range(len(losslist)):
        if(losslist[i]>=thresh):
            converted_list += [1]
        else:
            converted_list += [0]
    return converted_list

precision_list = []
recall_list = []
f_score_list = []

def evaluate(thresh):
    converted_angint_list = convert_angint(thresh, loss2)
    # true positives calc
    tp = 0
    for i in range(len(X)):
        predicted_val = converted_angint_list[i]
        actual_val = angint[i]
        if(predicted_val==1 and actual_val==1):
            tp += 1.0
    # false positives calc
    fp = 0
    for i in range(len(X)):
        predicted_val_fp = converted_angint_list[i]
        actual_val_fp = angint[i]
        if(predicted_val_fp==1 and actual_val_fp==0):
            fp += 1.0
    # false negatives calc 
    fn = 0
    for i in range(len(X)):
        predicted_val_fn = converted_angint_list[i]
        actual_val_fn = angint[i]
        if(predicted_val_fn==0 and actual_val_fn==1):
            fn += 1.0
    # Precision calc
    precision = tp/(tp+fp)
    #print("precision : "+str(precision))
    recall = tp/(tp+fn)
    #print("recall : "+str(recall))
    #print("fn : "+ str(fn))
    #print("tp : "+str(tp))
    #print("fp : "+str(fp))
    f_score = (2*recall*precision)/(recall+precision)
    #print("f_score : "+str(f_score))
    temparr = np.array([precision, recall, f_score])
    return temparr
    
x_values = [x*0.005 for x in range(19)]
f_scores = [evaluate(x*0.005)[2] for x in range(19)]
precisions = [evaluate(x*0.005)[0] for x in range(19)]
recalls = [evaluate(x*0.005)[1] for x in range(19)]

plot.figure(2)
plot.plot(x_values, f_scores, label="f_scores")
#plot.plot(x_values, precisions, label="precision scores")
#plot.plot(x_values, recalls, label="recalls")
plot.legend()
plot.show()

#Calculate roc_auc_score
auc_score = metrics.roc_auc_score(angint, loss2)
print("ROC AUC Score = "+ str(auc_score))

#Calculate first accuracy @ threshold of 0.5
first_accuracy_score = metrics.accuracy_score(angint, convert_angint(0.5, loss2))
print("Accuracy at a threshold of 0.5 ="+str(first_accuracy_score*100)+" %")

#Calculate positive cases of angina and number of samples
positives_angint = angint.count(1)
total_angint_samples = len(angint)
print("positive cases of angina = "+ str(positives_angint))
print("total samples = "+ str(total_angint_samples))
print("fraction of negative cases of angina to total samples = "+str(((float(total_angint_samples)-float(positives_angint))/float(total_angint_samples))*100)+" %")

#Highest F-score is 0.09 at a threshold of 0.045
second_accuracy = metrics.accuracy_score(angint, convert_angint(0.035, loss2))
print("Accuracy at threshold of 0.035 = "+str(second_accuracy*100)+" %")
print("Recall at threshold of 0.035 = "+str(evaluate(0.035)[1]*100)+" %")
print("Precision at threshold of 0.035 = "+str(evaluate(0.035)[0]*100)+" %")


## New Work --- 30 Jan
loss_test=[]
for i in range(len(X_test_actual)):
    val=model(np.dot(X_test_actual[i], clf1[i]) + clf.intercept_)
    loss_test +=[val[0]]

accuracy_on_test = metrics.accuracy_score(y_test_actual, convert_angint(0.035, loss_test))
print("Accuracy on test data at threshold of 0.035 = "+str(accuracy_on_test*100)+" %")
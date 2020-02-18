# --------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

df = pd.read_csv(path)
print(df.columns)
print(df.head(2))
#print('The number of men and women in the dataset are', df['sex'].value_counts())

#print('The average age of women is', df[df['sex']=='Female']['age'].mean())

# Explore the data 
germandata = df[df['native-country']=='Germany']
percentage_of_german = germandata.shape[0]/df.shape[0]
print('The percentage of German citizens is', percentage_of_german)
# mean and standard deviation of their age
meanage1 = df[df['salary']== '>50K']['age'].mean()
print('The mean age of people who earn more than 50K is', meanage1)

meanage2 = df[df['salary']== '<=50K']['age'].mean()
print('The mean age of people who earn less than 50K is', meanage2)

stad1 = df[df['salary']== '>50K']['age'].std()
stad2 = df[df['salary']=='<=50K']['age'].std()
print('The std of age of people who earn more than 50K is', stad1)
print('The std of age of people who earn less than 50K is', stad2)

# Display the statistics of age for each gender of all the races (race feature).
df.groupby(['race','sex'])['age'].describe()

# encoding the categorical features.
df['salary'] = df['salary'].replace({'>50K':'1', '<=50K':'0'})
print(df['salary'].value_counts())


columnscat = df.select_dtypes(include='object').columns
print(columnscat)
print(df.info())

one_hot = pd.get_dummies(df[columnscat])
#print(one_hot.head(2))

df = df.drop(columnscat,axis = 1)
# Join the encoded df
df = df.join(one_hot)
#print(df.columns)
# Split the data and apply decision tree classifier
print(df.columns)
X= df.iloc[:,:-1]


#print(y.head(1))
# Perform the boosting task
 

#  plot a bar plot of the model's top 10 features with it's feature importance score


#  Plot the training and testing error vs. number of trees




# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 01:48:52 2018

@author: ravul
"""

# Importing required packages
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import random
import matplotlib.pyplot as plt


# Generation random data set
x = []
Y = []
for i in range(5000):
    height = random.randint(150,180)
    weight = random.randint(60,85)
    shoeSize = random.randint(36, 48)
    gender = random.choice(['male', 'female'])
    x.append([height,weight,shoeSize])
    Y.append(gender)


results_all_algos = []
scoring = 'accuracy'
seed = 7 # Configuration for the cross validation harness
kfold = KFold(n_splits = 10, random_state = seed)

# Training and prediction ( Decision tree classifier )
DecisionTree = DecisionTreeClassifier()
result =  cross_val_score(DecisionTree, x, Y, cv=kfold, scoring = scoring)
results_all_algos.append(['Decision Tree', result.mean(), result.std()])

# Training and prediction (Logistic Regression)
LogisticRegre = LogisticRegression()
result = cross_val_score(LogisticRegre, x, Y, cv=kfold, scoring = scoring)
results_all_algos.append(['Logistic Regression', result.mean(), result.std()])

# Training and predicion (SVC - Support Vector Classification)
SupportVectorClassi = SVC()
result = cross_val_score(SupportVectorClassi, x, Y, cv=kfold, scoring = scoring)
results_all_algos.append(['SVC', result.mean(), result.std()])

# Training and prediction (KNN)
KNearestNeb = KNeighborsClassifier()
result = cross_val_score(KNearestNeb, x, Y, cv=kfold, scoring=scoring)
results_all_algos.append(['KNN', result.mean(), result.std()])

print(results_all_algos)
names = [reslt[0] for reslt in results_all_algos]
results = [[reslt[1],reslt[2]] for reslt in results_all_algos]

# Box plot for the algorithm performance comparision
fig = plt.figure()
fig.suptitle('Algorithm Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



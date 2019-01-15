# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:01:38 2019

@author: vishr
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
y_pred_class = dtc.predict(X_test)
print("The predicted classes are:",y_pred_class)
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))
print(" A 3x3 confusion matrix is obtained",metrics.confusion_matrix(y_test,y_pred_class))
print(classification_report(y_test,y_pred_class))
confusion=metrics.confusion_matrix(y_test,y_pred_class);
sen_A=confusion[0,0]/(confusion[0,0]+confusion[0,1]+confusion[0,2])
sen_B=confusion[0,1]/(confusion[0,0]+confusion[0,1]+confusion[0,2])
sen_C=confusion[0,2]/(confusion[0,0]+confusion[0,1]+confusion[0,2])
print("Sensitivity of class 0, class 1 and class 2 are",sen_A,sen_B,sen_C)
spec_A=(confusion[1,1]+confusion[1,2]+confusion[2,1]+confusion[2,2])/(confusion[1,0]+confusion[1,1]+confusion[1,2]+confusion[2,0]+confusion[2,1]+confusion[2,2])
spec_B=(confusion[0,0]+confusion[0,2]+confusion[2,0]+confusion[2,2])/(confusion[0,0]+confusion[0,1]+confusion[0,2]+confusion[2,0]+confusion[2,1]+confusion[2,2])
spec_C=(confusion[0,0]+confusion[0,1]+confusion[1,0]+confusion[1,1])/(confusion[0,0]+confusion[0,1]+confusion[0,2]+confusion[1,0]+confusion[1,1]+confusion[1,2])
print("Specificity of class 0,class 1 and class 2 are",spec_A,spec_B,spec_C)
print("Classification error",1-metrics.accuracy_score(y_test,y_pred_class))

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:28:00 2019

@author: vishrutha
"""
from sklearn.datasets import load_iris
iris = load_iris()
X=iris.data
y = iris.target
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X);
print("The predictions are:",y_pred)
print("The accuracy is ",metrics.accuracy_score(y, y_pred))
print(" A 3x3 confusion matrix is obtained",metrics.confusion_matrix(y,y_pred))
print(classification_report(y,y_pred))
confusion=metrics.confusion_matrix(y,y_pred);
sen_A=confusion[0,0]/(confusion[0,0]+confusion[0,1]+confusion[0,2])
sen_B=confusion[0,1]/(confusion[0,0]+confusion[0,1]+confusion[0,2])
sen_C=confusion[0,2]/(confusion[0,0]+confusion[0,1]+confusion[0,2])
print("Sensitivity of class 0, class 1 and class 2 are",sen_A,sen_B,sen_C)
spec_A=(confusion[1,1]+confusion[1,2]+confusion[2,1]+confusion[2,2])/(confusion[1,0]+confusion[1,1]+confusion[1,2]+confusion[2,0]+confusion[2,1]+confusion[2,2])
spec_B=(confusion[0,0]+confusion[0,2]+confusion[2,0]+confusion[2,2])/(confusion[0,0]+confusion[0,1]+confusion[0,2]+confusion[2,0]+confusion[2,1]+confusion[2,2])
spec_C=(confusion[0,0]+confusion[0,1]+confusion[1,0]+confusion[1,1])/(confusion[0,0]+confusion[0,1]+confusion[0,2]+confusion[1,0]+confusion[1,1]+confusion[1,2])
print("Specificity of class 0,class 1 and class 2 are",spec_A,spec_B,spec_C)
print("Classification error",1-metrics.accuracy_score(y,y_pred))


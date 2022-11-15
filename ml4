Implement K-Nearest Neighbors algorithm on diabetes.csv dataset. Compute confusion matrix, accuracy, error rate, precision and recall on the given dataset.¶

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df=pd.read_csv("diabetes.csv") #Reading the Dataset
df.head()

df.dtypes

df["Glucose"].replace(0,df["Glucose"].mean(), inplace=True)
df["BloodPressure"].replace(0,df["BloodPressure"].mean(), inplace=True)
df["SkinThickness"].replace(0,df["SkinThickness"].mean(), inplace=True)
df["Insulin"].replace(0,df["Insulin"].mean(), inplace=True)
df["BMI"].replace(0,df["BMI"].mean(), inplace=True)
df.head()

X = df.iloc[:, :8]
Y = df.iloc[:, 8:]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

knn = KNeighborsClassifier(n_neighbors=5) #KNN Model
apply_model(knn)

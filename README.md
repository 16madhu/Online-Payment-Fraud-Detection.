# Online-Payment-Fraud-Detection.
"An AI/ML project that detects fraudulent online payment transactions using data analysis and machine learning to enhance security and prevent financial losses."
CODE:
import pandas as pd
import numpy as np
data = pd.read_csv("filepath/onlinefraud.csv")
print(data.isnull().sum())
#explore transaction type
data.type.value_counts()
type=data["type"].value_counts()
print(type)
transaction=type.index
quantity=type.values
import plotly.express as px
figure=px.pie(data,values=quantity,names=transaction,hole=0.5,title="distor of transaction type")
figure.show()
numeric_cols=data.select_dtypes(include=['float64','int64'])
correlation=numeric_cols.corr()
print(correlation)
correlation["isFraud"].sort_values(ascending=False)
data["type"]=data["type"].map({"CASH_OUT":1,"PAYMENT":2,"CASH_IN":3,"TRANSFER":4,"DEBIT":5})
data["isFraud"]=data["isFraud"].map({0:"no fraud",1:"fraud"})
data.head(5)
#train the model
from sklearn.model_selection import train_test_split
x=np.array(data[["type","amount","oldbalanceOrg","newbalanceOrig"]])
y=np.array(data[["isFraud"]])
from sklearn.tree import DecisionTreeClassifier
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.10,random_state=42)
model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))
#sklearn randomforestclassifier silently converts array to float32

#solve:
#we need to change
#prediction
features=np.array([[4,9000.0,9000.0,0.0]])
print(model.predict(features))

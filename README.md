# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Samuel M
RegisterNumber:  212222040142
*/


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred =reg.predict(X_test)
print(Y_pred)
print(Y_test)
#Graph plot for training data
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
1.df.head()

![image](https://github.com/ARJUN19122004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119429483/4347ec84-8cfd-4722-8707-442b74ed84db)

2.df.tail()

![image](https://github.com/ARJUN19122004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119429483/b3b57e12-c29c-4470-ba57-9154e6e307e3)

3.Array Value of X

![image](https://github.com/ARJUN19122004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119429483/68130589-41ea-46d8-8cf2-c4bbcd18109e)

4.Array Value of Y

![image](https://github.com/ARJUN19122004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119429483/23b868d2-50f3-4830-9d61-f87da1174b64)

5.Values of Y prediction

![image](https://github.com/ARJUN19122004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119429483/6a2754c0-ba1e-4392-bae3-6785e239d5b6)

6.Array values of Y test

![image](https://github.com/ARJUN19122004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119429483/b2ffaeac-8a6f-4dd6-a924-d0281225adf6)

7.Training Set Graph

![image](https://github.com/ARJUN19122004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119429483/1d570b06-4905-48f6-9e67-e9888bab1b2a)

8.Test Set Graph

![image](https://github.com/ARJUN19122004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119429483/0e33ead9-327e-47ff-9afa-0713f8895b6d)

9.Values of MSE, MAE and RMSE

![image](https://github.com/ARJUN19122004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119429483/3cff30f6-5e80-4013-b71d-691f6a4e5906)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

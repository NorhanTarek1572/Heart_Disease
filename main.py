# 1. importing  the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score


# 2. data collection


heart_data=pd.read_csv('D:\projects\Machine Learning\Heart_Disease\heart_disease_data.csv')
"""
# some operation on datasets 
print(heart_data.head(10)) # Print the first 10 rows

print(heart_data.tail(10)) # Print last 10 rows

# number of rows and columns in the dataset
print(heart_data.shape) # print(heart_data.shape) not print(" number of rows and columns in the dataset"+heart_data.shape) => we can not concatenate string with number 

# statistical measures about the data
print(heart_data.describe())

# show the number of the  missing value in all column
print(heart_data.isnull().sum())


# statistical measures about the data
heart_data.describe()

# checking the distribution of Target Variable
heart_data['target'].value_counts()

"""
X = heart_data.drop(columns='target', axis=1)  # all dataset except the result
Y = heart_data['target']   # the result dataset
"""
print(X)
print(Y)
"""



# 3. Splitting the Data into Training data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)  # testing 20%   ,  training 80%
# print(X.shape, X_train.shape, X_test.shape)


# 4. Model Training

# ( 4.1. Logistic Regression)
model = LogisticRegression()
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# ( 4.2 ANN)

# ( 4.3 SVM)

# 5. Model Evaluation(Accuracy Score)

# 5.1  accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# 5.2  accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)




# 6. Building a Predictive System

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)
# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')


#importing the libraries
import numpy as np 
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

#importing the dataset
dataset = pd.read_csv('coronaTrain.csv')
test_dataset = pd.read_csv('coronaTest.csv')

#extracting the test and train data from the dataset imported earlier 
X = dataset.iloc[:, 1:1001].values
Y = dataset.iloc[:, 1001].values
X_test_origin = test_dataset.iloc[:, 1:1001].values

#Initializing the ANN
model = Sequential()

#Adding the First Layer of the ANN 
model.add(Dense(900, activation='relu', input_shape = (1000,))) 

#Adding the Second Layer of the ANN 
model.add(Dense(820, activation='relu'))
#Dropping out 0.25*820 neurons
model.add(Dropout(0.25))

#Adding the 3rd Layer of the ANN
model.add(Dense(750, activation='relu'))

#Adding the 4th Layer of the ANN and Dropout
model.add(Dense(630, activation='relu'))
model.add(Dropout(0.15))

#Adding the 5th Layer of the ANN and Dropout
model.add(Dense(510, activation='relu'))
model.add(Dropout(0.15))

#Adding the 6th Layer of the ANN and Dropout
model.add(Dense(375, activation='relu'))
model.add(Dropout(0.1))

#Adding the 7th Layer of the ANN
model.add(Dense(215, activation='relu'))

#Adding the 8th Layer of the ANN
model.add(Dense(120, activation='relu'))

#Adding the 9th layer of the ANN
model.add(Dense(45, activation='relu'))

#Adding the final Output Layer of the ANN
model.add(Dense(3, activation='softmax'))

#Compiling the Model
model.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) 

#Fitting the dataset to the model 
model.fit(X, Y, batch_size = 10, epochs = 20)

#Predicting the results 
Y_pred_actual = model.predict(X_test_origin)

#Converting Raw Result to final data:
Y_pred = []
for i in range(0, len(Y_pred_actual)):
    if Y_pred_actual[i][0] > Y_pred_actual[i][1]:
        if Y_pred_actual[i][0] > Y_pred_actual[i][2]:
            Y_pred.append(0)
        else:
            Y_pred.append(2)
    else:
        if Y_pred_actual[i][1] > Y_pred_actual[i][2]:
            Y_pred.append(1)
        else:
            Y_pred.append(2)
            
#Preparing the resut set 
Status_d = pd.DataFrame(Y_pred)
Status_d.columns = ['Status']
Index_d = pd.DataFrame(test_dataset.iloc[:, 0].values)
Index_d.columns = ['Index']
result_set = pd.concat(objs=[Index_d, Status_d], axis=1, sort=False)
result_set.to_csv('Test_Final_27.csv', index = False)

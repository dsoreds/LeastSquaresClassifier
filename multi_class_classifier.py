import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("iris_data.csv")

data_setosa = dataset[0:50]
data_versicolor = dataset[50:100]
data_virginica = dataset[100:150]

test_size = 0.50

X_setosa = data_setosa.values[:,0:4].astype(np.float64)
Y_setosa = data_setosa.values[:,4]
X_train_setosa, X_test_setosa, Y_train_setosa, Y_test_setosa = train_test_split(X_setosa, Y_setosa, test_size = test_size)

X_versicolor = data_versicolor.values[:,0:4].astype(np.float64)
Y_versicolor = data_versicolor.values[:,4]
X_train_versicolor, X_test_versicolor, Y_train_versicolor, Y_test_versicolor = train_test_split(X_versicolor, Y_versicolor, test_size = test_size)

X_virginica = data_virginica.values[:,0:4].astype(np.float64)
Y_virginica = data_virginica.values[:,4]
X_train_virginica, X_test_virginica, Y_train_virginica, Y_test_virginica = train_test_split(X_virginica, Y_virginica, test_size = test_size)

X_train = np.array([])
X_train_temp = np.concatenate((X_train_setosa, X_train_versicolor, X_train_virginica), axis = 0)
for x_data in X_train_temp:
    x_data = np.append(x_data, [1])
    X_train = np.concatenate((X_train, x_data))
size = int(((150-test_size*(150))-1))
X_train = X_train.reshape((size,5))

Y_train_class = np.concatenate((Y_train_setosa, Y_train_versicolor, Y_train_virginica), axis=0)
Y_train = np.array([])
for cls in Y_train_class:
    if cls == 'Iris-setosa':
        class_arry = [1, 0, 0]
    elif cls == 'Iris-versicolor':
        class_arry = [0, 1, 0]
    elif cls == 'Iris-virginica':
        class_arry = [0, 0, 1]

    Y_train = np.concatenate((Y_train, class_arry), axis=0)
Y_train = Y_train.reshape((size,3))

dem = 0
num = 0
for i in range(size):
    a = np.array([X_train[i]])
    b = a.T
    c = b.dot(a)
    d = np.array([Y_train[i]])
    e = d.T
    f = b.dot(d)
    num = num + f
    dem = dem + c

lamda = 1000

dem = dem + lamda

deminv = inv(np.matrix(dem))
w = deminv.dot(num)

size_test = int(test_size*150)
X_test = np.array([])
X_test_temp = np.concatenate((X_test_setosa, X_test_versicolor, X_test_virginica), axis = 0)
for x_test_data in X_test_temp:
    x_test_data = np.append(x_test_data, [1])
    X_test = np.concatenate((X_test, x_test_data))
X_test = X_test.reshape((size_test,5))

predictions = np.array([])
for test_data in X_test:
    v1 = w[:,0].T.dot(test_data)
    v2 = w[:,1].T.dot(test_data)
    v3 = w[:,2].T.dot(test_data)
    if ((v1 > v2) and (v1 > v3)):
        predictions = np.append(predictions, [0])
    elif ((v2 > v1) and (v2 > v3)):
        predictions = np.append(predictions, [1])
    else:
        predictions = np.append(predictions, [2])


Y_test_class = np.concatenate((Y_test_setosa, Y_test_versicolor, Y_test_virginica), axis=0)
Y_test = np.array([])
for cls in Y_test_class:
    if cls == 'Iris-setosa':
        Y_test = np.append(Y_test, [0])
    elif cls == 'Iris-versicolor':
        Y_test = np.append(Y_test, [1])
    elif cls == 'Iris-virginica':
        Y_test = np.append(Y_test, [2])


cm = confusion_matrix(Y_test, predictions)
print("Confusion  Matrix:")
print(cm)

accuracy1 = accuracy_score(Y_test[:(int(size_test/3))], predictions[:int(size_test/3)])
error_rate1 = 1 - accuracy1
print("Misclassification error for setosa class : {}".format(error_rate1))

accuracy2 = accuracy_score(Y_test[int(size_test/3):int((size_test/3)*2)], predictions[int(size_test/3):int((size_test/3)*2)])
error_rate2 = 1 - accuracy2
print("Misclassification error for versicolor class : {}".format(error_rate2))

accuracy3 = accuracy_score(Y_test[int((size_test/3)*2):], predictions[int((size_test/3)*2):])
error_rate3 = 1 - accuracy3
print("Misclassification error for vriginica class : {}".format(error_rate3))

overallacc = accuracy_score(Y_test, predictions)
overall_error = 1 - overallacc
print("Overall misclassification error : {}".format(overall_error))
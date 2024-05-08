from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#get the dataset
#X, y = load_digits('mnist_784', version=1, return_X_y=True) #X is the data, y is the target, 
'''mnist_784 is the dataset name
#return_X_y=True returns the data and target as separate variables'''
digits = load_digits()
X = digits.data
y = digits.target
#create train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

#create our MLP model
model = MLPClassifier(hidden_layer_sizes=(20,20, 20 )) #one hidden layer with 20 neurons

#train the model
model.fit(X_train, y_train)

#test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
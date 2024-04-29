#cambia nome
'''
Digits classification with scikit learn

'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
#load the dataset
digits = load_digits()
#print(digits.images.shape)

#fig, axes = plt.subplots(8, 8, figsize = (8,8), subplot_kw = {'xticks':[],'yticks' : []}, gridspec_kw = dict(hspace = 0.1, wspace = 0.1)) #numbers on x-axis
'''
#plot a grid of images
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = 'binary')
    ax.text(0.05, 0.05, str(digits.target[i]), color = 'green')

plt.show()
'''
#get the data
X = digits.data

y = digits.target

#split the data into training and testing datasets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 0, train_size = 0.8)

#create an instance of the model 
model = GaussianNB()
#train the model 
model.fit(Xtrain, ytrain)
#use the model on new data
y_pred = model.predict(Xtest)

#compute the accurancy
acc = accuracy_score(ytest, y_pred)
#plot the confusion matrix
mat = confusion_matrix(ytest, y_pred)

sns.heatmap(mat, square = True, annot=True, cbar = False)
plt.xlabel('predicted value')
plt.ylabel('true value')

plt.show()
#print(acc)

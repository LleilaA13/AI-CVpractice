import seaborn as sns
import matplotlib.pylab as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
iris = sns.load_dataset('iris')
#print(iris.head())

#our_table['Name']

#sns.pairplot(iris, hue = 'species')
#plt.show()

X_iris = iris.drop('species', axis = 1) #remove column  -> (axis = 1)
#print(X_iris.shape)
y_iris = iris['species']
#print(y_iris.shape)

model = GaussianNB()
model.fit(X_iris, y_iris)

y_pred = model.predict(X_iris)
accuracy = accuracy_score(y_pred, y_iris)

print(accuracy)
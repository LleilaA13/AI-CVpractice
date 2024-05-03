# import the dataset and the modules
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# get a subset of categories 
categories = ['talk.religion.misc', 'sci.space', 'comp.graphics']

#load the dataset:
train = fetch_20newsgroups(subset = 'train', categories=categories) #data loader program that picks the data from you computer, you can build a dataset 

test = fetch_20newsgroups(subset = 'test', categories=categories)

#prepare the data and create the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

#train the model with the training data
model.fit(train.data, train.target)

#predict the labels of the test data
labels = model.predict(test.data)   

#plot the confusion matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')

plt.ylabel('predicted label')
plt.show()  
'''
#predict the category of a new text
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

print(predict_category('sending a payload to the ISS')) #sci.space
print(predict_category('discussing islam vs atheism')) #talk.religion.misc          #this is a simple text classifier that uses the Naive Bayes algorithm to classify text into categories              
'''

#try the classifier with a new text
s = 'god is love'
pred = model.predict([s])
print(train.target_names[pred[0]])

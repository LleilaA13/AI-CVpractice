from sklearn.feature_extraction import DictVectorizer

data = [
    {'price': 834648, 'rooms': 4, 'neighborhood': 'Tiburtina'},
    {'price': 324554, 'rooms': 3, 'neighborhood': 'Tuscolana'},
    {'price': 145875, 'rooms': 3, 'neighborhood': 'Appia'}
]

#{ 'Tiburtina':1, 'Tuscolana': 2, 'Appia': 3}

# convert data to one hot enconding representation
vec = DictVectorizer(sparse=False, dtype=int)

result = vec.fit_transform(data)  # allows us to transform data

names = vec.feature_names_
print(names)

'''
one hot encoding consists into converting into a set of digits, encoding
'''
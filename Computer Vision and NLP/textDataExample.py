from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = [
    'problem of evil',
    'evil queen',
    'horizon problem'
]

vec = CountVectorizer() #how many times a word appears?

X  = vec.fit_transform(data) #transform data, 2 operations done: first you fit then u transform  
'''
#convert the data into a pandas table for readability
vis_data = pd.DataFrame(X.toarray(),columns=vec.get_feature_names_out() ) #first thing is giving the data, then we need to provide the name of the cols that we want to use        

print(vis_data)
# TF -IDF = term freq - inverse doc freq
'''

vec2 = TfidfVectorizer()
X2 = vec2.fit_transform(data)

vis_data2 = pd.DataFrame(X2.toarray(), columns=vec2.get_feature_names_out())

print(vis_data2)
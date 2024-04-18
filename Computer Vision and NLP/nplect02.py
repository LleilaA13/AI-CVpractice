from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

quote = "It's leviosa not leviosaaa"

words_in_quote = word_tokenize(quote)
#print(words_in_quote)

stop_words = set(stopwords.words('english'))
filtered_list = []

for word in words_in_quote:
    if word.casefold() not in stop_words:
        filtered_list.append(word)

print(filtered_list)
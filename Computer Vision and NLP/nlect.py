from nltk.tokenize import sent_tokenize, word_tokenize
#tokenizing makes us split smth into smaller parts

ex_string = "It's dangerous to go alone, Please take this. SImone è gay"

res = word_tokenize(ex_string)
res2 = ex_string.split()
sentRes = sent_tokenize(ex_string)

print(res)
print(sentRes)
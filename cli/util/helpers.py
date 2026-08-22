import json
import string
from nltk.stem import PorterStemmer

table = str.maketrans("","",string.punctuation)
with open("data/stopwords.txt", 'r', encoding='utf-8') as stopwords_file_P:
    stopwords_file = stopwords_file_P.read()
stopwords_list = stopwords_file.splitlines()
for index in range(len(stopwords_list)):
    stopwords_list[index] = (stopwords_list[index].translate(table).lower())
stopwords_set = set(stopwords_list)

def load_movies() -> dict:
    with open("data/movies.json", 'r', encoding='utf-8') as movies_file_P:
        return json.load(movies_file_P)

def tokenize(string):
    
    stemmer = PorterStemmer()
    
    string = string.lower().translate(table)
    temp = []
    temp = string.split()
    index_list=[]
    for index in range(len(temp)):
        if temp[index] == " " or temp[index] in stopwords_set:
            index_list.append(index)
    reverse_index = reversed(index_list)
    for index in reverse_index:  
        temp[index] = " "
        temp.remove(" ")
    for index in range(len(temp)):
        temp[index] = stemmer.stem(temp[index])
    return temp

def single_tokenize(string):
    temp = tokenize(string)
    if(len(temp)==1):
        return temp[0]
    else:
        raise Exception 
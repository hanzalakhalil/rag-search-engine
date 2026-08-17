import argparse
import json
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

table = str.maketrans("","",string.punctuation)
stopwords_file_P=open("data/stopwords.txt", 'r', encoding='utf-8')
stopwords_file = stopwords_file_P.read()
stopwords_list = stopwords_file.splitlines()
for index in range(len(stopwords_list)):
    stopwords_list[index] = (stopwords_list[index].translate(table).lower())
stopwords_set = set(stopwords_list)
    

def main() -> None:
    
    
    movies_file_P=open("data/movies.json", 'r', encoding='utf-8')
    movies_dict=json.load(movies_file_P)
    
    
    results=[]

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            query_list = tokenize(args.query.lower().translate(table))
            print(query_list)
            
            for movie in movies_dict["movies"]:
                title_list = tokenize(movie["title"].translate(table).lower())
                if set(query_list) & set(title_list):
                    results.append(movie["title"])
            print(f"Searching for: {args.query}")  
             
            if(len(results)>=5):       
                for index in range(5):
                    print(f'{index+1}. {results[index]}')
            else:
                for index in range(len(results)):
                    print(f'{index+1}. {results[index]}')
            pass
        case _:
            parser.print_help()
            
    movies_file_P.close()
    stopwords_file_P.close()

def tokenize(string):
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

if __name__ == "__main__":
    main()

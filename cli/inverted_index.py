import keyword_search_cli
import pickle
import helpers
from collections import Counter

counter = Counter()
class InvertedIndex:
    
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        
    def __add_document(self,doc_id,text):
        text = helpers.tokenize(text)
        self.term_frequencies.setdefault(doc_id, Counter())
        for token in text:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] +=1
            
    def get_document(self,term):
        return (sorted(self.index.get(term,set())))
    
    def get_tf(self,doc_id,term):
        return self.term_frequencies.get(doc_id,-1).get(term,0)

    def build(self):
        movies_dict=helpers.load_movies()
        for m in movies_dict["movies"]:
            self.__add_document(m["id"],f"{m["title"]} {m["description"]}")
            self.docmap[m["id"]]= m
        self.save()
        
    
    def save(self):
        with open("cache/index.pkl","wb") as index_pkl:
            pickle.dump(self.index,index_pkl)
            
        with open("cache/docmap.pkl","wb") as docmap_pkl:
            pickle.dump(self.docmap,docmap_pkl)
            
        with open("cache/term_frequencies.pkl","wb") as tf_pkl:
            pickle.dump(self.term_frequencies,tf_pkl)
            
    def load(self):
        with open("cache/index.pkl","rb") as index_pkl:
            self.index = pickle.load(index_pkl)
            
        with open("cache/docmap.pkl","rb") as docmap_pkl:
            self.docmap = pickle.load(docmap_pkl)
            
        with open("cache/term_frequencies.pkl","rb") as tf_pkl:
            self.term_frequencies = pickle.load(tf_pkl)
        
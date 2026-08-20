import pickle
import util.helpers as helpers
import math
import util.constants as constants
from collections import Counter
import os


counter = Counter()
class InvertedIndex:
    
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.index_path = os.path.join(constants.CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(constants.CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(constants.CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies = {}
        self.doc_lengths_path = os.path.join(constants.CACHE_DIR, "doc_lengths.pkl")
        self.doc_lengths = {}
        
    def __add_document(self,doc_id,text):
        text = helpers.tokenize(text)
        self.doc_lengths[doc_id]=len(text)
        self.term_frequencies.setdefault(doc_id, Counter())
        for token in text:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] +=1
            
    def __get_avg_doc_length(self):
        if(not self.doc_lengths or len(self.doc_lengths)==0):
            return 0.0
        total_length = 0
        for length in self.doc_lengths.values():
            total_length += length
        return total_length / len(self.doc_lengths)
            
            
    def get_document(self,term):
        return (sorted(self.index.get(term,set())))
    
    def get_tf(self,doc_id,term):
        return self.term_frequencies.get(doc_id,-1).get(term,0)
    
    def get_bm25_idf(self,term):
        bm25_idf = math.log(((len(self.docmap) - len(self.get_document(term)) + 0.5) / (len(self.get_document(term)) + 0.5)) + 1)
        return bm25_idf
        
    def get_bm25_tf(self, doc_id, term,k1=constants. BM25_K1, b=constants.BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def bm25(self,doc_id,term):
        bm25_idf= self.get_bm25_idf(term)
        bm25_tf= self.get_bm25_tf(doc_id,term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query, limit):
        query_list = helpers.tokenize(query)
        scores={}
        temp=[]
        for token in query_list:
            temp=self.index.get(token,[])
            for doc_id in temp:
                scores[doc_id]=scores.get(doc_id,0.0)+self.bm25(doc_id,token)
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs[:limit]
        
    def build(self):
        movies_dict=helpers.load_movies()
        for m in movies_dict["movies"]:
            self.__add_document(m["id"],f"{m["title"]} {m["description"]}")
            self.docmap[m["id"]]= m
        self.save()
        
    
    def save(self):
        with open(self.index_path,"wb") as f:
            pickle.dump(self.index,f)
            
        with open(self.docmap_path,"wb") as f:
            pickle.dump(self.docmap,f)
            
        with open(self.tf_path,"wb") as f:
            pickle.dump(self.term_frequencies,f)
            
        with open(self.doc_lengths_path,"wb") as f:
            pickle.dump(self.doc_lengths,f)            
            
    def load(self):
        with open(self.index_path,"rb") as f:
            self.index = pickle.load(f)
            
        with open(self.docmap_path,"rb") as f:
            self.docmap = pickle.load(f)
            
        with open(self.tf_path,"rb") as f:
            self.term_frequencies = pickle.load(f)
            
        with open(self.doc_lengths_path,"rb") as f:
            self.doc_lengths = pickle.load(f)
        
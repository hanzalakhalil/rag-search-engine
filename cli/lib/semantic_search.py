from sentence_transformers import SentenceTransformer
import numpy as np
import os
import util.helpers as helpers
import json
import re


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        
    def search(self, query: str, limit: int):
        
        if self.embeddings.shape is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scores=[]
        i=1
        for embedding in self.embeddings: 
            similarity_score = cosine_similarity(query_embedding,embedding)
            scores.append((similarity_score,self.document_map[i]))
            i+=1
        temp_list = sorted(scores,key= lambda pair: pair[0] ,reverse=True)    
        temp_list = temp_list[:limit] 
        results_list = []
        for score, doc in temp_list:
            results_dict = {}
            results_dict["score"]=score
            results_dict["title"]=doc["title"]
            results_dict["description"]=doc["description"]
            results_list.append(results_dict)
        return results_list
        
        
    def build_embedding(self, documents: list[dict]) -> list[list[float]]:
        self.documents = documents
        text=[]
        for doc in documents:
            self.document_map[doc["id"]]=doc
            text.append( f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(text,show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]) -> list[list[float]]:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]]=doc
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy",)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        return self.build_embedding(documents)
        
    def generate_embedding (self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("The query is empty")
        embedding = self.model.encode([text])
        return embedding[0]
  
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        chunk_list:list[str]=[]
        metadata:list[dict]=[]
        for movie_idx,doc in enumerate(self.documents):
            self.document_map[movie_idx]=doc
        temp_list:list[str]=[]
        for movie_idx,doc in enumerate(self.documents):
            if doc["description"]=="":
                continue
            for chunk in semantic_chunk_command(doc["description"],4,1):
                temp=""
                for sen in chunk:
                    temp= temp+ " " + sen 
                chunk_list.append(temp)
                temp_list.append(temp)
            for chunk_idx,chunk_text in enumerate(temp_list):
                metadata.append({"movie_idx":movie_idx,"chunk_idx":chunk_idx,"total_chunks":len(temp_list)})
            temp_list=[]

        
        self.chunk_metadata = metadata
        self.chunk_embeddings = self.model.encode(chunk_list,show_progress_bar=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunk_list)}, f, indent=2)    
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for movie_idx,doc in enumerate(self.documents):
            self.document_map[movie_idx]=doc
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings=np.load("cache/chunk_embeddings.npy",)
            with open("cache/chunk_metadata.json", "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(self.documents)

  
def verify_embeddings() -> None:
    search = SemanticSearch()
    documents = helpers.load_movies()["movies"]
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    print(embeddings.shape)
        
def embed_text(text) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model() -> None:
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
    
def embed_query_text(query: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
    
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int) -> None:
    search_instance = SemanticSearch()
    documents = helpers.load_movies()["movies"]
    search_instance.load_or_create_embeddings(documents)
    results = search_instance.search(query,limit)
    i=1
    for result in results:
        print(f'{i}. {result["title"]} {result["score"]}') 
        print(f'{result["description"]:.80} ...')
        i+=1
    
def chunk_command(text:str , chunk_size : int, overlap: int):
    text_list = text.split()
    chunk_list=[]
    for i in range(0,len(text_list),chunk_size):
        if(i-overlap+chunk_size < len(text_list)):
            if(i-overlap<0):
                chunk_list.append(text_list[i:i+chunk_size])
            else:
                chunk_list.append(text_list[i-overlap:i-overlap+chunk_size])
                
        else:
            chunk_list.append(text_list[i-overlap:])
            
    print(f"Chunking {len(text)} characters")
    i=0 
    for chunk in chunk_list:
        j=0
        temp:str=""
        for word in chunk:
            temp = temp + word+" "
            j+=1
        print(f"{i+1}. {temp}")
        i+=1
        
def semantic_chunk_command(text:str , max_chunk_size : int, overlap: int) -> list:
    text_list = re.split(r"(?<=[.!?])\s+",text) 
    chunk_list=[]
    for i in range(0,len(text_list),max_chunk_size-overlap):
        
        chunk_list.append(text_list[i:i+max_chunk_size])
        if(i+max_chunk_size >= len(text_list)):
            break
    return chunk_list

def embed_chunks():
    chunked_semantic_search = ChunkedSemanticSearch()
    documents = helpers.load_movies()["movies"]
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")
    
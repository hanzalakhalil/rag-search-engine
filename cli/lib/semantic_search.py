from sentence_transformers import SentenceTransformer
import numpy as np
import os
import util.helpers as helpers

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
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import util.helpers as helpers
import json
import re
import util.constants as constants



class SemanticSearch:
    """Embedding-based search over whole movie documents.

    Each document (title + description) is encoded into a single vector with a
    SentenceTransformer model, and queries are ranked by cosine similarity
    against those vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Sentence embedding model used for both documents and queries.
        self.model = SentenceTransformer(model_name)
        # Matrix of document embeddings, one row per document (set by build/load).
        self.embeddings = None
        # Raw list of document dicts the embeddings were built from.
        self.documents = None
        # Lookup from document id -> document dict.
        self.document_map = {}
        
    def search(self, query: str, limit: int):
        """Return the top `limit` documents most similar to `query`.

        Each result is a dict with "score", "title" and "description".
        Embeddings must be loaded first via `load_or_create_embeddings`.
        """
        
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scores=[]
        # Document ids are assumed to start at 1 and follow the same order as
        # the rows of self.embeddings.
        i=1
        for embedding in self.embeddings: 
            similarity_score = cosine_similarity(query_embedding,embedding)
            scores.append((similarity_score,self.document_map[i]))
            i+=1
        # Rank by similarity score (highest first) and keep only the top results.
        temp_list = sorted(scores,key= lambda pair: pair[0] ,reverse=True)    
        temp_list = temp_list[:limit] 
        # Convert (score, doc) pairs into plain result dicts for the caller.
        results_list = []
        for score, doc in temp_list:
            results_dict = {}
            results_dict["score"]=score
            results_dict["title"]=doc["title"]
            results_dict["description"]=doc["description"]
            results_list.append(results_dict)
        return results_list
        
        
    def build_embedding(self, documents: list[dict]) -> list[list[float]]:
        """Encode every document as "title: description" and cache the result.

        The embeddings are saved to cache/movie_embeddings.npy so later runs
        can skip re-encoding.
        """
        self.documents = documents
        text=[]
        for doc in documents:
            self.document_map[doc["id"]]=doc
            # Combine title and description so both contribute to the embedding.
            text.append( f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(text,show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]) -> list[list[float]]:
        """Load cached document embeddings, or build them if missing/stale.

        The cache is only reused when it has one embedding per document;
        otherwise the embeddings are rebuilt from scratch.
        """
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]]=doc
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy",)
            # Only trust the cache if its size matches the current document set.
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        return self.build_embedding(documents)
        
    def generate_embedding (self, text: str) -> list[float]:
        """Encode a single piece of text (e.g. a query) into an embedding vector."""
        if not text or not text.strip():
            raise ValueError("The query is empty")
        # encode() expects a list and returns one vector per input; take the first.
        embedding = self.model.encode([text])
        return embedding[0]
  
class ChunkedSemanticSearch(SemanticSearch):
    """Semantic search that embeds sentence-based chunks of each description.

    Instead of one vector per movie, each description is split into
    overlapping groups of sentences, and each group gets its own embedding.
    This lets a query match a specific passage within a long description.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        # Matrix of chunk embeddings, one row per chunk across all documents.
        self.chunk_embeddings = None
        # Per-chunk info (movie_idx, chunk_idx, total_chunks), aligned with
        # the rows of self.chunk_embeddings.
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Chunk every description, embed all chunks, and cache the results.

        Writes the embeddings to cache/chunk_embeddings.npy and the chunk
        metadata to cache/chunk_metadata.json.
        """
        self.documents = documents
        # All chunk texts across every document, in embedding order.
        chunk_list:list[str]=[]
        # Metadata entry for each chunk in chunk_list (same order).
        metadata:list[dict]=[]
        for doc in documents:
            self.document_map[doc["id"]] = doc
        # Chunks belonging to the current document only; reset per document.
        temp_list:list[str]=[]
        for movie_idx,doc in enumerate(self.documents):
            # Skip documents with no description; there is nothing to chunk.
            if doc["description"]=="":
                continue
            # Split into chunks of up to 4 sentences with 1 sentence of overlap.
            for chunk in semantic_chunk(doc["description"],4,1):
                # Join the sentences of this chunk back into a single string.
                temp=""
                for sen in chunk:
                    temp= temp+ " " + sen 
                chunk_list.append(temp)
                temp_list.append(temp)
            # Record where each chunk came from so search results can be
            # mapped back to their movie.
            for chunk_idx,chunk_text in enumerate(temp_list):
                metadata.append({"movie_idx":movie_idx,"chunk_idx":chunk_idx,"total_chunks":len(temp_list)})
            temp_list=[]

        
        self.chunk_metadata = metadata
        self.chunk_embeddings = self.model.encode(chunk_list,show_progress_bar=True)
        # Cache embeddings and metadata so they can be reloaded without re-encoding.
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunk_list)}, f, indent=2)    
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Load cached chunk embeddings and metadata, or build them if missing."""
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        # Both cache files must exist to reuse the cache.
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings=np.load("cache/chunk_embeddings.npy",)
            with open("cache/chunk_metadata.json", "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(self.documents)
    
    def search_chunks(self,query:str,limit:int=10) ->list[dict]:
        """Score every chunk against `query` (work in progress).

        Currently computes a similarity per chunk and looks up the chunk's
        source document, but does not yet collect, rank or return results.
        """
        query_embedding : list[float] = []
        chunk_score_list: list[dict] = []
        query_embedding = self.generate_embedding(query)
        # Load chunk embeddings and metadata from the on-disk cache.
        self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
        data = {}
        with open("cache/chunk_metadata.json", "r") as f:
            data = json.load(f)
        self.chunk_metadata=data["chunks"]
        for i,chunk_vec in enumerate(self.chunk_embeddings):
            # Similarity between the query and this chunk (result not stored yet).
            cosine_similarity(query_embedding,chunk_vec)
            
            # Map the chunk back to the movie it was taken from.
            meta = self.chunk_metadata[i]
            movie_idx = meta["movie_idx"]
            doc = self.documents[movie_idx]
  
def verify_embeddings() -> None:
    """CLI helper: load/build movie embeddings and print their shape."""
    search = SemanticSearch()
    documents = helpers.load_movies()["movies"]
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    print(embeddings.shape)
        
def embed_text(text) -> None:
    """CLI helper: embed `text` and print a preview of the vector."""
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model() -> None:
    """CLI helper: load the embedding model and print basic info about it."""
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
    
def embed_query_text(query: str) -> None:
    """CLI helper: embed a search query and print a preview of the vector."""
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
    
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return the cosine similarity of two vectors (0.0 if either is all zeros)."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Avoid division by zero for zero-length vectors.
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int) -> None:
    """CLI entry point: run a semantic search over movies and print the results."""
    search_instance = SemanticSearch()
    documents = helpers.load_movies()["movies"]
    search_instance.load_or_create_embeddings(documents)
    results = search_instance.search(query,limit)
    # Print a numbered list: title + score, then a truncated description.
    i=1
    for result in results:
        print(f'{i}. {result["title"]} {result["score"]}') 
        print(f'{result["description"]:.80} ...')
        i+=1
    
def chunk_command(text:str , chunk_size : int, overlap: int):
    """CLI entry point: split `text` into fixed-size word chunks and print them.

    Chunks are `chunk_size` words long; each chunk after the first starts
    `overlap` words before its nominal start so neighbouring chunks share words.
    """
    text_list = text.split()
    chunk_list=[]
    for i in range(0,len(text_list),chunk_size):
        if(i-overlap+chunk_size < len(text_list)):
            if(i-overlap<0):
                # First chunk: nothing before it to overlap with.
                chunk_list.append(text_list[i:i+chunk_size])
            else:
                # Shift the window back by `overlap` words.
                chunk_list.append(text_list[i-overlap:i-overlap+chunk_size])
                
        else:
            # Last chunk: take everything remaining from the overlapped start.
            chunk_list.append(text_list[i-overlap:])
            
    print(f"Chunking {len(text)} characters")
    # Print each chunk as a numbered line of space-joined words.
    i=0 
    for chunk in chunk_list:
        j=0
        temp:str=""
        for word in chunk:
            temp = temp + word+" "
            j+=1
        print(f"{i+1}. {temp}")
        i+=1
        
def semantic_chunk(text:str , max_chunk_size : int, overlap: int) -> list:
    """Split `text` into chunks of whole sentences.

    Each chunk holds up to `max_chunk_size` sentences, and consecutive chunks
    share `overlap` sentences. Returns a list of chunks, each a list of
    sentence strings.
    """
    # Split on whitespace that follows sentence-ending punctuation (. ! ?).
    text_list = re.split(r"(?<=[.!?])\s+",text) 
    chunk_list=[]
    # Step forward by (max_chunk_size - overlap) so chunks overlap.
    for i in range(0,len(text_list),max_chunk_size-overlap):
        
        chunk_list.append(text_list[i:i+max_chunk_size])
        # Stop once a chunk has reached the last sentence, to avoid emitting
        # trailing chunks that are fully contained in the previous one.
        if(i+max_chunk_size >= len(text_list)):
            break
    return chunk_list

def semantic_chunk_command(text:str , max_chunk_size : int, overlap: int):
    """CLI entry point: sentence-chunk `text` and print each chunk."""
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def embed_chunks():
    """CLI entry point: load/build chunk embeddings for all movies and report the count."""
    chunked_semantic_search = ChunkedSemanticSearch()
    documents = helpers.load_movies()["movies"]
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")
    
def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata
):
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, constants.SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
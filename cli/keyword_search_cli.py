import argparse
import util.inverted_index as inverted_index
import util.helpers as helpers
import util.constants as constants 
import math

def main() -> None:
    movies_dict=helpers.load_movies()

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="build index object")

    tf_parser = subparsers.add_parser("tf", help="term frequency")
    tf_parser.add_argument("doc_id", type=int, help = "Document ID")
    tf_parser.add_argument("term", type=str, help="Token")
    
    idf_parser = subparsers.add_parser("idf", help="Inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Token")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Term frequency * Inverse document frequency")
    tfidf_parser.add_argument("doc_id", type=int, help = "Document ID")
    tfidf_parser.add_argument("term", type=str, help="Token")
    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=constants.BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=constants.BM25_B, help="Tunable BM25 b parameter")
    
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    
            #make command for all cases
    match args.command:
        case "search":
            #make command
            inv = inverted_index.InvertedIndex()
            inv.load()
            results = []
            query_list = helpers.tokenize(args.query)
            
            temp_list=[]
            
            for query_token in query_list:
                temp_list=inv.get_document(query_token)[:5] 
            
            for id in temp_list:
                print(f"id={id} title={inv.docmap[id]['title']}")
            
            print(f"Searching for: {args.query}")  
        case "build":
            build_command()
            
        case "tf":
            #make command
            inv = inverted_index.InvertedIndex()
            inv.load()
            term = helpers.single_tokenize(args.term)
            print(args.doc_id,inv.get_tf(args.doc_id,term))
            #make command
        case "idf":
            #make command
            inv = inverted_index.InvertedIndex()
            inv.load()
            term = helpers.single_tokenize(args.term)
            idf = math.log((len(inv.docmap) + 1) / (len(inv.get_document(term)) + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":    
            #make command
            inv = inverted_index.InvertedIndex()
            inv.load()
            term = helpers.single_tokenize(args.term)
            tf = inv.get_tf(args.doc_id,term)
            idf = math.log((len(inv.docmap) + 1) / (len(inv.get_document(term)) + 1))
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")    
        case "bm25search":
            results = bm25_search_command(args.query)
            for line in results:
                print(line)
            
        case _:
            parser.print_help()

def build_command():
    inv = inverted_index.InvertedIndex()
    inv.build()

def bm25_idf_command(term):
    inv = inverted_index.InvertedIndex()
    inv.load()
    term = helpers.single_tokenize(term)
    return inv.get_bm25_idf(term)

def bm25_tf_command(doc_id, term, k1=constants.BM25_K1, b=constants.BM25_B):
    inv = inverted_index.InvertedIndex()
    inv.load()
    term = helpers.single_tokenize(term)
    return inv.get_bm25_tf(doc_id,term,k1,b)

def bm25_search_command(query,limit=5):
    inv = inverted_index.InvertedIndex()
    inv.load()
    scores = inv.bm25_search(query,limit)
    results=[]
    i=1
    for doc_id,bm25 in scores:
        results.append(f'{i}. ({doc_id}) {inv.docmap[doc_id]["title"]} - Score: {bm25:.2f}')
        i+=1
    return results
            
if __name__ == "__main__":
    main()

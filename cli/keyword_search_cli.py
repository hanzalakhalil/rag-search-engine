import argparse
import inverted_index
import helpers
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
    
    args = parser.parse_args()

    match args.command:
        case "search":
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
            
             
            # if(len(results)>=5):       
            #     for index in range(5):
            #         print(f'{index+1}. {results[index]}')
            # else:
            #     for index in range(len(results)):
            #         print(f'{index+1}. {results[index]}')
        case "build":
            
            inv = inverted_index.InvertedIndex()
            inv.build()
        case "tf":
            inv = inverted_index.InvertedIndex()
            inv.load()
            term = helpers.single_tokenize(args.term)
            print(args.doc_id,inv.get_tf(args.doc_id,term))
        case "idf":
            inv = inverted_index.InvertedIndex()
            inv.load()
            term = helpers.single_tokenize(args.term)
            idf = math.log((len(inv.docmap) + 1) / (len(inv.get_document(term)) + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":    
            inv = inverted_index.InvertedIndex()
            inv.load()
            term = helpers.single_tokenize(args.term)
            tf = inv.get_tf(args.doc_id,term)
            idf = math.log((len(inv.docmap) + 1) / (len(inv.get_document(term)) + 1))
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()
            
if __name__ == "__main__":
    main()

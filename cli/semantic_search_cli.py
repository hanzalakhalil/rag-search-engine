import argparse
import lib.semantic_search as ss
import util.helpers as helpers

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    verify_parser = subparsers.add_parser("verify" ,help="To verify model" )
    
    embed_parser = subparsers.add_parser("embed_text", help="Create embedding for given text")
    embed_parser.add_argument("text", type=str, help="Text to be embedded")
    
    verify_embed_parser = subparsers.add_parser("verify_embeddings", help="verify embeddings")
    
    embed_query_parser = subparsers.add_parser("embed_query", help="Create embedding for given query")
    embed_query_parser.add_argument("query", type=str, help="query to be embedded")
    
    search_parser = subparsers.add_parser("search", help="search")
    search_parser.add_argument("query", type=str, help="query")
    search_parser.add_argument("-l","--limit",default=5,help="optional limit")
    
    chunk_parser = subparsers.add_parser("chunk", help="for chunking")
    chunk_parser.add_argument("text", type=str, help="the text to make chunks out of")
    chunk_parser.add_argument("--chunk-size", default=200, type=int, help="number of words per chunk")
    chunk_parser.add_argument("--overlap",default=0, type=int, help="overlap between chunks")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="for semantic chunking")
    semantic_chunk_parser.add_argument("text", type=str, help="the text to make chunks out of")
    semantic_chunk_parser.add_argument("--max-chunk-size", default=4, type=int, help="number of words per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int,default=0, help="overlap between chunks")
    
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="for embedding chunks")
    
    args = parser.parse_args()
    
    

    match args.command:
        case "verify":
            ss.verify_model()
        case "embed_text":
            ss.embed_text(args.text)
        case "verify_embeddings":
            ss.verify_embeddings()
        case "embed_query":
            ss.embed_query_text(args.query)
        case "search":
            ss.search_command(args.query,int(args.limit))
        case "chunk":
            ss.chunk_command(args.text,args.chunk_size,args.overlap)
        case "semantic_chunk":
            ss.semantic_chunk_command(args.text,args.max_chunk_size,args.overlap)
        case "embed_chunks":
            ss.embed_chunks()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
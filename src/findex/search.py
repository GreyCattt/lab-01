import argparse
import pickle
import time
import tracemalloc
from pathlib import Path

from findex.tokenize import tokenize
from findex.models import Posting, DocMeta

def search_set(postings_dict: dict, terms: list[str]) -> set[int]:
    """Boolean AND пошук через перетин множин Python."""
    if not terms:
        return set()
    

    if any(term not in postings_dict for term in terms):
        return set()

    result = {p.doc_id for p in postings_dict[terms[0]]}
    
    for term in terms[1:]:
        result &= {p.doc_id for p in postings_dict[term]}
        
    return result

def search_merge(postings_dict: dict, terms: list[str]) -> list[int]:
    """Boolean AND пошук через двовказівниковий merge відсортованих списків."""
    if not terms:
        return []
        
    if any(term not in postings_dict for term in terms):
        return []
    sorted_terms = sorted(terms, key=lambda t: len(postings_dict[t]))
    result = [p.doc_id for p in postings_dict[sorted_terms[0]]]
    
    for term in sorted_terms[1:]:
        next_list = [p.doc_id for p in postings_dict[term]]
        merged = []
        i, j = 0, 0
        
        while i < len(result) and j < len(next_list):
            if result[i] == next_list[j]:
                merged.append(result[i])
                i += 1
                j += 1
            elif result[i] < next_list[j]:
                i += 1
            else:
                j += 1
                
        result = merged
        if not result:
            break
            
    return result

def main():
    parser = argparse.ArgumentParser(description="Search in Inverted Index")
    parser.add_argument("index_file", type=Path, help="Path to index.pkl")
    parser.add_argument("query", type=str, help="Search query (terms will be ANDed)")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Search algorithm")
    args = parser.parse_args()

    tracemalloc.start()
    start_load = time.perf_counter()

    # Завантажуємо індекс
    with args.index_file.open("rb") as f:
        postings_dict, metadata = pickle.load(f)

    load_time = time.perf_counter() - start_load

    # Токенізуємо запит
    query_terms = list(tokenize(args.query))
    
    start_search = time.perf_counter()
    
    if args.engine == "set":
        result_doc_ids = search_set(postings_dict, query_terms)
    else:
        result_doc_ids = search_merge(postings_dict, query_terms)

    search_time = time.perf_counter() - start_search
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Query terms: {query_terms}")
    print(f"Engine: {args.engine.upper()}")
    print(f"Found matches: {len(result_doc_ids)}")
    print(f"\nTime to load index: {load_time:.4f} s")
    print(f"Time to search: {search_time:.6f} s")
    print(f"Peak memory (load + search): {peak_mem / (1024 * 1024):.2f} MB")
    
    print("\nFirst 5 matching documents:")
    for doc_id in list(result_doc_ids)[:5]:
        print(f"- {metadata[doc_id].path}")

if __name__ == "__main__":
    main()
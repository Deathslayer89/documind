"""
Retrieval Evaluation Script
Compares different retrieval approaches and selects the best one.
"""

import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
from langchain.schema import Document
import json
from datetime import datetime

# Load environment
load_dotenv()

class RetrievalEvaluator:
    """Evaluates different retrieval approaches."""
    
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        self.rag_pipeline = RAGPipeline(google_api_key)
        
    def evaluate_retrieval_approaches(self) -> Dict:
        """
        Evaluate multiple retrieval approaches:
        1. Semantic search only (current approach)
        2. Different k values (3, 5, 10 results)
        3. Different search types (similarity, mmr)
        """
        
        # Test queries with expected topics
        test_queries = [
            {
                "query": "What is the Floyd-Warshall algorithm?",
                "expected_topic": "algorithms",
                "expected_keywords": ["floyd", "warshall", "shortest", "path", "dynamic"]
            },
            {
                "query": "Explain dynamic programming",
                "expected_topic": "algorithms",
                "expected_keywords": ["dynamic", "programming", "optimization", "subproblem"]
            },
            {
                "query": "How does gradient descent work?",
                "expected_topic": "machine learning",
                "expected_keywords": ["gradient", "descent", "optimization", "loss"]
            },
            {
                "query": "What is object-oriented programming?",
                "expected_topic": "programming",
                "expected_keywords": ["object", "oriented", "class", "inheritance"]
            },
            {
                "query": "What are sorting algorithms?",
                "expected_topic": "algorithms",
                "expected_keywords": ["sort", "algorithm", "complexity", "time"]
            }
        ]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "approaches": []
        }
        
        # Approach 1: Semantic search with k=3
        print("\n" + "="*70)
        print("Approach 1: Semantic Search (k=3)")
        print("="*70)
        approach1_results = self._evaluate_approach(
            test_queries, 
            k=3, 
            search_type="similarity",
            name="Semantic Search (k=3)"
        )
        results["approaches"].append(approach1_results)
        
        # Approach 2: Semantic search with k=5 (current default)
        print("\n" + "="*70)
        print("Approach 2: Semantic Search (k=5) - CURRENT")
        print("="*70)
        approach2_results = self._evaluate_approach(
            test_queries, 
            k=5, 
            search_type="similarity",
            name="Semantic Search (k=5) - CURRENT"
        )
        results["approaches"].append(approach2_results)
        
        # Approach 3: Semantic search with k=10
        print("\n" + "="*70)
        print("Approach 3: Semantic Search (k=10)")
        print("="*70)
        approach3_results = self._evaluate_approach(
            test_queries, 
            k=10, 
            search_type="similarity",
            name="Semantic Search (k=10)"
        )
        results["approaches"].append(approach3_results)
        
        # Approach 4: MMR (Maximal Marginal Relevance) with k=5
        print("\n" + "="*70)
        print("Approach 4: MMR Search (k=5)")
        print("="*70)
        approach4_results = self._evaluate_approach(
            test_queries, 
            k=5, 
            search_type="mmr",
            name="MMR Search (k=5)"
        )
        results["approaches"].append(approach4_results)
        
        return results
    
    def _evaluate_approach(self, test_queries: List[Dict], k: int, search_type: str, name: str) -> Dict:
        """Evaluate a single retrieval approach."""
        
        total_queries = len(test_queries)
        total_relevant = 0
        total_retrieved = 0
        keyword_matches = []
        
        approach_results = {
            "name": name,
            "k": k,
            "search_type": search_type,
            "query_results": []
        }
        
        for i, test_query in enumerate(test_queries, 1):
            print(f"\nQuery {i}/{total_queries}: {test_query['query']}")
            
            # Retrieve documents
            if search_type == "similarity":
                retrieved_docs = self.rag_pipeline.vector_store.similarity_search(
                    test_query['query'], 
                    k=k
                )
            elif search_type == "mmr":
                retrieved_docs = self.rag_pipeline.vector_store.max_marginal_relevance_search(
                    test_query['query'], 
                    k=k,
                    fetch_k=k*2  # Fetch more candidates for diversity
                )
            
            # Evaluate relevance
            relevant_count = 0
            keyword_match_count = 0
            
            for doc in retrieved_docs:
                content_lower = doc.page_content.lower()
                
                # Check if any expected keywords are in the content
                matches = sum(1 for keyword in test_query['expected_keywords'] 
                            if keyword.lower() in content_lower)
                
                if matches > 0:
                    relevant_count += 1
                    keyword_match_count += matches
            
            # Calculate metrics for this query
            precision = relevant_count / len(retrieved_docs) if retrieved_docs else 0
            avg_keyword_matches = keyword_match_count / len(retrieved_docs) if retrieved_docs else 0
            
            query_result = {
                "query": test_query['query'],
                "retrieved_count": len(retrieved_docs),
                "relevant_count": relevant_count,
                "precision": precision,
                "avg_keyword_matches": avg_keyword_matches
            }
            
            approach_results["query_results"].append(query_result)
            
            print(f"  Retrieved: {len(retrieved_docs)} documents")
            print(f"  Relevant: {relevant_count} documents")
            print(f"  Precision: {precision:.2%}")
            print(f"  Avg keyword matches: {avg_keyword_matches:.2f}")
            
            total_relevant += relevant_count
            total_retrieved += len(retrieved_docs)
            keyword_matches.append(avg_keyword_matches)
        
        # Calculate overall metrics
        overall_precision = total_relevant / total_retrieved if total_retrieved > 0 else 0
        avg_keyword_score = sum(keyword_matches) / len(keyword_matches) if keyword_matches else 0
        
        approach_results["overall_metrics"] = {
            "total_queries": total_queries,
            "total_retrieved": total_retrieved,
            "total_relevant": total_relevant,
            "overall_precision": overall_precision,
            "avg_keyword_score": avg_keyword_score
        }
        
        print(f"\n{name} - Overall Metrics:")
        print(f"  Overall Precision: {overall_precision:.2%}")
        print(f"  Avg Keyword Score: {avg_keyword_score:.2f}")
        
        return approach_results
    
    def select_best_approach(self, results: Dict) -> Dict:
        """Select the best retrieval approach based on evaluation metrics."""
        
        best_approach = None
        best_score = -1
        
        print("\n" + "="*70)
        print("COMPARISON OF ALL APPROACHES")
        print("="*70)
        
        for approach in results["approaches"]:
            metrics = approach["overall_metrics"]
            
            # Combined score: 70% precision + 30% keyword relevance
            score = (metrics["overall_precision"] * 0.7) + (metrics["avg_keyword_score"] * 0.3)
            approach["combined_score"] = score
            
            print(f"\n{approach['name']}:")
            print(f"  Precision: {metrics['overall_precision']:.2%}")
            print(f"  Keyword Score: {metrics['avg_keyword_score']:.2f}")
            print(f"  Combined Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_approach = approach
        
        print("\n" + "="*70)
        print(f"BEST APPROACH: {best_approach['name']}")
        print(f"Combined Score: {best_score:.4f}")
        print("="*70)
        
        results["best_approach"] = best_approach["name"]
        results["best_score"] = best_score
        
        return results

def main():
    """Run the retrieval evaluation."""
    
    print("="*70)
    print("RETRIEVAL EVALUATION - Multiple Approaches Comparison")
    print("="*70)
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(api_key)
    
    # Load vector store
    print("\nLoading vector store...")
    evaluator.rag_pipeline.vector_store = evaluator.rag_pipeline.load_vector_store()
    
    if evaluator.rag_pipeline.vector_store is None:
        print("ERROR: Vector store not found. Please initialize the system first.")
        return
    
    print("Vector store loaded successfully!")
    
    # Run evaluation
    print("\nStarting evaluation of multiple retrieval approaches...")
    results = evaluator.evaluate_retrieval_approaches()
    
    # Select best approach
    final_results = evaluator.select_best_approach(results)
    
    # Save results
    output_file = "retrieval_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Generate summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\nEvaluated {len(final_results['approaches'])} different retrieval approaches:")
    for approach in final_results['approaches']:
        print(f"  - {approach['name']}")
    
    print(f"\n✅ Best Approach: {final_results['best_approach']}")
    print(f"✅ Best Score: {final_results['best_score']:.4f}")
    print("\nRecommendation: Use the best approach in production for optimal retrieval quality.")
    print("="*70)

if __name__ == "__main__":
    main()


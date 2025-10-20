"""
LLM Evaluation Script
Compares different prompt templates and LLM configurations to select the best approach.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import json
from datetime import datetime

# Load environment
load_dotenv()

class LLMEvaluator:
    """Evaluates different LLM prompt templates and configurations."""
    
    # Define different prompt templates to test
    PROMPTS = {
        "detailed": {
            "name": "Detailed Context-Based (Current)",
            "template": """Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, just say that you don't know.
Try to be as helpful as possible and provide a detailed answer based on the context.

Context: {context}

Question: {question}

Detailed Answer:"""
        },
        
        "concise": {
            "name": "Concise Direct",
            "template": """Answer the question based on the context below. Be clear and concise.
If the context doesn't contain the answer, say "I don't know based on the provided context."

Context: {context}

Question: {question}

Answer:"""
        },
        
        "structured": {
            "name": "Structured Response",
            "template": """Based on the context provided, answer the question following this structure:
1. Direct answer (2-3 sentences)
2. Key details (if applicable)
3. Additional context (if relevant)

If you cannot answer from the context, state "The provided context does not contain information to answer this question."

Context: {context}

Question: {question}

Structured Answer:"""
        },
        
        "expert": {
            "name": "Expert Technical Style",
            "template": """You are a technical expert assistant. Using the context provided, give a comprehensive technical answer.
Include relevant terminology, concepts, and explanations.
If information is not in the context, explicitly state what you don't know.

Context: {context}

Question: {question}

Expert Answer:"""
        }
    }
    
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        self.rag_pipeline = RAGPipeline(google_api_key)
        
        # Load vector store
        self.rag_pipeline.vector_store = self.rag_pipeline.load_vector_store()
        if self.rag_pipeline.vector_store is None:
            raise ValueError("Vector store not found. Please initialize the system first.")
    
    def evaluate_prompts(self) -> Dict:
        """Evaluate all prompt templates."""
        
        # Test questions with expected characteristics
        test_queries = [
            {
                "query": "What is the Floyd-Warshall algorithm?",
                "expected_qualities": ["algorithm_name", "purpose", "complexity"],
                "topic": "algorithms"
            },
            {
                "query": "Explain dynamic programming",
                "expected_qualities": ["definition", "approach", "examples"],
                "topic": "algorithms"
            },
            {
                "query": "How does gradient descent work?",
                "expected_qualities": ["process", "optimization", "mathematical"],
                "topic": "machine_learning"
            },
            {
                "query": "What is cryptography?",
                "expected_qualities": ["definition", "purpose", "techniques"],
                "topic": "cryptography"
            },
            {
                "query": "What is reinforcement learning?",
                "expected_qualities": ["honest_unknown"],  # Not in our dataset
                "topic": "not_in_dataset"
            }
        ]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "prompt_evaluations": []
        }
        
        for prompt_key, prompt_config in self.PROMPTS.items():
            print(f"\n{'='*70}")
            print(f"Evaluating: {prompt_config['name']}")
            print(f"{'='*70}")
            
            prompt_results = self._evaluate_single_prompt(
                prompt_key,
                prompt_config,
                test_queries
            )
            results["prompt_evaluations"].append(prompt_results)
        
        return results
    
    def _evaluate_single_prompt(self, prompt_key: str, prompt_config: Dict, test_queries: List[Dict]) -> Dict:
        """Evaluate a single prompt template."""
        
        # Create QA chain with this prompt
        retriever = self.rag_pipeline.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_config["template"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.rag_pipeline.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        
        prompt_results = {
            "prompt_key": prompt_key,
            "prompt_name": prompt_config["name"],
            "query_results": []
        }
        
        total_score = 0
        
        for i, test_query in enumerate(test_queries, 1):
            print(f"\nQuery {i}/{len(test_queries)}: {test_query['query']}")
            
            # Get answer
            result = qa_chain({"query": test_query['query']})
            answer = result.get('result', result.get('answer', ''))
            
            # Evaluate answer quality
            score = self._score_answer(answer, test_query)
            total_score += score
            
            # Calculate answer metrics
            word_count = len(answer.split())
            has_unknown_admission = any(phrase in answer.lower() for phrase in [
                "don't know", "do not know", "cannot", "not contain", "don't have"
            ])
            
            query_result = {
                "query": test_query['query'],
                "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                "word_count": word_count,
                "has_unknown_admission": has_unknown_admission,
                "quality_score": score,
                "sources_count": len(result.get('source_documents', []))
            }
            
            prompt_results["query_results"].append(query_result)
            
            print(f"  Word Count: {word_count}")
            print(f"  Unknown Admission: {'Yes' if has_unknown_admission else 'No'}")
            print(f"  Quality Score: {score:.2f}/10")
            print(f"  Sources: {len(result.get('source_documents', []))}")
        
        # Calculate overall metrics
        avg_score = total_score / len(test_queries)
        avg_word_count = sum(qr['word_count'] for qr in prompt_results['query_results']) / len(test_queries)
        
        prompt_results["overall_metrics"] = {
            "average_quality_score": avg_score,
            "average_word_count": avg_word_count,
            "total_queries": len(test_queries)
        }
        
        print(f"\n{prompt_config['name']} - Overall Metrics:")
        print(f"  Average Quality Score: {avg_score:.2f}/10")
        print(f"  Average Word Count: {avg_word_count:.1f}")
        
        return prompt_results
    
    def _score_answer(self, answer: str, test_query: Dict) -> float:
        """Score an answer based on quality criteria (0-10 scale)."""
        
        score = 5.0  # Base score
        answer_lower = answer.lower()
        
        # Check for expected qualities
        if test_query['topic'] == "not_in_dataset":
            # Should honestly admit not knowing
            if any(phrase in answer_lower for phrase in ["don't know", "cannot", "not contain"]):
                score += 5.0  # Perfect for honesty
            else:
                score -= 3.0  # Hallucination penalty
        else:
            # Check for expected content
            if "algorithm" in test_query['expected_qualities']:
                if any(word in answer_lower for word in ["algorithm", "complexity", "time"]):
                    score += 2.0
            
            if "definition" in test_query['expected_qualities']:
                if len(answer.split()) > 30:  # Sufficient detail
                    score += 1.5
            
            if "examples" in test_query['expected_qualities']:
                if any(word in answer_lower for word in ["example", "such as", "like", "for instance"]):
                    score += 1.0
            
            # Check for structure and clarity
            if len(answer.split('.')) >= 3:  # Multiple sentences
                score += 1.0
            
            # Penalty for too short answers (likely incomplete)
            if len(answer.split()) < 20 and test_query['topic'] != "not_in_dataset":
                score -= 2.0
            
            # Bonus for detailed technical content
            if len(answer.split()) > 100:
                score += 0.5
        
        return max(0, min(10, score))  # Clamp between 0-10
    
    def select_best_prompt(self, results: Dict) -> Dict:
        """Select the best prompt template based on evaluation."""
        
        print(f"\n{'='*70}")
        print("COMPARISON OF ALL PROMPTS")
        print(f"{'='*70}")
        
        best_prompt = None
        best_score = -1
        
        for prompt_eval in results["prompt_evaluations"]:
            metrics = prompt_eval["overall_metrics"]
            score = metrics["average_quality_score"]
            
            print(f"\n{prompt_eval['prompt_name']}:")
            print(f"  Average Quality Score: {score:.2f}/10")
            print(f"  Average Word Count: {metrics['average_word_count']:.1f}")
            
            if score > best_score:
                best_score = score
                best_prompt = prompt_eval
        
        print(f"\n{'='*70}")
        print(f"BEST PROMPT: {best_prompt['prompt_name']}")
        print(f"Quality Score: {best_score:.2f}/10")
        print(f"{'='*70}")
        
        results["best_prompt"] = best_prompt["prompt_name"]
        results["best_prompt_key"] = best_prompt["prompt_key"]
        results["best_score"] = best_score
        
        return results

def main():
    """Run the LLM evaluation."""
    
    print("="*70)
    print("LLM EVALUATION - Multiple Prompt Templates Comparison")
    print("="*70)
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    try:
        evaluator = LLMEvaluator(api_key)
        print("✓ Vector store loaded successfully!")
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    
    # Run evaluation
    print("\nStarting evaluation of multiple prompt templates...")
    results = evaluator.evaluate_prompts()
    
    # Select best prompt
    final_results = evaluator.select_best_prompt(results)
    
    # Save results
    output_file = "llm_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Generate summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"\nEvaluated {len(final_results['prompt_evaluations'])} different prompt templates:")
    for prompt_eval in final_results['prompt_evaluations']:
        print(f"  - {prompt_eval['prompt_name']}")
    
    print(f"\n✅ Best Prompt: {final_results['best_prompt']}")
    print(f"✅ Best Score: {final_results['best_score']:.2f}/10")
    print("\nRecommendation: Use the best prompt template in production for optimal answer quality.")
    print("="*70)

if __name__ == "__main__":
    main()


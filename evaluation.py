#!/usr/bin/env python3
"""
Evaluation script for the RAG system.
Provides comprehensive evaluation metrics and analysis.
"""

import os
import json
import time
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor

class RAGEvaluator:
    def __init__(self, google_api_key: str):
        self.api_key = google_api_key
        self.rag_pipeline = RAGPipeline(google_api_key)
        self.test_questions = []
        self.test_results = []

    def load_test_questions(self, questions_file: str = None):
        """Load test questions from a file or use default questions."""
        if questions_file and os.path.exists(questions_file):
            with open(questions_file, 'r') as f:
                self.test_questions = json.load(f)
        else:
            # Default test questions for CS textbooks
            self.test_questions = [
                {
                    "question": "What is an algorithm?",
                    "expected_keywords": ["step-by-step", "procedure", "instructions", "solve", "problem"],
                    "category": "definitions"
                },
                {
                    "question": "What are the main properties of good algorithms?",
                    "expected_keywords": ["correctness", "efficiency", "scalability", "performance"],
                    "category": "properties"
                },
                {
                    "question": "What is Big O notation?",
                    "expected_keywords": ["complexity", "time", "space", "asymptotic", "growth"],
                    "category": "complexity"
                },
                {
                    "question": "What are common data structures in computer science?",
                    "expected_keywords": ["arrays", "linked lists", "stacks", "queues", "trees", "graphs"],
                    "category": "data_structures"
                },
                {
                    "question": "What is time complexity?",
                    "expected_keywords": ["runtime", "input size", "growth", "performance"],
                    "category": "complexity"
                },
                {
                    "question": "How do you analyze algorithm efficiency?",
                    "expected_keywords": ["benchmarking", "profiling", "complexity analysis", "measurement"],
                    "category": "analysis"
                },
                {
                    "question": "What are the main programming paradigms?",
                    "expected_keywords": ["imperative", "declarative", "object-oriented", "functional"],
                    "category": "paradigms"
                },
                {
                    "question": "What is computer architecture about?",
                    "expected_keywords": ["CPU", "memory", "storage", "hardware", "organization"],
                    "category": "architecture"
                }
            ]

    def initialize_system(self, force_recreate: bool = False):
        """Initialize the RAG system."""
        return self.rag_pipeline.initialize_vector_store(force_recreate=force_recreate)

    def keyword_relevance_score(self, answer: str, keywords: List[str]) -> float:
        """Calculate keyword relevance score for an answer."""
        if not keywords:
            return 0.0

        answer_lower = answer.lower()
        keywords_found = sum(1 for keyword in keywords if keyword.lower() in answer_lower)
        return keywords_found / len(keywords)

    def answer_quality_score(self, answer: str) -> Dict[str, float]:
        """Calculate basic answer quality metrics."""
        metrics = {
            "length_score": min(len(answer) / 200, 1.0),  # Prefer longer answers (up to 200 chars)
            "has_explanation": 1.0 if any(expl in answer.lower() for expl in ["because", "therefore", "since", "due to"]) else 0.5,
            "has_examples": 1.0 if any(ex in answer.lower() for ex in ["example", "for instance", "such as"]) else 0.5,
            "completeness": min(len(answer.split()) / 50, 1.0)  # Prefer more detailed answers
        }
        return metrics

    def evaluate_single_question(self, question_data: Dict) -> Dict[str, Any]:
        """Evaluate a single question."""
        question = question_data["question"]
        expected_keywords = question_data.get("expected_keywords", [])
        category = question_data.get("category", "general")

        print(f"Evaluating: {question}")

        start_time = time.time()
        try:
            result = self.rag_pipeline.query(question)
            response_time = time.time() - start_time

            answer = result.get("answer", "")
            num_sources = result.get("num_sources", 0)
            has_sources = num_sources > 0

            # Calculate scores
            keyword_score = self.keyword_relevance_score(answer, expected_keywords)
            quality_metrics = self.answer_quality_score(answer)
            overall_quality = sum(quality_metrics.values()) / len(quality_metrics)

            evaluation = {
                "question": question,
                "answer": answer,
                "category": category,
                "response_time": response_time,
                "num_sources": num_sources,
                "has_sources": has_sources,
                "keyword_relevance": keyword_score,
                "answer_quality": quality_metrics,
                "overall_quality": overall_quality,
                "expected_keywords": expected_keywords,
                "success": not result.get("error"),
                "error": result.get("error")
            }

        except Exception as e:
            evaluation = {
                "question": question,
                "answer": "",
                "category": category,
                "response_time": time.time() - start_time,
                "num_sources": 0,
                "has_sources": False,
                "keyword_relevance": 0.0,
                "answer_quality": {},
                "overall_quality": 0.0,
                "expected_keywords": expected_keywords,
                "success": False,
                "error": str(e)
            }

        return evaluation

    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation."""
        print("Starting RAG system evaluation...")

        if not self.test_questions:
            print("No test questions loaded!")
            return {"error": "No test questions available"}

        results = []
        for question_data in self.test_questions:
            result = self.evaluate_single_question(question_data)
            results.append(result)

        self.test_results = results

        # Calculate overall metrics
        total_questions = len(results)
        successful_queries = sum(1 for r in results if r["success"])
        success_rate = successful_queries / total_questions

        avg_response_time = sum(r["response_time"] for r in results) / total_questions
        avg_sources = sum(r["num_sources"] for r in results) / total_questions
        avg_keyword_relevance = sum(r["keyword_relevance"] for r in results) / total_questions
        avg_overall_quality = sum(r["overall_quality"] for r in results) / total_questions

        # Category breakdown
        category_stats = {}
        for result in results:
            cat = result["category"]
            if cat not in category_stats:
                category_stats[cat] = {
                    "count": 0,
                    "success": 0,
                    "avg_quality": 0,
                    "avg_sources": 0
                }
            category_stats[cat]["count"] += 1
            category_stats[cat]["success"] += 1 if result["success"] else 0
            category_stats[cat]["avg_quality"] += result["overall_quality"]
            category_stats[cat]["avg_sources"] += result["num_sources"]

        for cat in category_stats:
            stats = category_stats[cat]
            stats["success_rate"] = stats["success"] / stats["count"]
            stats["avg_quality"] /= stats["count"]
            stats["avg_sources"] /= stats["count"]

        evaluation_summary = {
            "total_questions": total_questions,
            "successful_queries": successful_queries,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "avg_sources_retrieved": avg_sources,
            "avg_keyword_relevance": avg_keyword_relevance,
            "avg_overall_quality": avg_overall_quality,
            "category_breakdown": category_stats,
            "detailed_results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        return evaluation_summary

    def save_results(self, results: Dict[str, Any], filename: str = "evaluation_results.json"):
        """Save evaluation results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {filename}")

    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted evaluation summary."""
        print("\n" + "="*60)
        print("RAG SYSTEM EVALUATION SUMMARY")
        print("="*60)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Successful Queries: {results['successful_queries']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Average Response Time: {results['avg_response_time']:.2f}s")
        print(f"Average Sources Retrieved: {results['avg_sources_retrieved']:.1f}")
        print(f"Average Keyword Relevance: {results['avg_keyword_relevance']:.1%}")
        print(f"Average Overall Quality: {results['avg_overall_quality']:.1%}")

        print("\n" + "-"*40)
        print("CATEGORY BREAKDOWN")
        print("-"*40)
        for category, stats in results['category_breakdown'].items():
            print(f"{category.title()}:")
            print(f"  Questions: {stats['count']}")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
            print(f"  Avg Quality: {stats['avg_quality']:.1%}")
            print(f"  Avg Sources: {stats['avg_sources']:.1f}")
            print()

        print("-"*40)
        print("DETAILED RESULTS")
        print("-"*40)
        for i, result in enumerate(results['detailed_results']):
            print(f"\n{i+1}. Q: {result['question']}")
            print(f"   Success: {result['success']}")
            print(f"   Sources: {result['num_sources']}")
            print(f"   Quality: {result['overall_quality']:.1%}")
            print(f"   Response Time: {result['response_time']:.2f}s")
            if result['error']:
                print(f"   Error: {result['error']}")
            else:
                print(f"   Answer: {result['answer'][:100]}...")

def main():
    """Main evaluation script."""
    # Load environment variables
    load_dotenv()

    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Please set GOOGLE_API_KEY in your environment variables")
        return

    print("🚀 Starting RAG system evaluation...")

    # Initialize evaluator
    evaluator = RAGEvaluator(api_key)

    # Initialize the system
    if not evaluator.initialize_system():
        print("❌ Failed to initialize RAG system")
        return

    print("✅ RAG system initialized successfully")

    # Load test questions
    evaluator.load_test_questions()
    print(f"✅ Loaded {len(evaluator.test_questions)} test questions")

    # Run evaluation
    results = evaluator.run_evaluation()

    # Save results
    evaluator.save_results(results)

    # Print summary
    evaluator.print_summary()

if __name__ == "__main__":
    main()
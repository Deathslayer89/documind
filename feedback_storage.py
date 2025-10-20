"""
Feedback Storage System
Stores user feedback and interaction metrics for monitoring.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

class FeedbackStorage:
    """Manages storage and retrieval of user feedback."""
    
    def __init__(self, feedback_file: str = "feedback_data.json"):
        self.feedback_file = feedback_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure the feedback file exists."""
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                json.dump({"feedback": [], "interactions": []}, f)
    
    def add_feedback(self, 
                    question: str, 
                    answer: str, 
                    feedback_type: str,  # 'positive' or 'negative'
                    sources_count: int,
                    comment: Optional[str] = None):
        """Add user feedback for a question-answer pair."""
        
        data = self._load_data()
        
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
            "feedback": feedback_type,
            "sources_count": sources_count,
            "comment": comment,
            "answer_length": len(answer.split())
        }
        
        data["feedback"].append(feedback_entry)
        self._save_data(data)
    
    def add_interaction(self,
                       question: str,
                       answer_length: int,
                       sources_count: int,
                       response_time: float):
        """Log a user interaction (query)."""
        
        data = self._load_data()
        
        interaction_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "question_length": len(question.split()),
            "answer_length": answer_length,
            "sources_count": sources_count,
            "response_time_seconds": response_time
        }
        
        data["interactions"].append(interaction_entry)
        self._save_data(data)
    
    def get_all_feedback(self) -> List[Dict]:
        """Get all feedback entries."""
        data = self._load_data()
        return data.get("feedback", [])
    
    def get_all_interactions(self) -> List[Dict]:
        """Get all interaction logs."""
        data = self._load_data()
        return data.get("interactions", [])
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics about feedback."""
        feedback = self.get_all_feedback()
        
        if not feedback:
            return {
                "total_feedback": 0,
                "positive_count": 0,
                "negative_count": 0,
                "positive_percentage": 0,
                "negative_percentage": 0
            }
        
        positive = sum(1 for f in feedback if f['feedback'] == 'positive')
        negative = sum(1 for f in feedback if f['feedback'] == 'negative')
        total = len(feedback)
        
        return {
            "total_feedback": total,
            "positive_count": positive,
            "negative_count": negative,
            "positive_percentage": (positive / total * 100) if total > 0 else 0,
            "negative_percentage": (negative / total * 100) if total > 0 else 0
        }
    
    def get_interaction_stats(self) -> Dict:
        """Get statistics about interactions."""
        interactions = self.get_all_interactions()
        
        if not interactions:
            return {
                "total_queries": 0,
                "avg_response_time": 0,
                "avg_answer_length": 0,
                "avg_sources_count": 0
            }
        
        total = len(interactions)
        
        return {
            "total_queries": total,
            "avg_response_time": sum(i['response_time_seconds'] for i in interactions) / total,
            "avg_answer_length": sum(i['answer_length'] for i in interactions) / total,
            "avg_sources_count": sum(i['sources_count'] for i in interactions) / total,
            "max_response_time": max(i['response_time_seconds'] for i in interactions),
            "min_response_time": min(i['response_time_seconds'] for i in interactions)
        }
    
    def get_feedback_dataframe(self) -> pd.DataFrame:
        """Get feedback as a pandas DataFrame."""
        feedback = self.get_all_feedback()
        if not feedback:
            return pd.DataFrame()
        return pd.DataFrame(feedback)
    
    def get_interactions_dataframe(self) -> pd.DataFrame:
        """Get interactions as a pandas DataFrame."""
        interactions = self.get_all_interactions()
        if not interactions:
            return pd.DataFrame()
        return pd.DataFrame(interactions)
    
    def _load_data(self) -> Dict:
        """Load data from file."""
        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"feedback": [], "interactions": []}
    
    def _save_data(self, data: Dict):
        """Save data to file."""
        with open(self.feedback_file, 'w') as f:
            json.dump(data, f, indent=2)


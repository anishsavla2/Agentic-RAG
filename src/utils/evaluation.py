import time
from typing import Dict
import psutil

class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'avg_response_time': 0,
            'total_chunks_retrieved': 0
        }
    
    def update_metrics(self, query_metrics: Dict):
        """Update running metrics"""
        self.metrics['total_queries'] += 1
        self.metrics['avg_response_time'] = (
            (self.metrics['avg_response_time'] * (self.metrics['total_queries'] - 1) +
             query_metrics['processing_time']) / self.metrics['total_queries']
        )
        self.metrics['total_chunks_retrieved'] += query_metrics['num_chunks']
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        return {
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'cpu_percent': psutil.cpu_percent()
        }

"""
Base crawler class for app store review collection
"""
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseCrawler(ABC):
    """Abstract base class for app store crawlers"""
    
    def __init__(self, app_id: str):
        self.app_id = app_id
    
    @abstractmethod
    def collect_reviews(self, count: int = 100) -> List[Dict]:
        """
        Collect reviews from the app store
        
        Args:
            count: Number of reviews to collect
            
        Returns:
            List of review dictionaries with standardized format:
            {
                'userName': str,
                'content': str, 
                'score': int,
                'at': datetime,
                'platform': str
            }
        """
        pass
    
    @abstractmethod
    def get_platform_name(self) -> str:
        """Get the platform name"""
        pass

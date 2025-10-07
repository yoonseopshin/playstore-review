"""
Google Play Store review crawler
"""
from datetime import datetime
from typing import List, Dict

from google_play_scraper import reviews, Sort

from .base import BaseCrawler


class PlayStoreCrawler(BaseCrawler):
    """Google Play Store review crawler"""
    
    def __init__(self, app_package: str):
        super().__init__(app_package)
        self.app_package = app_package
    
    def collect_reviews(self, count: int = 100) -> List[Dict]:
        """
        Collect reviews from Google Play Store
        
        Args:
            count: Number of reviews to collect
            
        Returns:
            List of review dictionaries
        """
        print(f"Fetching reviews from Google Play Store...")
        
        try:
            result, _ = reviews(
                self.app_package,
                lang='ko',
                country='kr',
                sort=Sort.NEWEST,
                count=count
            )
            
            # Standardize format
            standardized_reviews = []
            for review in result:
                standardized_reviews.append({
                    'userName': review.get('userName', 'Anonymous'),
                    'content': review.get('content', ''),
                    'score': review.get('score', 0),
                    'at': review.get('at', datetime.now()),
                    'platform': self.get_platform_name()
                })
            
            print(f"✓ Fetched {len(standardized_reviews)} reviews from Play Store")
            return standardized_reviews
            
        except Exception as e:
            print(f"❌ Error fetching Play Store reviews: {e}")
            return []
    
    def get_platform_name(self) -> str:
        """Get the platform name"""
        return "Google Play Store"

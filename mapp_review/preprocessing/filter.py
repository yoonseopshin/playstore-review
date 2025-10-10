"""
Data filtering and processing module
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class ReviewFilter:
    """Filter and process review data"""
    
    def __init__(self, days: int = 7):
        self.days = days
        self.end_date = (datetime.now() - timedelta(days=1)).date()
        self.start_date = (self.end_date - timedelta(days=days-1))
    
    def process_reviews(self, review_data: List[Dict]) -> Optional[pd.DataFrame]:
        """
        Process and filter review data from multiple platforms
        
        Args:
            review_data: Raw review data from both platforms
            
        Returns:
            Processed DataFrame with platform information
        """
        if not review_data:
            print("No review data to process.")
            return None
            
        # Create DataFrame and clean up
        df = pd.DataFrame(review_data)
        
        # Ensure we have the required columns
        required_columns = ['userName', 'content', 'score', 'at', 'platform']
        optional_columns = ['version']  # Optional columns that should be preserved if present
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return None
        
        # Include optional columns if they exist
        columns_to_keep = required_columns.copy()
        for col in optional_columns:
            if col in df.columns:
                columns_to_keep.append(col)
        
        df = df[columns_to_keep]
        df.rename(columns={'content': 'review'}, inplace=True)
        df['at'] = pd.to_datetime(df['at'])
        
        # Filter for date range
        df = df[(df['at'].dt.date >= self.start_date) & (df['at'].dt.date <= self.end_date)]
        df.sort_values('at', inplace=True)
        
        if df.empty:
            print(f"No reviews found for the last {self.days} days.")
            return None
        
        # Print platform breakdown
        platform_counts = df['platform'].value_counts()
        print(f"\n✓ Processed {len(df)} reviews within date range:")
        for platform, count in platform_counts.items():
            print(f"   • {platform}: {count} reviews")
            
        return df
    
    def get_date_range(self):
        """Get the date range for filtering"""
        return self.start_date, self.end_date

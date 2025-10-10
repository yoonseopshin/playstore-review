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
        Process and filter review data from multiple platforms with enhanced debugging
        
        Args:
            review_data: Raw review data from both platforms
            
        Returns:
            Processed DataFrame with platform information
        """
        print(f"\n🔍 === REVIEW FILTER DEBUG INFO ===")
        print(f"📅 Filter Date Range: {self.start_date} to {self.end_date}")
        
        if not review_data:
            print("❌ No review data to process.")
            return None
        
        print(f"📦 Raw review data count: {len(review_data)}")
        
        # Debug raw data by platform
        platform_raw_counts = {}
        for review in review_data:
            platform = review.get('platform', 'Unknown')
            platform_raw_counts[platform] = platform_raw_counts.get(platform, 0) + 1
        
        print(f"📊 Raw data by platform:")
        for platform, count in platform_raw_counts.items():
            print(f"   • {platform}: {count} reviews")
            
        # Create DataFrame and clean up
        df = pd.DataFrame(review_data)
        print(f"📋 DataFrame columns: {list(df.columns)}")
        
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
        
        print(f"📅 Date conversion completed. Sample dates:")
        if len(df) > 0:
            for i, date in enumerate(df['at'].head(3)):
                print(f"   • Review {i+1}: {date} ({date.date()})")
        
        # Show date range analysis before filtering
        if len(df) > 0:
            min_date = df['at'].min()
            max_date = df['at'].max()
            print(f"📊 All reviews date range: {min_date.date()} to {max_date.date()}")
        
        # Filter for date range
        before_filter_count = len(df)
        df = df[(df['at'].dt.date >= self.start_date) & (df['at'].dt.date <= self.end_date)]
        after_filter_count = len(df)
        
        print(f"🔍 Date filtering results:")
        print(f"   • Before filter: {before_filter_count} reviews")
        print(f"   • After filter: {after_filter_count} reviews")
        print(f"   • Filtered out: {before_filter_count - after_filter_count} reviews")
        
        df.sort_values('at', inplace=True)
        
        if df.empty:
            print(f"❌ No reviews found for the last {self.days} days ({self.start_date} to {self.end_date}).")
            print(f"💡 Suggestions:")
            print(f"   • Check if the date range is correct")
            print(f"   • Verify review dates from the source")
            print(f"   • Consider increasing the analysis days")
            return None
        
        # Print platform breakdown
        platform_counts = df['platform'].value_counts()
        print(f"\n✅ Final processed {len(df)} reviews within date range:")
        for platform, count in platform_counts.items():
            print(f"   • {platform}: {count} reviews")
        
        # Show sample of filtered data
        print(f"\n🔍 Sample filtered reviews:")
        for i, row in df.head(3).iterrows():
            print(f"   {i+1}. {row['at'].date()} - {row['platform']} - Score: {row['score']}")
        
        print(f"🔍 === END REVIEW FILTER DEBUG ===")
        return df
    
    def get_date_range(self):
        """Get the date range for filtering"""
        return self.start_date, self.end_date

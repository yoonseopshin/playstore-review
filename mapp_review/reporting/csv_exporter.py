"""
CSV export functionality
"""
import pandas as pd
import os
from datetime import datetime


class CSVExporter:
    """Export review data to CSV format"""
    
    def save_csv(self, df: pd.DataFrame, output_dir: str, 
                 start_date: datetime.date, end_date: datetime.date, 
                 suffix: str = "") -> str:
        """Save DataFrame to CSV file"""
        
        filename = f"user_reviews_summary{suffix}_{start_date}_{end_date}.csv"
        csv_file = os.path.join(output_dir, filename)
        
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"CSV saved successfully: {csv_file} (Total {len(df)} reviews)")
        return csv_file

"""
HTML report generation
"""
import pandas as pd
import os
from datetime import datetime
from typing import Dict
from jinja2 import Environment, FileSystemLoader


class HTMLReportGenerator:
    """Generate HTML reports using Jinja2 templates"""
    
    def __init__(self, template_dir: str = None):
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
        
        self.env = Environment(loader=FileSystemLoader(template_dir))
    
    def generate_html_report(self, df: pd.DataFrame, output_dir: str, start_date: datetime.date, 
                            end_date: datetime.date, summary_img: str, wc_img: str, 
                            topic_img: str = "", topic_summary: Dict = None) -> str:
        """Generate responsive HTML report using Jinja2 templates"""
        
        try:
            template = self.env.get_template('report.html')
            
            # Prepare data for template
            columns_to_drop = ['date']
            if 'topic' in df.columns:
                columns_to_drop.append('topic')
                
            df_for_html = df.drop(columns=[col for col in columns_to_drop if col in df.columns]).copy()
            df_for_html['at'] = df_for_html['at'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Convert DataFrame to list of dictionaries for Jinja2
            reviews_data = df_for_html.to_dict('records')
            
            # Calculate overall statistics
            avg_rating = df['score'].mean()
            total_reviews = len(df)
            
            # Calculate platform-specific statistics
            platform_stats = {}
            if 'platform' in df.columns:
                for platform in df['platform'].unique():
                    platform_df = df[df['platform'] == platform]
                    platform_stats[platform] = {
                        'count': len(platform_df),
                        'avg_rating': platform_df['score'].mean()
                    }
            
            # Template context
            context = {
                'start_date': start_date,
                'end_date': end_date,
                'summary_img': os.path.basename(summary_img),
                'wordcloud_img': os.path.basename(wc_img),
                'reviews': reviews_data,
                'total_reviews': total_reviews,
                'avg_rating': avg_rating,
                'days': (end_date - start_date).days + 1,
                'has_topics': 'topic_label' in df.columns,
                'topic_summary': topic_summary or {},
                'topic_img': os.path.basename(topic_img) if topic_img else "",
                'platform_stats': platform_stats
            }
            
            # Render template
            html_content = template.render(context)
            
            # Save HTML file
            html_file = os.path.join(output_dir, f"mapp_report_{start_date}_{end_date}.html")
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            print(f"HTML report generated successfully: {html_file}")
            return html_file
            
        except Exception as e:
            print(f"Error generating HTML report: {str(e)}")
            raise

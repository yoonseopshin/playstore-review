#!/usr/bin/env python3
"""
PlayStore Review Analysis Script

This script collects Google Play Store reviews for a specified app,
analyzes them, and generates visualization reports including:
- CSV data export
- Time series charts
- Word clouds
- Interactive HTML reports
"""

from google_play_scraper import reviews, Sort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from datetime import datetime, timedelta
import os
import platform
from typing import Tuple, Optional
from jinja2 import Environment, FileSystemLoader


# ===============================
# Configuration
# ===============================
class Config:
    """Configuration settings for the review analysis"""
    APP_PACKAGE = 'com.kakao.yellowid'
    REVIEW_COUNT = 1000
    DAYS = 7
    OUTPUT_DIR = "./output"
    
    # Font settings by OS
    FONT_SETTINGS = {
        'darwin': {  # macOS
            'name': "AppleGothic",
            'path': "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
        },
        'linux': {
            'name': "NanumGothic", 
            'path': "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        }
    }


def setup_environment() -> Tuple[str, str, str]:
    """
    Set up the environment and return font configuration
    
    Returns:
        Tuple of (font_name, font_path, output_dir)
    """
    # Detect OS and set font
    current_os = platform.system().lower()
    font_config = Config.FONT_SETTINGS.get(current_os, Config.FONT_SETTINGS['linux'])
    
    # Create output directory
    print(f"Current working directory: {os.getcwd()}")
    print(f"Creating output directory: {Config.OUTPUT_DIR}")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Verify directory creation
    if os.path.exists(Config.OUTPUT_DIR):
        print(f"✓ Output directory created successfully: {os.path.abspath(Config.OUTPUT_DIR)}")
    else:
        print(f"✗ Failed to create output directory: {Config.OUTPUT_DIR}")
        raise RuntimeError(f"Failed to create output directory: {Config.OUTPUT_DIR}")
    
    # Create .nojekyll file for GitHub Pages
    nojekyll_path = os.path.join(Config.OUTPUT_DIR, ".nojekyll")
    with open(nojekyll_path, "w") as f:
        f.write("")
    print(f"✓ .nojekyll file created: {nojekyll_path}")
    
    return font_config['name'], font_config['path'], Config.OUTPUT_DIR


def get_date_range() -> Tuple[datetime.date, datetime.date]:
    """
    Calculate the date range for review collection
    
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.today().date() - timedelta(days=1)  # Based on yesterday
    start_date = end_date - timedelta(days=Config.DAYS - 1)
    
    print(f"Collecting reviews from the last {Config.DAYS} days ({start_date} ~ {end_date})...")
    return start_date, end_date


def collect_reviews() -> list:
    """
    Collect reviews from Google Play Store
    
    Returns:
        List of review data
    """
    print("Fetching reviews from Google Play Store...")
    result, _ = reviews(
        Config.APP_PACKAGE,
        lang='ko',
        country='kr',
        count=Config.REVIEW_COUNT,
        sort=Sort.NEWEST
    )
    print(f"✓ Fetched {len(result)} reviews from Play Store")
    return result


def process_reviews(review_data: list, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Process and filter review data
    
    Args:
        review_data: Raw review data from Play Store
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Processed DataFrame
    """
    # Create DataFrame and clean up
    df = pd.DataFrame(review_data)
    df = df[['userName', 'content', 'score', 'at']]
    df.rename(columns={'content': 'review'}, inplace=True)
    df['at'] = pd.to_datetime(df['at'])
    
    # Filter for date range
    df = df[(df['at'].dt.date >= start_date) & (df['at'].dt.date <= end_date)]
    df.sort_values('at', inplace=True)
    
    if df.empty:
        print(f"No reviews found for the last {Config.DAYS} days.")
        return None
        
    print(f"✓ Processed {len(df)} reviews within date range")
    return df


def save_csv(df: pd.DataFrame, output_dir: str, start_date: datetime.date, end_date: datetime.date) -> str:
    """
    Save DataFrame to CSV file
    
    Args:
        df: DataFrame to save
        output_dir: Output directory path
        start_date: Start date for filename
        end_date: End date for filename
        
    Returns:
        Path to saved CSV file
    """
    csv_file = os.path.join(output_dir, f"user_reviews_summary_{start_date}_{end_date}.csv")
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"CSV saved successfully: {csv_file} (Total {len(df)} reviews)")
    return csv_file


def create_summary_chart(df: pd.DataFrame, font_name: str, output_dir: str, 
                        start_date: datetime.date, end_date: datetime.date) -> str:
    """
    Create time series summary chart
    
    Args:
        df: DataFrame with review data
        font_name: Font name for chart
        output_dir: Output directory path
        start_date: Start date for title
        end_date: End date for title
        
    Returns:
        Path to saved chart image
    """
    # Prepare data
    df['date'] = df['at'].dt.date
    daily_score = df.groupby('date')['score'].mean().reset_index()
    daily_count = df.groupby('date').size().reset_index(name='count')
    
    # Create chart
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    plt.plot(daily_score['date'], daily_score['score'], marker='o', label='평균 별점')
    plt.bar(daily_count['date'], daily_count['count'], alpha=0.3, label='리뷰 수', color='skyblue')
    
    plt.xlabel("날짜", fontname=font_name)
    plt.ylabel("평균 별점 / 리뷰 수", fontname=font_name)
    plt.title(f"최근 {Config.DAYS}일 리뷰 요약 ({start_date} ~ {end_date})", fontname=font_name)
    plt.legend(prop={"family": font_name})
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save chart
    summary_img = os.path.join(output_dir, f"review_summary_{start_date}_{end_date}.png")
    plt.savefig(summary_img)
    plt.close()
    print(f"Review summary visualization image saved: {summary_img}")
    return summary_img


def create_wordcloud(df: pd.DataFrame, font_path: str, output_dir: str,
                    start_date: datetime.date, end_date: datetime.date) -> str:
    """
    Create word cloud from review text
    
    Args:
        df: DataFrame with review data
        font_path: Font path for word cloud
        output_dir: Output directory path
        start_date: Start date for filename
        end_date: End date for filename
        
    Returns:
        Path to saved word cloud image
    """
    # Prepare text
    text = " ".join(df['review'].astype(str))
    text = re.sub(r"[^가-힣a-zA-Z\s]", " ", text)
    
    # Generate word cloud
    wordcloud = WordCloud(
        font_path=font_path,
        width=1200,
        height=600,
        background_color='white',
        max_words=500
    ).generate(text)
    
    # Save word cloud
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    
    wc_img = os.path.join(output_dir, f"review_wordcloud_{start_date}_{end_date}.png")
    plt.savefig(wc_img)
    plt.close()
    print(f"Word cloud image saved: {wc_img}")
    return wc_img


def setup_jinja_environment() -> Environment:
    """
    Set up Jinja2 environment for template rendering
    
    Returns:
        Jinja2 Environment instance
    """
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    env = Environment(loader=FileSystemLoader(template_dir))
    return env


def generate_html_report(df: pd.DataFrame, output_dir: str, start_date: datetime.date, 
                        end_date: datetime.date, summary_img: str, wc_img: str) -> str:
    """
    Generate responsive HTML report using Jinja2 templates
    
    Args:
        df: DataFrame with review data
        output_dir: Output directory path
        start_date: Start date for report
        end_date: End date for report
        summary_img: Path to summary image
        wc_img: Path to word cloud image
        
    Returns:
        Path to generated HTML file
    """
    try:
        # Setup Jinja2 environment
        env = setup_jinja_environment()
        template = env.get_template('report.html')
        
        # Prepare data for template
        df_for_html = df.drop(columns=['date']).copy()
        df_for_html['at'] = df_for_html['at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert DataFrame to list of dictionaries for Jinja2
        reviews_data = df_for_html.to_dict('records')
        
        # Calculate statistics
        avg_rating = df['score'].mean()
        total_reviews = len(df)
        
        # Template context
        context = {
            'start_date': start_date,
            'end_date': end_date,
            'summary_img': os.path.basename(summary_img),
            'wordcloud_img': os.path.basename(wc_img),
            'reviews': reviews_data,
            'total_reviews': total_reviews,
            'avg_rating': avg_rating,
            'days': Config.DAYS
        }
        
        # Render template
        html_content = template.render(context)
        
        # Save HTML file
        html_file = os.path.join(output_dir, f"review_report_{start_date}_{end_date}.html")
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"HTML report generated successfully: {html_file}")
        return html_file
        
    except Exception as e:
        print(f"Error generating HTML report: {str(e)}")
        raise


def main():
    """
    Main execution function
    """
    try:
        print("=== PlayStore Review Analysis Started ===")
        
        # Step 1: Setup environment
        font_name, font_path, output_dir = setup_environment()
        
        # Step 2: Get date range
        start_date, end_date = get_date_range()
        
        # Step 3: Collect reviews
        review_data = collect_reviews()
        
        # Step 4: Process reviews
        df = process_reviews(review_data, start_date, end_date)
        if df is None:
            return
        
        # Step 5: Save CSV
        save_csv(df, output_dir, start_date, end_date)
        
        # Step 6: Create visualizations
        summary_img = create_summary_chart(df, font_name, output_dir, start_date, end_date)
        wc_img = create_wordcloud(df, font_path, output_dir, start_date, end_date)
        
        # Step 7: Generate HTML report
        generate_html_report(df, output_dir, start_date, end_date, summary_img, wc_img)
        
        print(f"✓ Review summary files for the last {Config.DAYS} days generated successfully!")
        print("=== Analysis Complete ===")
        
    except Exception as e:
        print(f"✗ Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()

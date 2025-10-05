#!/usr/bin/env python3
"""
Multi-Platform Review Analysis Script

This script collects Google Play Store reviews for a specified app,
analyzes them, and generates visualization reports including:
- CSV data export
- Time series charts
- Word clouds
- Interactive HTML reports with platform comparison
- Topic modeling analysis
"""

from google_play_scraper import reviews, Sort
import feedparser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from datetime import datetime, timedelta
import os
import platform
from typing import Tuple, Optional, List, Dict
from jinja2 import Environment, FileSystemLoader
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    BERTOPIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: BERTopic dependencies not available: {e}")
    print("Topic modeling will be skipped. Install with: pip install -r requirements.txt")
    BERTOPIC_AVAILABLE = False
import json


# ===============================
# Configuration
# ===============================
class Config:
    """Configuration settings for the review analysis"""
    # Google Play Store settings
    PLAYSTORE_APP_PACKAGE = 'com.kakao.yellowid'
    
    # Apple App Store settings (RSS)
    APPSTORE_APP_ID = '990571676' 
    
    @classmethod
    def get_appstore_rss_url(cls):
        return f'https://itunes.apple.com/kr/rss/customerreviews/id={cls.APPSTORE_APP_ID}/sortby=mostrecent/xml'
    
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


def collect_playstore_reviews() -> list:
    """
    Collect reviews from Google Play Store
    
    Returns:
        List of review data with platform information
    """
    print("Fetching reviews from Google Play Store...")
    try:
        result, _ = reviews(
            Config.PLAYSTORE_APP_PACKAGE,
            lang='ko',
            country='kr',
            count=Config.REVIEW_COUNT,
            sort=Sort.NEWEST
        )
        # Add platform information
        for review in result:
            review['platform'] = 'Google Play Store'
        print(f"✓ Fetched {len(result)} reviews from Play Store")
        return result
    except Exception as e:
        print(f"❌ Error fetching Play Store reviews: {e}")
        return []


def collect_appstore_reviews() -> list:
    """
    Collect reviews from Apple App Store using RSS feed with multiple pages
    
    Returns:
        List of review data with platform information
    """
    print("Fetching reviews from Apple App Store (RSS)...")
    try:
        all_results = []
        
        # Try to get more reviews by using different RSS endpoints
        # Apple RSS feeds support different sorting and page parameters
        rss_urls = [
            f'https://itunes.apple.com/kr/rss/customerreviews/id={Config.APPSTORE_APP_ID}/sortby=mostrecent/xml',
            f'https://itunes.apple.com/kr/rss/customerreviews/id={Config.APPSTORE_APP_ID}/sortby=mostrelevant/xml',
            f'https://itunes.apple.com/kr/rss/customerreviews/id={Config.APPSTORE_APP_ID}/page=1/sortby=mostrecent/xml',
            f'https://itunes.apple.com/kr/rss/customerreviews/id={Config.APPSTORE_APP_ID}/page=2/sortby=mostrecent/xml',
            f'https://itunes.apple.com/kr/rss/customerreviews/id={Config.APPSTORE_APP_ID}/page=3/sortby=mostrecent/xml',
            f'https://itunes.apple.com/kr/rss/customerreviews/id={Config.APPSTORE_APP_ID}/page=4/sortby=mostrecent/xml',
            f'https://itunes.apple.com/kr/rss/customerreviews/id={Config.APPSTORE_APP_ID}/page=5/sortby=mostrecent/xml'
        ]
        
        seen_reviews = set()  # To avoid duplicates
        
        for i, rss_url in enumerate(rss_urls):
            try:
                print(f"  Fetching from RSS page {i+1}...")
                
                # Parse RSS feed
                feed = feedparser.parse(rss_url)
                
                if feed.bozo and i == 0:  # Only warn for first URL
                    print(f"⚠️  RSS feed parsing warning: {feed.bozo_exception}")
                
                page_results = []
                for entry in feed.entries:
                    try:
                        # Extract review data from RSS entry
                        
                        # Parse rating from title (usually format: "★★★★☆ - Title")
                        title = entry.get('title', '')
                        rating = 0
                        if '★' in title:
                            rating = title.count('★')
                        
                        # Get review content
                        content = ''
                        if hasattr(entry, 'content') and entry.content:
                            content = entry.content[0].value if isinstance(entry.content, list) else entry.content
                        elif hasattr(entry, 'summary'):
                            content = entry.summary
                        
                        # Clean content (remove HTML tags if any)
                        content = re.sub(r'<[^>]+>', '', content) if content else ''
                        
                        # Get author name
                        author = entry.get('author', 'Anonymous')
                        
                        # Get date
                        date = datetime.now()
                        if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            date = datetime(*entry.updated_parsed[:6])
                        elif hasattr(entry, 'published_parsed') and entry.published_parsed:
                            date = datetime(*entry.published_parsed[:6])
                        
                        # Create unique identifier to avoid duplicates
                        review_id = f"{author}_{content[:50]}_{date.strftime('%Y%m%d')}"
                        
                        if review_id not in seen_reviews:
                            seen_reviews.add(review_id)
                            page_results.append({
                                'userName': author,
                                'content': content,
                                'score': rating,
                                'at': date,
                                'platform': 'Apple App Store'
                            })
                        
                    except Exception as e:
                        print(f"⚠️  Error parsing RSS entry: {e}")
                        continue
                
                all_results.extend(page_results)
                print(f"    ✓ Got {len(page_results)} new reviews from page {i+1}")
                
                # If no new reviews found, stop trying more pages
                if len(page_results) == 0 and i > 0:
                    print(f"    No more reviews found, stopping at page {i+1}")
                    break
                    
            except Exception as e:
                print(f"⚠️  Error fetching RSS page {i+1}: {e}")
                continue
        
        print(f"✓ Fetched total {len(all_results)} unique reviews from App Store RSS")
        return all_results
        
    except Exception as e:
        print(f"❌ Error fetching App Store RSS reviews: {e}")
        return []


def collect_all_reviews() -> list:
    """
    Collect reviews from both platforms
    
    Returns:
        Combined list of review data from both platforms
    """
    print("=== Starting multi-platform review collection ===")
    
    # Collect from both platforms
    playstore_reviews = collect_playstore_reviews()
    appstore_reviews = collect_appstore_reviews()
    
    # Combine results
    all_reviews = playstore_reviews + appstore_reviews
    
    total_count = len(all_reviews)
    playstore_count = len(playstore_reviews)
    appstore_count = len(appstore_reviews)
    
    print(f"\n📊 Collection Summary:")
    print(f"   • Google Play Store: {playstore_count} reviews")
    print(f"   • Apple App Store: {appstore_count} reviews")
    print(f"   • Total: {total_count} reviews")
    
    return all_reviews


def process_reviews(review_data: list, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Process and filter review data from multiple platforms
    
    Args:
        review_data: Raw review data from both platforms
        start_date: Start date for filtering
        end_date: End date for filtering
        
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
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return None
    
    df = df[required_columns]
    df.rename(columns={'content': 'review'}, inplace=True)
    df['at'] = pd.to_datetime(df['at'])
    
    # Filter for date range
    df = df[(df['at'].dt.date >= start_date) & (df['at'].dt.date <= end_date)]
    df.sort_values('at', inplace=True)
    
    if df.empty:
        print(f"No reviews found for the last {Config.DAYS} days.")
        return None
    
    # Print platform breakdown
    platform_counts = df['platform'].value_counts()
    print(f"\n✓ Processed {len(df)} reviews within date range:")
    for platform, count in platform_counts.items():
        print(f"   • {platform}: {count} reviews")
        
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
    Create time series summary chart with platform comparison
    
    Args:
        df: DataFrame with review data including platform info
        font_name: Font name for chart
        output_dir: Output directory path
        start_date: Start date for title
        end_date: End date for title
        
    Returns:
        Path to saved chart image
    """
    # Prepare data
    df['date'] = df['at'].dt.date
    
    # Create figure with subplots
    sns.set(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Chart 1: Daily average scores by platform
    if 'platform' in df.columns:
        daily_score_platform = df.groupby(['date', 'platform'])['score'].mean().reset_index()
        
        for platform in df['platform'].unique():
            platform_data = daily_score_platform[daily_score_platform['platform'] == platform]
            color = '#1f77b4' if 'Google' in platform else '#ff7f0e'
            ax1.plot(platform_data['date'], platform_data['score'], 
                    marker='o', label=f'{platform} 평균 별점', color=color, linewidth=2)
    else:
        daily_score = df.groupby('date')['score'].mean().reset_index()
        ax1.plot(daily_score['date'], daily_score['score'], marker='o', label='평균 별점')
    
    ax1.set_xlabel("날짜", fontname=font_name)
    ax1.set_ylabel("평균 별점", fontname=font_name)
    ax1.set_title(f"플랫폼별 일일 평균 별점 ({start_date} ~ {end_date})", fontname=font_name, fontsize=14, fontweight='bold')
    ax1.legend(prop={"family": font_name})
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Daily review counts by platform
    if 'platform' in df.columns:
        daily_count_platform = df.groupby(['date', 'platform']).size().reset_index(name='count')
        
        # Create stacked bar chart
        platforms = df['platform'].unique()
        dates = sorted(df['date'].unique())
        
        bottom = None
        colors = ['#1f77b4', '#ff7f0e']
        
        for i, platform in enumerate(platforms):
            platform_counts = []
            for date in dates:
                count = daily_count_platform[
                    (daily_count_platform['date'] == date) & 
                    (daily_count_platform['platform'] == platform)
                ]['count'].sum()
                platform_counts.append(count)
            
            ax2.bar(dates, platform_counts, label=f'{platform} 리뷰 수', 
                   bottom=bottom, color=colors[i % len(colors)], alpha=0.8)
            
            if bottom is None:
                bottom = platform_counts
            else:
                bottom = [b + c for b, c in zip(bottom, platform_counts)]
    else:
        daily_count = df.groupby('date').size().reset_index(name='count')
        ax2.bar(daily_count['date'], daily_count['count'], alpha=0.8, label='리뷰 수', color='skyblue')
    
    ax2.set_xlabel("날짜", fontname=font_name)
    ax2.set_ylabel("리뷰 수", fontname=font_name)
    ax2.set_title(f"플랫폼별 일일 리뷰 수 ({start_date} ~ {end_date})", fontname=font_name, fontsize=14, fontweight='bold')
    ax2.legend(prop={"family": font_name})
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    summary_img = os.path.join(output_dir, f"multi_platform_summary_{start_date}_{end_date}.png")
    plt.savefig(summary_img, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Multi-platform summary visualization saved: {summary_img}")
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


def preprocess_text_for_bert(text: str) -> str:
    """
    Preprocess text for BERT analysis
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Remove special characters but keep Korean, English, and spaces
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove very short texts
    if len(text) < 10:
        return ""
    return text


def perform_topic_modeling(df: pd.DataFrame, output_dir: str, 
                          start_date: datetime.date, end_date: datetime.date) -> Tuple[pd.DataFrame, str, Dict]:
    """
    Perform topic modeling using BERTopic
    
    Args:
        df: DataFrame with review data
        output_dir: Output directory path
        start_date: Start date for filename
        end_date: End date for filename
        
    Returns:
        Tuple of (DataFrame with topics, visualization path, topic info)
    """
    if not BERTOPIC_AVAILABLE:
        print("⚠️  BERTopic not available. Skipping topic modeling...")
        return df, "", {}
    
    print("Starting topic modeling with BERTopic...")
    
    # Preprocess reviews
    reviews = df['review'].astype(str).apply(preprocess_text_for_bert)
    reviews = reviews[reviews.str.len() > 0].tolist()
    
    if len(reviews) < 10:
        print(f"⚠️  Not enough reviews for topic modeling (need at least 10, got {len(reviews)})")
        return df, "", {}
    
    print(f"📊 Processing {len(reviews)} reviews for topic modeling...")
    
    try:
        # Initialize sentence transformer for Korean
        sentence_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # Import UMAP and HDBSCAN for custom configuration
        from umap import UMAP
        from hdbscan import HDBSCAN
        
        # Configure UMAP with safe parameters
        n_components = min(5, len(reviews) - 1)  # Ensure n_components < n_samples
        n_neighbors = min(15, len(reviews) - 1)  # Ensure n_neighbors < n_samples
        
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # Configure HDBSCAN
        min_cluster_size = max(2, len(reviews) // 20)  # More conservative clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Initialize BERTopic with custom models
        topic_model = BERTopic(
            embedding_model=sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            language="korean",
            calculate_probabilities=True,
            verbose=True,
            min_topic_size=max(2, len(reviews) // 15)  # Adaptive min_topic_size
        )
        
        # Fit the model and predict topics
        topics, probs = topic_model.fit_transform(reviews)
        
        # Validate results
        if topics is None or len(topics) == 0:
            raise ValueError("No topics were generated")
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        
        if topic_info is None or len(topic_info) == 0:
            raise ValueError("No topic information available")
            
        print(f"✓ Found {len(topic_info)} topics")
        
        # Validate that we have valid topics (not just outliers)
        valid_topics = topic_info[topic_info['Topic'] != -1]
        if len(valid_topics) == 0:
            print("⚠️  Only outlier topics found, using simplified approach...")
            # Create simple topic assignment based on sentiment
            df_with_topics = df.copy()
            df_with_topics['topic'] = 0  # Single topic for all
            df_with_topics['topic_label'] = "일반 리뷰"
        else:
            # Add topics to dataframe
            df_with_topics = df.copy()
            
            # Create proper mapping between original reviews and topics
            # The 'topics' array corresponds to the filtered 'reviews' list
            df_with_topics['topic'] = -1  # Default to outlier
            
            # Map topics back to original dataframe
            valid_review_idx = 0
            for df_idx in df_with_topics.index:
                original_review = df_with_topics.loc[df_idx, 'review']
                processed_review = preprocess_text_for_bert(str(original_review))
                
                if len(processed_review) > 0:  # This review was included in topic modeling
                    if valid_review_idx < len(topics):
                        df_with_topics.loc[df_idx, 'topic'] = topics[valid_review_idx]
                        valid_review_idx += 1
                    else:
                        df_with_topics.loc[df_idx, 'topic'] = -1
                else:
                    df_with_topics.loc[df_idx, 'topic'] = -1
            
            # Generate topic labels with safety checks
            topic_labels = {}
            for topic_id in topic_info['Topic'].tolist():
                if topic_id != -1:
                    try:
                        topic_words = topic_model.get_topic(topic_id)
                        if topic_words and len(topic_words) > 0:
                            words = [word for word, _ in topic_words[:3]]
                            if words:
                                topic_labels[topic_id] = f"주제 {topic_id}: {', '.join(words)}"
                            else:
                                topic_labels[topic_id] = f"주제 {topic_id}"
                        else:
                            topic_labels[topic_id] = f"주제 {topic_id}"
                    except Exception as e:
                        print(f"Warning: Could not get words for topic {topic_id}: {e}")
                        topic_labels[topic_id] = f"주제 {topic_id}"
                else:
                    topic_labels[topic_id] = "기타"
            
            df_with_topics['topic_label'] = df_with_topics['topic'].map(topic_labels).fillna("기타")
        
        # Create topic distribution chart (safer approach)
        topic_counts = df_with_topics['topic_label'].value_counts()
        
        if len(topic_counts) > 0:
            try:
                # Skip BERTopic's built-in visualization for small datasets
                # It often fails with "zero-size array" error
                plot_file = ""
                
                # Create topic visualization only if we have enough data
                if len(valid_topics) > 0 and len(reviews) > 50:
                    try:
                        fig = topic_model.visualize_topics()
                        plot_file = os.path.join(output_dir, f"topic_visualization_{start_date}_{end_date}.html")
                        plot(fig, filename=plot_file, auto_open=False)
                        print(f"Topic visualization saved: {plot_file}")
                    except Exception as viz_error:
                        print(f"Skipping BERTopic visualization due to small dataset: {viz_error}")
                        plot_file = ""
                else:
                    print("Skipping BERTopic visualization (dataset too small or no valid topics)")
                
                # Create distribution chart with safe data
                if len(topic_counts) > 0 and topic_counts.sum() > 0:
                    # Ensure we have valid data for plotting
                    valid_counts = topic_counts[topic_counts > 0]
                    
                    if len(valid_counts) > 0:
                        fig_dist = px.bar(
                            x=valid_counts.values,
                            y=valid_counts.index,
                            orientation='h',
                            title='리뷰 주제 분포',
                            labels={'x': '리뷰 수', 'y': '주제'}
                        )
                        fig_dist.update_layout(
                            height=max(300, len(valid_counts) * 60),  # Dynamic height
                            font=dict(family="Arial, sans-serif", size=12),
                            title_font_size=16,
                            margin=dict(l=200, r=50, t=80, b=50),  # More space for topic labels
                            showlegend=False
                        )
                        
                        # Create a simple but reliable matplotlib chart
                        dist_img = ""
                        try:
                            # Use the same font setup as other charts
                            font_name, font_path, _ = setup_environment()[:3]
                            
                            # Create figure with proper font
                            fig_height = max(4, len(valid_counts) * 0.8)
                            plt.figure(figsize=(12, fig_height))
                            
                            # Use the same font as other charts
                            if font_name:
                                plt.rcParams['font.family'] = font_name
                            plt.rcParams['axes.unicode_minus'] = False
                            
                            # Create horizontal bar chart with colors
                            colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#607D8B']
                            y_pos = range(len(valid_counts))
                            
                            bars = plt.barh(y_pos, valid_counts.values, 
                                          color=colors[:len(valid_counts)], alpha=0.8)
                            
                            # Customize chart
                            plt.yticks(y_pos, valid_counts.index, fontsize=12)
                            plt.xlabel('리뷰 수', fontsize=14, fontweight='bold')
                            plt.title('리뷰 주제 분포', fontsize=16, fontweight='bold', pad=20)
                            
                            # Add value labels on bars
                            for i, value in enumerate(valid_counts.values):
                                plt.text(value + max(valid_counts.values) * 0.01, i, f'{value}건', 
                                        va='center', ha='left', fontweight='bold', fontsize=11)
                            
                            # Style improvements
                            plt.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
                            ax = plt.gca()
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_linewidth(0.5)
                            ax.spines['bottom'].set_linewidth(0.5)
                            
                            # Set margins
                            plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)
                            
                            # Save image
                            dist_img = os.path.join(output_dir, f"topic_distribution_{start_date}_{end_date}.png")
                            plt.savefig(dist_img, dpi=150, bbox_inches='tight', 
                                      facecolor='white', edgecolor='none')
                            plt.close()
                            print(f"✓ Topic distribution chart saved: {dist_img}")
                            
                        except Exception as chart_error:
                            print(f"Warning: Could not create topic distribution chart: {chart_error}")
                            print("Continuing without chart image...")
                            dist_img = ""
                    else:
                        print("No valid topic counts for visualization")
                        dist_img = ""
                else:
                    print("No topic data available for visualization")
                    dist_img = ""
                
            except Exception as viz_error:
                print(f"Warning: Could not create visualizations: {viz_error}")
                plot_file = ""
                dist_img = ""
        else:
            print("No topics to visualize")
            plot_file = ""
            dist_img = ""
        
        # Prepare topic summary for template
        valid_topic_count = len(valid_topics) if len(valid_topics) > 0 else 1
        topic_summary = {
            'total_topics': valid_topic_count,
            'topic_distribution': topic_counts.to_dict() if len(topic_counts) > 0 else {"일반 리뷰": len(df_with_topics)},
            'visualization_file': os.path.basename(plot_file) if plot_file else "",
            'distribution_image': os.path.basename(dist_img) if dist_img else ""
        }
        
        print(f"✅ Topic modeling completed successfully!")
        
        return df_with_topics, dist_img, topic_summary
        
    except Exception as e:
        print(f"❌ Error in topic modeling: {str(e)}")
        print("💡 Tip: This often happens with small datasets. Try with more reviews.")
        print("🔄 Continuing without topic analysis...")
        return df, "", {}


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
                        end_date: datetime.date, summary_img: str, wc_img: str, 
                        topic_img: str = "", topic_summary: Dict = None) -> str:
    """
    Generate responsive HTML report using Jinja2 templates
    
    Args:
        df: DataFrame with review data (including topic information)
        output_dir: Output directory path
        start_date: Start date for report
        end_date: End date for report
        summary_img: Path to summary image
        wc_img: Path to word cloud image
        topic_img: Path to topic distribution image
        topic_summary: Dictionary with topic analysis results
        
    Returns:
        Path to generated HTML file
    """
    try:
        # Setup Jinja2 environment
        env = setup_jinja_environment()
        template = env.get_template('report.html')
        
        # Prepare data for template
        columns_to_drop = ['date']
        if 'topic' in df.columns:
            # Keep topic_label but drop topic number
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
            'days': Config.DAYS,
            'has_topics': 'topic_label' in df.columns,
            'topic_summary': topic_summary or {},
            'topic_img': os.path.basename(topic_img) if topic_img else "",
            'platform_stats': platform_stats
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
        print("=== Multi-Platform Review Analysis Started ===")
        
        # Step 1: Setup environment
        font_name, font_path, output_dir = setup_environment()
        
        # Step 2: Get date range
        start_date, end_date = get_date_range()
        
        # Step 3: Collect reviews from both platforms
        review_data = collect_all_reviews()
        
        # Step 4: Process reviews
        df = process_reviews(review_data, start_date, end_date)
        if df is None:
            return
        
        # Step 5: Save CSV (initial)
        save_csv(df, output_dir, start_date, end_date)
        
        # Step 6: Create visualizations
        summary_img = create_summary_chart(df, font_name, output_dir, start_date, end_date)
        wc_img = create_wordcloud(df, font_path, output_dir, start_date, end_date)
        
        # Step 7: Perform topic modeling with BERTopic
        df_with_topics, topic_img, topic_summary = perform_topic_modeling(df, output_dir, start_date, end_date)
        
        # Step 8: Save updated CSV with topics
        if 'topic_label' in df_with_topics.columns:
            csv_file_with_topics = os.path.join(output_dir, f"user_reviews_with_topics_{start_date}_{end_date}.csv")
            df_with_topics.to_csv(csv_file_with_topics, index=False, encoding='utf-8-sig')
            print(f"CSV with topics saved: {csv_file_with_topics}")
        
        # Step 9: Generate HTML report with topic analysis
        generate_html_report(df_with_topics, output_dir, start_date, end_date, 
                           summary_img, wc_img, topic_img, topic_summary)
        
        print(f"✓ Review analysis with topic modeling for the last {Config.DAYS} days completed!")
        print("=== Analysis Complete ===")
        
    except Exception as e:
        print(f"✗ Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()

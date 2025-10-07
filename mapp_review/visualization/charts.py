"""
Chart generation for review analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from datetime import datetime


class ChartGenerator:
    """Generate various charts for review analysis"""
    
    def __init__(self, font_name: str, font_path: str):
        self.font_name = font_name
        self.font_path = font_path
        
        # Set matplotlib font
        plt.rcParams['font.family'] = font_name
    
    def create_summary_chart(self, df: pd.DataFrame, output_dir: str, 
                            start_date: datetime.date, end_date: datetime.date) -> str:
        """Create time series summary chart with platform comparison"""
        
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
        
        ax1.set_xlabel("날짜", fontname=self.font_name)
        ax1.set_ylabel("평균 별점", fontname=self.font_name)
        ax1.set_title(f"플랫폼별 일일 평균 별점 ({start_date} ~ {end_date})", fontname=self.font_name, fontsize=14, fontweight='bold')
        ax1.legend(prop={"family": self.font_name})
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
        
        ax2.set_xlabel("날짜", fontname=self.font_name)
        ax2.set_ylabel("리뷰 수", fontname=self.font_name)
        ax2.set_title(f"플랫폼별 일일 리뷰 수 ({start_date} ~ {end_date})", fontname=self.font_name, fontsize=14, fontweight='bold')
        ax2.legend(prop={"family": self.font_name})
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        summary_img = os.path.join(output_dir, f"mapp_summary_{start_date}_{end_date}.png")
        plt.savefig(summary_img, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Mobile app summary visualization saved: {summary_img}")
        return summary_img
    
    def create_wordcloud(self, df: pd.DataFrame, output_dir: str,
                        start_date: datetime.date, end_date: datetime.date) -> str:
        """Create word cloud from review text with Korean morphological analysis"""
        
        # Combine all review text
        text = ' '.join(df['review'].astype(str))
        
        # Extract meaningful Korean keywords
        keywords = self._extract_korean_keywords(text)
        processed_text = ' '.join(keywords)
        
        # Create WordCloud
        wordcloud = WordCloud(
            font_path=self.font_path,
            width=1200,
            height=600,
            background_color='white',
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5,
            collocations=False  # Prevent duplicate word combinations
        ).generate(processed_text)
        
        # Create plot
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'리뷰 키워드 워드클라우드 ({start_date} ~ {end_date})', 
                 fontname=self.font_name, fontsize=16, pad=20)
        
        # Save image
        wc_img = os.path.join(output_dir, f"review_wordcloud_{start_date}_{end_date}.png")
        plt.savefig(wc_img, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Word cloud image saved: {wc_img}")
        return wc_img
    
    def _extract_korean_keywords(self, text: str) -> list:
        """Extract meaningful Korean keywords using morphological analysis"""
        try:
            from konlpy.tag import Okt
            okt = Okt()
            
            # Korean stopwords for mobile app reviews (based on common patterns)
            stopwords = {
                # Basic particles and endings
                '것', '수', '때', '거', '게', '걸', '건', '겠', '고', '는', '다', '도', '를', '이', '가',
                '에', '의', '로', '으로', '와', '과', '한', '을', '를', '께', '서', '부터', '까지',
                                
                # Demonstratives and question words
                '그', '저', '이', '이런', '저런', '그런', '어떤', '무슨', '어느', '여기', '거기', '저기',
                
                # Adverbs and intensifiers
                '너무', '정말', '진짜', '완전', '엄청', '되게', '좀', '잘', '못', '안', '아주', '매우',
                '많이', '조금', '약간', '꽤', '상당히', '굉장히', '정말로', '진짜로',
                
                # Time expressions (usually not meaningful for sentiment)
                '오늘', '어제', '내일', '지금', '나중', '전에', '후에', '동안', '때문', '이후',
            }
            
            # Perform morphological analysis and extract nouns/adjectives
            # Use stem=False to preserve original word forms and avoid over-stemming
            words = okt.pos(text, stem=False)
            keywords = []
            
            for word, pos in words:
                # Select meaningful nouns and adjectives only
                if pos in ['Noun', 'Adjective'] and len(word) > 1:
                    if word not in stopwords and word.isalpha():
                        keywords.append(word)
            
            return keywords
            
        except ImportError:
            print("⚠️  KoNLPy not available, using simple text processing")
            return self._simple_korean_processing(text)
    
    def _simple_korean_processing(self, text: str) -> list:
        """Fallback: Simple Korean text processing without morphological analysis"""
        import re
        
        # Simple preprocessing
        # Remove special characters, extract Korean text only
        korean_text = re.sub(r'[^가-힣\s]', '', text)
        words = korean_text.split()
        
        # Select words with length >= 2
        keywords = [word for word in words if len(word) >= 2]
        
        return keywords

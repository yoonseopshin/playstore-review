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
        """Create combined dual-axis chart with review counts and average ratings"""
        
        # Prepare data
        df['date'] = df['at'].dt.date
        
        # Create figure with single plot - modern styling
        sns.set_style("whitegrid")
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
        fig.patch.set_facecolor('white')
        
        # Define modern UI colors for better visualization
        platform_colors = {
            'Google Play Store': {
                'bar': '#3b82f6',      # Modern Blue
                'line': '#1d4ed8'      # Deep Blue
            },
            'Apple App Store': {
                'bar': '#f59e0b',      # Amber/Orange
                'line': '#d97706'      # Deep Orange
            }
        }
        
        if 'platform' in df.columns:
            # Prepare data for both metrics
            daily_count_platform = df.groupby(['date', 'platform']).size().reset_index(name='count')
            daily_score_platform = df.groupby(['date', 'platform'])['score'].mean().reset_index()
            
            platforms = df['platform'].unique()
            dates = sorted(df['date'].unique())
            
            # Left axis (ax1): Review counts (Bar chart)
            ax1.set_xlabel("날짜", fontname=self.font_name, fontsize=12)
            ax1.set_ylabel("일일 리뷰 수", fontname=self.font_name, fontsize=12)
            
            # Create grouped bar chart for review counts
            bar_width = 0.35
            x_positions = range(len(dates))
            
            for i, platform in enumerate(platforms):
                platform_counts = []
                for date in dates:
                    count = daily_count_platform[
                        (daily_count_platform['date'] == date) & 
                        (daily_count_platform['platform'] == platform)
                    ]['count'].sum()
                    platform_counts.append(count)
                
                # Offset bars for each platform
                offset_positions = [x + (i - 0.5) * bar_width for x in x_positions]
                colors = platform_colors.get(platform, {'bar': '#1f77b4', 'line': '#1f77b4'})
                bar_color = colors['bar']
                
                ax1.bar(offset_positions, platform_counts, 
                       width=bar_width, label=f'{platform} 리뷰 수', 
                       color=bar_color, alpha=0.85, edgecolor='white', linewidth=1.5,
                       capstyle='round')
            
            # Right axis (ax2): Average ratings (Line chart)
            ax2 = ax1.twinx()
            ax2.set_ylabel("평균 별점", fontname=self.font_name, fontsize=12)
            ax2.set_ylim(0, 5.5)  # Rating scale from 0 to 5
            
            for platform in platforms:
                platform_data = daily_score_platform[daily_score_platform['platform'] == platform]
                
                # Match dates with x_positions
                platform_scores = []
                platform_x_pos = []
                for i, date in enumerate(dates):
                    score_data = platform_data[platform_data['date'] == date]
                    if not score_data.empty:
                        platform_scores.append(score_data['score'].iloc[0])
                        platform_x_pos.append(i)
                
                if platform_scores:  # Only plot if there's data
                    colors = platform_colors.get(platform, {'bar': '#1f77b4', 'line': '#1f77b4'})
                    line_color = colors['line']
                    
                    ax2.plot(platform_x_pos, platform_scores, 
                            marker='o', label=f'{platform} 평균 별점', 
                            color=line_color, linewidth=3.5, markersize=10,
                            markerfacecolor='white', markeredgecolor=line_color, 
                            markeredgewidth=2.5, alpha=0.9)
            
            # Set x-axis labels
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels([str(date) for date in dates], rotation=45, ha='right')
            
        else:
            # Fallback for single platform
            daily_count = df.groupby('date').size().reset_index(name='count')
            daily_score = df.groupby('date')['score'].mean().reset_index()
            
            ax1.bar(daily_count['date'], daily_count['count'], alpha=0.85, label='리뷰 수', 
                   color='#3b82f6', edgecolor='white', linewidth=1.5)
            ax1.set_ylabel("리뷰 수", fontname=self.font_name)
            
            ax2 = ax1.twinx()
            ax2.plot(daily_score['date'], daily_score['score'], marker='o', label='평균 별점', 
                    color='#1d4ed8', linewidth=3.5, markersize=10,
                    markerfacecolor='white', markeredgecolor='#1d4ed8', markeredgewidth=2.5, alpha=0.9)
            ax2.set_ylabel("평균 별점", fontname=self.font_name)
        
        # Title and styling
        ax1.set_title(f"플랫폼별 일일 리뷰 수 및 평균 별점 ({start_date} ~ {end_date})", 
                     fontname=self.font_name, fontsize=16, fontweight='bold', pad=20)
        
        # Grid styling
        ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.8, color='#e5e7eb')
        ax2.grid(False)  # Disable grid on secondary axis
        
        # Modern legend styling
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, 
                           loc='upper left', prop={"family": self.font_name, "size": 11}, 
                           frameon=True, fancybox=True, shadow=False,
                           facecolor='white', edgecolor='#d1d5db', framealpha=0.95)
        legend.get_frame().set_linewidth(1.2)
        
        plt.tight_layout()
        
        # Save chart
        summary_img = os.path.join(output_dir, f"mapp_summary_{start_date}_{end_date}.png")
        plt.savefig(summary_img, dpi=150, bbox_inches='tight', facecolor='white')
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

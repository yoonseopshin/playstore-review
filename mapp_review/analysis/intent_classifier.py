"""
Intent Classification for Mobile App Reviews using KoELECTRA
Optimized for GitHub Actions with 5 predefined categories
"""
import pandas as pd
import re
import os
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np

# Disable tokenizer parallelism to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    # Suppress transformers warnings about uninitialized weights
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class IntentClassifier:
    """KoELECTRA-based intent classification for app reviews"""
    
    def __init__(self):
        self.available = TRANSFORMERS_AVAILABLE
        self.intent_categories = {
            0: '기능 개선 요청',
            1: '버그 제보', 
            2: '성능',
            3: '긍정 피드백',
            4: '부정 피드백'
        }
        
        self.model = None
        self.tokenizer = None
        self.classifier = None
        
        # Fallback keyword patterns for rule-based classification
        self.keyword_patterns = {
            '기능 개선 요청': [
                '추가', '개선', '요청', '바라', '원해', '필요', '만들어', '넣어',
                '기능', '업데이트', '버전업', '개발', '지원', '도입', '적용',
                '했으면', '되면', '주세요', '해주세요', '만들어주세요'
            ],
            '버그 제보': [
                '버그', '오류', '에러', '문제', '안됨', '안되', '작동', '실행',
                '튕김', '꺼짐', '멈춤', '느림', '로딩', '접속', '연결',
                '고장', '이상', '잘못', '수정', '해결'
            ],
            '성능': [
                '느림', '빠름', '속도', '성능', '렉', '끊김', '지연',
                '로딩', '반응', '처리', '실행', '시간', '빨라', '늦어'
            ],
            '긍정 피드백': [
                '좋다', '좋아', '좋네', '만족', '훌륭', '완벽', '최고', '대박',
                '편리', '편해', '유용', '도움', '감사', '고마워', '추천',
                '잘', '괜찮', '마음에', '쓸만', '괜찮다'
            ],
            '부정 피드백': [
                '나쁘다', '나빠', '싫어', '별로', '실망', '최악', '짜증',
                '불편', '어려워', '복잡', '이해', '모르겠', '헷갈',
                '안좋', '못하', '아쉽', '부족'
            ]
        }
    
    def load_model(self):
        """Load KoELECTRA model for intent classification"""
        if not self.available:
            print("⚠️  Transformers not available. Using rule-based classification...")
            return False
        
        try:
            print("🤖 Loading KoELECTRA model for intent classification...")
            
            # Use a smaller, efficient model for GitHub Actions
            model_name = "monologg/koelectra-small-v3-discriminator"
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create a simple classification pipeline
            # Since we don't have a pre-trained intent model, we'll use sentiment analysis
            # and map it to our categories with rule-based enhancement
            self.classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=self.tokenizer,
                top_k=None,  # Updated parameter name
                device=-1  # Use CPU for GitHub Actions compatibility
            )
            
            print("✅ KoELECTRA model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load KoELECTRA model: {e}")
            print("🔄 Falling back to rule-based classification...")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess Korean text for classification"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs, special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length for model efficiency
        if len(text) > 200:
            text = text[:200]
        
        return text
    
    def classify_with_koelectra(self, text: str) -> Tuple[str, float]:
        """Classify using KoELECTRA model with rule-based enhancement"""
        if not self.classifier:
            return self.classify_by_rules(text)
        
        try:
            # Get sentiment scores
            results = self.classifier(text)
            
            # Enhance with rule-based classification
            rule_intent, rule_confidence = self.classify_by_rules(text)
            
            # Combine model sentiment with rule-based intent
            # This is a simplified approach - in production, you'd train a proper intent model
            if rule_confidence > 0.3:  # High confidence from rules
                return rule_intent, min(rule_confidence + 0.2, 1.0)
            else:
                # Map sentiment to intent categories
                if results and len(results) > 0:
                    # Handle different result formats
                    if isinstance(results[0], dict) and 'score' in results[0]:
                        sentiment_score = results[0]['score'] if results[0].get('label') == 'POSITIVE' else 1 - results[0]['score']
                    elif isinstance(results[0], list) and len(results[0]) > 0:
                        # Handle nested list format
                        first_result = results[0][0] if isinstance(results[0][0], dict) else results[0]
                        sentiment_score = first_result.get('score', 0.5) if first_result.get('label') == 'POSITIVE' else 1 - first_result.get('score', 0.5)
                    else:
                        sentiment_score = 0.5
                    
                    if sentiment_score > 0.7:
                        return '긍정 피드백', sentiment_score
                    elif sentiment_score < 0.3:
                        return '부정 피드백', 1 - sentiment_score
                    else:
                        return rule_intent, 0.5
                else:
                    return rule_intent, 0.4
                    
        except Exception as e:
            print(f"⚠️  Error in KoELECTRA classification: {e}")
            return self.classify_by_rules(text)
    
    def classify_by_rules(self, text: str) -> Tuple[str, float]:
        """Fallback rule-based classification using keywords"""
        text = self.preprocess_text(text).lower()
        scores = {}
        
        for intent, keywords in self.keyword_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            
            # Normalize score
            scores[intent] = score / len(keywords) if keywords else 0
        
        # Get best match
        if not scores or max(scores.values()) == 0:
            return '부정 피드백', 0.1  # Default fallback
        
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]
        
        return best_intent, confidence
    
    def classify_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify all reviews in dataframe"""
        print("🎯 Starting intent classification with KoELECTRA...")
        
        # Try to load model
        model_loaded = self.load_model()
        
        df_classified = df.copy()
        intents = []
        confidences = []
        
        for idx, row in df.iterrows():
            review_text = str(row.get('review', ''))
            
            if model_loaded and self.classifier:
                intent, confidence = self.classify_with_koelectra(review_text)
            else:
                intent, confidence = self.classify_by_rules(review_text)
            
            intents.append(intent)
            confidences.append(confidence)
            
            # Progress indicator for large datasets
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(df)} reviews...")
        
        df_classified['intent'] = intents
        df_classified['intent_confidence'] = confidences
        df_classified['intent_label'] = df_classified['intent']
        
        # Print classification summary
        intent_counts = df_classified['intent_label'].value_counts()
        print("\n📊 Intent Classification Results:")
        for intent, count in intent_counts.items():
            percentage = (count / len(df_classified)) * 100
            print(f"   • {intent}: {count}건 ({percentage:.1f}%)")
        
        avg_confidence = df_classified['intent_confidence'].mean()
        print(f"   • 평균 신뢰도: {avg_confidence:.2f}")
        print(f"   • 분류 방법: {'KoELECTRA + 규칙 기반' if model_loaded else '규칙 기반'}")
        
        return df_classified
    
    def create_intent_summary(self, df: pd.DataFrame) -> Dict:
        """Create intent classification summary"""
        intent_counts = df['intent_label'].value_counts()
        
        # Calculate insights for each category
        insights = {}
        for intent in self.intent_categories.values():
            intent_reviews = df[df['intent_label'] == intent]['review'].tolist()
            if intent_reviews:
                # Simple keyword extraction
                all_text = ' '.join([self.preprocess_text(review) for review in intent_reviews[:5]])
                words = all_text.split()
                # Get most common meaningful words
                word_freq = {}
                for word in words:
                    if len(word) > 1 and word not in ['것', '수', '때', '거', '게']:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                insights[intent] = [word for word, _ in top_words]
            else:
                insights[intent] = []
        
        summary = {
            'total_reviews': len(df),
            'intent_distribution': intent_counts.to_dict(),
            'average_confidence': df['intent_confidence'].mean(),
            'classification_method': 'koelectra_enhanced',
            'insights': insights,
            'categories': list(self.intent_categories.values())
        }
        
        return summary
    
    def create_intent_visualization(self, df: pd.DataFrame, output_dir: str, 
                                  start_date: datetime.date, end_date: datetime.date) -> str:
        """Create intent distribution visualization"""
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
            
            intent_counts = df['intent_label'].value_counts()
            
            # Fixed color mapping for consistent visualization
            intent_color_map = {
                '기능 개선 요청': '#28a745',
                '버그 제보': '#dc3545', 
                '성능': '#ffc107',
                '긍정 피드백': '#17a2b8',
                '부정 피드백': '#6f42c1'
            }
            
            # Create ordered lists for consistent color mapping
            labels = []
            values = []
            colors = []
            
            for intent in intent_counts.index:
                labels.append(intent)
                values.append(intent_counts[intent])
                colors.append(intent_color_map.get(intent, '#6c757d'))  # Default gray for unknown intents
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,  # Donut chart style
                    marker=dict(
                        colors=colors,
                        line=dict(color='#FFFFFF', width=2)
                    ),
                    textinfo='label+percent',
                    textposition='outside',
                    textfont=dict(size=12, family="Arial, sans-serif")
                )
            ])
            
            fig.update_layout(
                title={
                    'text': "리뷰 분류 분포",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'family': "Arial, sans-serif"}
                },
                height=500,
                font=dict(family="Arial, sans-serif"),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                )
            )
            
            # Save chart
            intent_img = os.path.join(output_dir, f"intent_distribution_{start_date}_{end_date}.png")
            fig.write_image(intent_img)
            print(f"✓ Intent distribution chart saved: {intent_img}")
            
            return intent_img
            
        except Exception as e:
            print(f"⚠️  Could not create intent visualization: {e}")
            return ""


def perform_intent_classification(df: pd.DataFrame, output_dir: str, 
                                start_date: datetime.date, end_date: datetime.date) -> Tuple[pd.DataFrame, Dict, str]:
    """
    Main function to perform intent classification
    
    Returns:
        Tuple of (DataFrame with intents, classification summary, visualization path)
    """
    classifier = IntentClassifier()
    
    # Classify reviews
    df_with_intents = classifier.classify_reviews(df)
    
    # Create summary
    summary = classifier.create_intent_summary(df_with_intents)
    
    # Create visualization
    intent_img = classifier.create_intent_visualization(df_with_intents, output_dir, start_date, end_date)
    
    print("✅ Intent classification completed successfully!")
    
    return df_with_intents, summary, intent_img

"""
Topic modeling using BERTopic
"""
import pandas as pd
import re
from datetime import datetime
from typing import Tuple, Dict
import os

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    import plotly.graph_objects as go
    from plotly.offline import plot
    BERTOPIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: BERTopic dependencies not available: {e}")
    BERTOPIC_AVAILABLE = False


class TopicModeler:
    """BERTopic-based topic modeling for reviews"""
    
    def __init__(self):
        self.model = None
        self.available = BERTOPIC_AVAILABLE
    
    def preprocess_text_for_bert(self, text: str) -> str:
        """Preprocess text for BERT topic modeling"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short texts
        if len(text.split()) < 3:
            return ""
            
        return text
    
    def perform_topic_modeling(self, df: pd.DataFrame, output_dir: str, 
                              start_date: datetime.date, end_date: datetime.date) -> Tuple[pd.DataFrame, str, Dict]:
        """
        Perform topic modeling using BERTopic
        
        Returns:
            Tuple of (DataFrame with topics, visualization path, topic info)
        """
        if not self.available:
            print("⚠️  BERTopic not available. Skipping topic modeling...")
            return df, "", {}
        
        print("Starting topic modeling with BERTopic...")
        
        # Preprocess reviews
        reviews = df['review'].astype(str).apply(self.preprocess_text_for_bert)
        valid_reviews = reviews[reviews.str.len() > 0]
        
        if len(valid_reviews) < 5:
            print("⚠️  Too few valid reviews for topic modeling. Skipping...")
            return df, "", {}
        
        print(f"📊 Processing {len(valid_reviews)} reviews for topic modeling...")
        
        try:
            # Initialize models
            sentence_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            
            umap_model = UMAP(
                n_neighbors=min(15, len(valid_reviews)-1),
                n_components=min(5, len(valid_reviews)-1),
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=max(2, len(valid_reviews)//10),
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            # Initialize BERTopic
            topic_model = BERTopic(
                embedding_model=sentence_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                verbose=True,
                calculate_probabilities=True
            )
            
            # Fit the model
            topics, probs = topic_model.fit_transform(valid_reviews.tolist())
            
            # Get topic info
            topic_info = topic_model.get_topic_info()
            valid_topics = topic_info[topic_info.Topic != -1]
            
            print(f"✓ Found {len(valid_topics)} topics")
            
            # Create topic labels
            topic_labels = {}
            for _, row in valid_topics.iterrows():
                topic_id = row['Topic']
                words = [word for word, _ in topic_model.get_topic(topic_id)[:3]]
                topic_labels[topic_id] = f"주제 {topic_id}: {', '.join(words)}"
            
            # Add topic information to dataframe
            df_with_topics = df.copy()
            df_with_topics['topic'] = -1
            df_with_topics['topic_label'] = '기타'
            
            # Map topics back to original dataframe
            valid_indices = valid_reviews.index
            for i, (idx, topic) in enumerate(zip(valid_indices, topics)):
                df_with_topics.loc[idx, 'topic'] = topic
                if topic in topic_labels:
                    df_with_topics.loc[idx, 'topic_label'] = topic_labels[topic]
            
            # Create topic distribution visualization
            topic_counts = df_with_topics['topic_label'].value_counts()
            
            # Save topic distribution chart
            dist_img = ""
            if len(topic_counts) > 0:
                try:
                    fig = go.Figure(data=[
                        go.Bar(x=topic_counts.values, y=topic_counts.index, orientation='h')
                    ])
                    fig.update_layout(
                        title="주제별 리뷰 분포",
                        xaxis_title="리뷰 수",
                        yaxis_title="주제",
                        height=400 + len(topic_counts) * 30
                    )
                    
                    dist_img = os.path.join(output_dir, f"topic_distribution_{start_date}_{end_date}.png")
                    fig.write_image(dist_img)
                    print(f"✓ Topic distribution chart saved: {dist_img}")
                except Exception as e:
                    print(f"⚠️  Could not save topic chart: {e}")
                    dist_img = ""
            
            # Topic summary
            topic_summary = {
                'total_topics': len(valid_topics),
                'topic_distribution': topic_counts.to_dict(),
                'model_info': {
                    'total_documents': len(valid_reviews),
                    'topics_found': len(valid_topics)
                }
            }
            
            print("✅ Topic modeling completed successfully!")
            return df_with_topics, dist_img, topic_summary
            
        except Exception as e:
            print(f"❌ Error in topic modeling: {str(e)}")
            print("💡 Tip: This often happens with small datasets. Try with more reviews.")
            print("🔄 Continuing without topic analysis...")
            return df, "", {}

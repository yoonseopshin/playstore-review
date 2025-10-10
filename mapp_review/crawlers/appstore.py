"""
Apple App Store review crawler using RSS feeds
"""
import re
from datetime import datetime
from typing import List, Dict

import feedparser

from .base import BaseCrawler


class AppStoreCrawler(BaseCrawler):
    """Apple App Store review crawler using RSS feeds"""
    
    def __init__(self, app_id: str):
        super().__init__(app_id)
        self.app_id = app_id
    
    def collect_reviews(self, count: int = 100) -> List[Dict]:
        """
        Collect reviews from Apple App Store using RSS feed with multiple pages
        
        Args:
            count: Number of reviews to collect (limited by RSS availability)
            
        Returns:
            List of review dictionaries
        """
        print("Fetching reviews from Apple App Store (RSS)...")
        
        try:
            all_results = []
            
            # Try to get more reviews by using different RSS endpoints
            rss_urls = [
                f'https://itunes.apple.com/kr/rss/customerreviews/id={self.app_id}/sortby=mostrecent/xml',
                f'https://itunes.apple.com/kr/rss/customerreviews/id={self.app_id}/sortby=mostrelevant/xml',
                f'https://itunes.apple.com/kr/rss/customerreviews/id={self.app_id}/page=1/sortby=mostrecent/xml',
                f'https://itunes.apple.com/kr/rss/customerreviews/id={self.app_id}/page=2/sortby=mostrecent/xml',
                f'https://itunes.apple.com/kr/rss/customerreviews/id={self.app_id}/page=3/sortby=mostrecent/xml',
                f'https://itunes.apple.com/kr/rss/customerreviews/id={self.app_id}/page=4/sortby=mostrecent/xml',
                f'https://itunes.apple.com/kr/rss/customerreviews/id={self.app_id}/page=5/sortby=mostrecent/xml'
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
                        review_data = self._parse_rss_entry(entry)
                        if review_data:
                            # Create unique identifier to avoid duplicates
                            review_id = f"{review_data['userName']}_{review_data['content'][:50]}_{review_data['at'].strftime('%Y%m%d')}"
                            
                            if review_id not in seen_reviews:
                                seen_reviews.add(review_id)
                                page_results.append(review_data)
                    
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
    
    def _parse_rss_entry(self, entry) -> Dict:
        """Parse a single RSS entry into standardized review format"""
        try:
            # Parse rating from title (usually format: "★★★★☆ - Title")
            title = entry.get('title', '')
            rating = None  # Default to None if we can't parse rating
            
            # Try multiple methods to extract rating
            if '★' in title:
                rating = title.count('★')
            elif hasattr(entry, 'im_rating'):
                # iTunes specific rating field
                try:
                    rating = int(float(entry.im_rating))
                except (ValueError, TypeError):
                    pass
            elif hasattr(entry, 'rating'):
                # Generic rating field
                try:
                    rating = int(float(entry.rating))
                except (ValueError, TypeError):
                    pass
            
            # If we couldn't parse rating, skip this entry as it's likely invalid
            # App Store requires rating for reviews, so 0 rating entries are parsing errors
            if rating is None or rating == 0:
                print(f"  ⚠️ Skipping entry with no valid rating: {title[:50]}...")
                return None
            
            # Get review content
            content = ''
            if hasattr(entry, 'content') and entry.content:
                content = entry.content[0].value if isinstance(entry.content, list) else entry.content
            elif hasattr(entry, 'summary'):
                content = entry.summary
            
            # Clean content (remove HTML tags if any)
            content = re.sub(r'<[^>]+>', '', content) if content else ''
            
            # Try to get version from im:version tag (iTunes specific)
            version_from_tag = None
            if hasattr(entry, 'im_version'):
                version_from_tag = entry.im_version
            elif hasattr(entry, 'tags'):
                for tag in entry.tags:
                    if 'version' in tag.get('term', '').lower():
                        version_from_tag = tag.get('term', '')
                        break
            
            # Get author name
            author = entry.get('author', 'Anonymous')
            
            # Get date
            date = datetime.now()
            if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                date = datetime(*entry.updated_parsed[:6])
            elif hasattr(entry, 'published_parsed') and entry.published_parsed:
                date = datetime(*entry.published_parsed[:6])
            
            # Try to extract version from content or title
            version = 'Unknown'
            
            # First, try version from RSS tag
            if version_from_tag:
                version = version_from_tag
            else:
                # Look for version patterns in content like "Version 1.2.3" or "v1.2.3"
                version_patterns = [
                    r'[Vv]ersion\s+(\d+\.\d+(?:\.\d+)?)',
                    r'[Vv]\s*(\d+\.\d+(?:\.\d+)?)',
                    r'앱\s*버전\s*(\d+\.\d+(?:\.\d+)?)',
                    r'버전\s*(\d+\.\d+(?:\.\d+)?)',
                    r'(\d+\.\d+\.\d+)',  # Simple pattern like 1.2.3
                    r'(\d+\.\d+)'        # Simple pattern like 1.2
                ]
                
                search_text = content + ' ' + title
                for pattern in version_patterns:
                    match = re.search(pattern, search_text)
                    if match:
                        version = match.group(1)
                        break
                
                # If no version found in content, keep as Unknown
            
            
            return {
                'userName': author,
                'content': content,
                'score': rating,
                'at': date,
                'platform': self.get_platform_name(),
                'version': version
            }
            
        except Exception as e:
            print(f"⚠️  Error parsing RSS entry: {e}")
            return None
    
    def get_platform_name(self) -> str:
        """Get the platform name"""
        return "Apple App Store"

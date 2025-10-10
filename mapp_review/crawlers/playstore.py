"""
Google Play Store review crawler
"""
from datetime import datetime
from typing import List, Dict

from google_play_scraper import reviews, Sort

from .base import BaseCrawler


class PlayStoreCrawler(BaseCrawler):
    """Google Play Store review crawler"""
    
    def __init__(self, app_package: str):
        super().__init__(app_package)
        self.app_package = app_package
    
    def collect_reviews(self, count: int = 100, target_days: int = 7) -> List[Dict]:
        """
        Collect reviews from Google Play Store with enhanced debugging
        
        Args:
            count: Number of reviews to collect
            target_days: Number of recent days to focus on
            
        Returns:
            List of review dictionaries
        """
        print(f"🔍 === PLAY STORE DEBUG INFO ===")
        print(f"📱 App Package: {self.app_package}")
        print(f"📊 Requested Count: {count}")
        print(f"📅 Target Days: {target_days}")
        
        # Environment debugging
        import os
        import platform
        import time
        import random
        import socket
        import requests
        
        print(f"🖥️  Environment Info:")
        print(f"   • OS: {platform.system()} {platform.release()}")
        print(f"   • Python: {platform.python_version()}")
        print(f"   • Working Directory: {os.getcwd()}")
        print(f"   • User Agent Env: {os.getenv('USER_AGENT', 'Not set')}")
        
        # Get IP and location info
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"   • Hostname: {hostname}")
            print(f"   • Local IP: {local_ip}")
            
            # Get public IP and location
            try:
                response = requests.get('https://ipapi.co/json/', timeout=5)
                if response.status_code == 200:
                    ip_info = response.json()
                    print(f"   • Public IP: {ip_info.get('ip', 'Unknown')}")
                    print(f"   • Location: {ip_info.get('city', 'Unknown')}, {ip_info.get('country_name', 'Unknown')}")
                    print(f"   • ISP: {ip_info.get('org', 'Unknown')}")
            except:
                print(f"   • Public IP: Unable to fetch")
        except Exception as e:
            print(f"   • Network info: Error - {e}")
        
        # Check google-play-scraper version
        try:
            import google_play_scraper
            print(f"   • google-play-scraper version: {google_play_scraper.__version__}")
        except:
            print(f"   • google-play-scraper version: Unknown")
        
        # Retry mechanism for robustness
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Random delay to avoid rate limiting (longer for retries)
                delay = random.uniform(1 + attempt * 2, 3 + attempt * 2)
                print(f"  🔄 Attempt {attempt + 1}/{max_retries}: Waiting {delay:.1f}s before request...")
                time.sleep(delay)
                
                print(f"  📡 Making request to Google Play Store...")
                start_time = time.time()
                
                # Set a more realistic User-Agent to avoid bot detection
                import os
                original_user_agent = os.environ.get('USER_AGENT')
                os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                
                # Try to get more recent reviews by using multiple smaller requests
                print(f"  📡 Making request to Google Play Store (Sort: NEWEST)...")
                print(f"  🕵️  Using User-Agent: {os.environ.get('USER_AGENT')}")
                
                # First try with global settings to get the absolute latest reviews
                print(f"  🌍 Trying global request first...")
                global_result, global_token = reviews(
                    self.app_package,
                    sort=Sort.NEWEST,
                    count=50  # Small sample to check latest
                )
                
                print(f"  📊 Global sample - Latest review: {global_result[0].get('at') if global_result else 'None'}")
                
                # Then get Korean reviews
                result, continuation_token = reviews(
                    self.app_package,
                    lang='ko',
                    country='kr',
                    sort=Sort.NEWEST,
                    count=min(count, 200)  # Limit initial request to get fresher results
                )
                
                print(f"  📊 Korean sample - Latest review: {result[0].get('at') if result else 'None'}")
                
                # If global has much newer reviews, mix them in
                if global_result and result:
                    global_latest = global_result[0].get('at')
                    korean_latest = result[0].get('at')
                    if global_latest and korean_latest and global_latest > korean_latest:
                        print(f"  🔄 Global reviews are newer, mixing in top global reviews...")
                        # Add top 20 global reviews to the beginning
                        result = global_result[:20] + result
                        print(f"  📦 Mixed result count: {len(result)}")
                
                # Restore original User-Agent
                if original_user_agent:
                    os.environ['USER_AGENT'] = original_user_agent
                else:
                    os.environ.pop('USER_AGENT', None)
                
                # If we need more reviews and have a continuation token, get more
                if len(result) < count and continuation_token:
                    print(f"  🔄 Getting additional reviews with continuation token...")
                    additional_result, _ = reviews(
                        self.app_package,
                        lang='ko',
                        country='kr',
                        sort=Sort.NEWEST,
                        count=count - len(result),
                        continuation_token=continuation_token
                    )
                    result.extend(additional_result)
                    print(f"  📦 Total after continuation: {len(result)} reviews")
                
                end_time = time.time()
                print(f"  ⏱️  Request completed in {end_time - start_time:.2f}s")
                print(f"  📦 Raw result count: {len(result)}")
                print(f"  🔗 Continuation token: {'Yes' if continuation_token else 'No'}")
                
                # Debug first few reviews
                if result:
                    print(f"  🔍 Sample review data:")
                    sample_review = result[0]
                    print(f"     • Keys: {list(sample_review.keys())}")
                    print(f"     • Date: {sample_review.get('at')}")
                    print(f"     • Content length: {len(sample_review.get('content', ''))}")
                
                # Sort by date to ensure we have the most recent reviews first
                print(f"  📅 Sorting reviews by date (newest first)...")
                result_with_dates = []
                for review in result:
                    review_date = review.get('at')
                    if review_date:
                        result_with_dates.append((review_date, review))
                
                # Sort by date (newest first)
                result_with_dates.sort(key=lambda x: x[0], reverse=True)
                sorted_result = [review for _, review in result_with_dates]
                
                print(f"  📊 Date range analysis:")
                if sorted_result:
                    newest_date = sorted_result[0].get('at')
                    oldest_date = sorted_result[-1].get('at')
                    print(f"     • Newest: {newest_date}")
                    print(f"     • Oldest: {oldest_date}")
                
                # Standardize format
                standardized_reviews = []
                for i, review in enumerate(sorted_result):
                    # Try multiple version fields
                    version = (review.get('reviewCreatedVersion') or 
                              review.get('appVersionCode') or 
                              review.get('appVersionName') or 
                              'Unknown')
                    
                    standardized_review = {
                        'userName': review.get('userName', 'Anonymous'),
                        'content': review.get('content', ''),
                        'score': review.get('score', 0),
                        'at': review.get('at', datetime.now()),
                        'platform': self.get_platform_name(),
                        'version': version
                    }
                    standardized_reviews.append(standardized_review)
                    
                    # Debug first few processed reviews
                    if i < 5:
                        print(f"     • Review {i+1}: {standardized_review['at']} - Score: {standardized_review['score']}")
                
                print(f"✅ Successfully fetched and sorted {len(standardized_reviews)} reviews from Play Store")
                print(f"🔍 === END PLAY STORE DEBUG ===")
                return standardized_reviews
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                print(f"   Error type: {type(e).__name__}")
                
                # Import traceback for detailed error info
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                
                if attempt == max_retries - 1:
                    print(f"❌ All {max_retries} attempts failed for Play Store reviews")
                    print(f"❌ This might be due to:")
                    print(f"   • Rate limiting from Google Play Store")
                    print(f"   • IP blocking in GitHub Actions environment")
                    print(f"   • Network connectivity issues")
                    print(f"   • Changes in Google Play Store API")
                    print(f"   • Invalid app package: {self.app_package}")
                    print(f"ℹ️  Continuing with App Store reviews only for {target_days} days analysis...")
                    return []
                else:
                    retry_delay = 2 + attempt
                    print(f"  ⏳ Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
        
        return []
    
    def get_platform_name(self) -> str:
        """Get the platform name"""
        return "Google Play Store"

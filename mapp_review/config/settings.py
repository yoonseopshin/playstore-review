"""
Configuration settings for MAPP Review Analysis
"""
import os
import platform


class Config:
    """Configuration settings for the review analysis"""
    
    # App Store settings
    PLAYSTORE_APP_PACKAGE = os.getenv('PLAYSTORE_APP_PACKAGE', 'com.kakao.yellowid')
    APPSTORE_APP_ID = os.getenv('APPSTORE_APP_ID', '990571676')
    
    # Analysis settings
    REVIEW_COUNT = int(os.getenv('REVIEW_COUNT', '1000'))
    DAYS = int(os.getenv('ANALYSIS_DAYS', '7'))
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')
    
    # Font settings by OS
    FONT_SETTINGS = {
        'darwin': {  # macOS
            'name': "AppleGothic",
            'path': "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
        },
        'linux': {
            'name': "NanumGothic", 
            'path': "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        },
        'windows': {
            'name': "Malgun Gothic",
            'path': "C:/Windows/Fonts/malgun.ttf"
        }
    }
    
    @classmethod
    def get_appstore_rss_url(cls):
        """Get Apple App Store RSS URL"""
        return f'https://itunes.apple.com/kr/rss/customerreviews/id={cls.APPSTORE_APP_ID}/sortby=mostrecent/xml'
    
    @classmethod
    def get_font_settings(cls):
        """Get font settings for current OS"""
        current_os = platform.system().lower()
        return cls.FONT_SETTINGS.get(current_os, cls.FONT_SETTINGS['linux'])

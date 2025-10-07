"""Crawlers module for different app stores"""

from .playstore import PlayStoreCrawler
from .appstore import AppStoreCrawler

__all__ = ["PlayStoreCrawler", "AppStoreCrawler"]

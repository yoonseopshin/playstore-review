#!/usr/bin/env python3
"""
MAPP - Mobile App Review Crawler
Complete pipeline execution script
"""

from mapp_review.config import Config
from mapp_review.crawlers import PlayStoreCrawler, AppStoreCrawler
from mapp_review.preprocessing.filter import ReviewFilter
from mapp_review.analysis.intent_classifier import perform_intent_classification
from mapp_review.visualization.charts import ChartGenerator
from mapp_review.reporting.html_generator import HTMLReportGenerator
from mapp_review.reporting.csv_exporter import CSVExporter
from mapp_review.utils.file_utils import FileUtils


def main():
    """Complete MAPP pipeline execution"""
    try:
        print("=== Mobile App Review Crawler (MAPP) Started ===")
        
        # Step 1: Setup environment
        output_dir = FileUtils.setup_environment(Config.OUTPUT_DIR)
        font_name, font_path = FileUtils.get_font_settings()
        
        # Step 2: Initialize components
        playstore_crawler = PlayStoreCrawler(Config.PLAYSTORE_APP_PACKAGE)
        appstore_crawler = AppStoreCrawler(Config.APPSTORE_APP_ID)
        review_filter = ReviewFilter(Config.DAYS)
        chart_generator = ChartGenerator(font_name, font_path)
        html_generator = HTMLReportGenerator()
        csv_exporter = CSVExporter()
        
        # Step 3: Get date range
        start_date, end_date = review_filter.get_date_range()
        print(f"Collecting reviews from the last {Config.DAYS} days ({start_date} ~ {end_date})...")
        
        # Step 4: Collect reviews from both platforms
        print("=== Starting mobile app review collection ===")
        playstore_reviews = playstore_crawler.collect_reviews(Config.REVIEW_COUNT)
        appstore_reviews = appstore_crawler.collect_reviews()
        
        # Combine results
        all_reviews = playstore_reviews + appstore_reviews
        total_count = len(all_reviews)
        playstore_count = len(playstore_reviews)
        appstore_count = len(appstore_reviews)
        
        print(f"\n📊 Collection Summary:")
        print(f"   • Google Play Store: {playstore_count} reviews")
        print(f"   • Apple App Store: {appstore_count} reviews")
        print(f"   • Total: {total_count} reviews")
        
        # Step 5: Process and filter reviews
        df = review_filter.process_reviews(all_reviews)
        if df is None or df.empty:
            print("❌ No reviews to process. Exiting.")
            return
        
        # Step 6: Save initial CSV
        csv_exporter.save_csv(df, output_dir, start_date, end_date)
        
        # Step 7: Generate visualizations
        summary_img = chart_generator.create_summary_chart(df, output_dir, start_date, end_date)
        wc_img = chart_generator.create_wordcloud(df, output_dir, start_date, end_date)
        
        # Step 8: Perform intent classification
        df_with_intents, intent_summary, intent_img = perform_intent_classification(
            df, output_dir, start_date, end_date)
        
        # Step 9: Save CSV with intent analysis
        csv_exporter.save_csv(df_with_intents, output_dir, start_date, end_date, "_with_intents")
        
        # Step 10: Generate HTML report
        html_generator.generate_html_report(
            df_with_intents, output_dir, start_date, end_date, 
            summary_img, wc_img, intent_img, intent_summary)
        
        print(f"\n✓ Mobile app review analysis completed for the last {Config.DAYS} days!")
        print("=== Mobile App Review Analysis Complete ===")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()

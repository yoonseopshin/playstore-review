# Mobile App Review Crawler

**Mobile App Review Crawler** is a comprehensive tool for collecting and analyzing mobile app reviews from multiple app stores.

## 📱 Supported App Stores

| App Store | Data Source | Method | Coverage |
|-----------|-------------|--------|----------|
| 🤖 **Google Play Store** | Web Scraping | `google-play-scraper` library | Full review access |
| 🍎 **Apple App Store** | RSS Feed | Official iTunes RSS API | Recent reviews (~500) |

## 📊 Live Report

[Latest review analysis](https://yoonseopshin.github.io/mapp-review/)

## 🛠 Usage

### Local execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run the crawler
python mapp-review-crawl.py
```

### Automated execution
Runs automatically via GitHub Actions:
- **Schedule**: Every Monday at 9 AM (KST)
- **Manual run**: Click "Run workflow" in GitHub Actions tab

# app-review-final.py

from google_play_scraper import reviews, Sort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from datetime import datetime, timedelta
import os

# ===============================
# 설정: 한글 폰트 경로 및 앱
# ===============================
# Mac
FONT_NAME = "AppleGothic"
WORDCLOUD_FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
# Windows
# FONT_NAME = "Malgun Gothic"
# WORDCLOUD_FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
# Linux
# FONT_NAME = "NanumGothic"
# WORDCLOUD_FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

APP_PACKAGE = 'com.kakao.yellowid'
REVIEW_COUNT = 1000   # 가져올 리뷰 수 (환경에 따라 늘릴 수 있음)
DAYS = 7              # 최근 N일 (요구에 따라 변경)

# 출력 디렉토리(선택)
OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# 날짜 범위
# ===============================
END_DATE = datetime.today().date() - timedelta(days=1)  # 어제 기준
START_DATE = END_DATE - timedelta(days=DAYS - 1)

print(f"최근 {DAYS}일({START_DATE} ~ {END_DATE}) 리뷰 수집 중...")

# ===============================
# 리뷰 수집 (google-play-scraper)
# ===============================
result, _ = reviews(
    APP_PACKAGE,
    lang='ko',
    country='kr',
    count=REVIEW_COUNT,
    sort=Sort.NEWEST
)

# ===============================
# DataFrame 정리
# ===============================
df = pd.DataFrame(result)
# 필요한 컬럼만
df = df[['userName', 'content', 'score', 'at']]
df.rename(columns={'content': 'review'}, inplace=True)
df['at'] = pd.to_datetime(df['at'])

# 최근 N일 필터링
df = df[(df['at'].dt.date >= START_DATE) & (df['at'].dt.date <= END_DATE)]
df.sort_values('at', inplace=True)

if df.empty:
    print(f"최근 {DAYS}일 리뷰가 없습니다.")
    exit(0)

# ===============================
# CSV 저장
# ===============================
csv_file = os.path.join(OUT_DIR, f"user_reviews_summary_{START_DATE}_{END_DATE}.csv")
df.to_csv(csv_file, index=False, encoding='utf-8-sig')
print(f"CSV 저장 완료: {csv_file} (총 {len(df)}건)")

# ===============================
# 시계열 시각화 (이미지)
# ===============================
# 날짜 컬럼 생성
df['date'] = df['at'].dt.date
daily_score = df.groupby('date')['score'].mean().reset_index()
daily_count = df.groupby('date').size().reset_index(name='count')

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
plt.plot(daily_score['date'], daily_score['score'], marker='o', label='평균 별점')
plt.bar(daily_count['date'], daily_count['count'], alpha=0.3, label='리뷰 수', color='skyblue')

plt.xlabel("날짜", fontname=FONT_NAME)
plt.ylabel("평균 별점 / 리뷰 수", fontname=FONT_NAME)
plt.title(f"최근 {DAYS}일 리뷰 요약 ({START_DATE} ~ {END_DATE})", fontname=FONT_NAME)
plt.legend(prop={"family": FONT_NAME})
plt.xticks(rotation=45)
plt.tight_layout()

summary_img = os.path.join(OUT_DIR, f"review_summary_{START_DATE}_{END_DATE}.png")
plt.savefig(summary_img)
plt.close()
print(f"리뷰 요약 시각화 이미지 저장 완료: {summary_img}")

# ===============================
# 워드클라우드 생성 (이미지)
# ===============================
text = " ".join(df['review'].astype(str))
text = re.sub(r"[^가-힣a-zA-Z\s]", " ", text)

wordcloud = WordCloud(
    font_path=WORDCLOUD_FONT_PATH,
    width=1200,
    height=600,
    background_color='white',
    max_words=500
).generate(text)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()

wc_img = os.path.join(OUT_DIR, f"review_wordcloud_{START_DATE}_{END_DATE}.png")
plt.savefig(wc_img)
plt.close()
print(f"워드클라우드 이미지 저장 완료: {wc_img}")

# ===============================
# HTML 보고서 생성
# - date 컬럼 제거 (중복)
# - at은 문자열(ISO)로 변환해 JS에서 안정적으로 정렬
# - score와 at 헤더에 class="sortable" 추가
# ===============================
df_for_html = df.drop(columns=['date']).copy()
# at 컬럼을 사람이 읽기 쉬운 포맷으로 (JS에서 Date로 파싱 가능)
df_for_html['at'] = df_for_html['at'].dt.strftime('%Y-%m-%d %H:%M:%S')

table_html = df_for_html.to_html(index=False, escape=False)
# pandas가 만든 <th>태그에 class 추가 (score, at)
table_html = table_html.replace('<th>score</th>', '<th class="sortable">score</th>')
table_html = table_html.replace('<th>at</th>', '<th class="sortable">at</th>')

total_count = len(df_for_html)

# 안전한 플레이스홀더 방식으로 HTML 구성
html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>PlayStore Review Report ([[START]] ~ [[END]])</title>
<style>
body { font-family: AppleGothic, "Malgun Gothic", "NanumGothic", sans-serif; margin: 20px; }
h1 { text-align: center; }
img { display: block; margin: 20px auto; max-width: 90%; }
table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
th { background-color: #f7f7f7; }
th.sortable { cursor: pointer; position: relative; padding-right: 24px; }
th.sortable::after { content: '⇅'; position: absolute; right: 8px; font-size: 12px; color: #888; }
th.sortable[aria-sort="ascending"]::after { content: '▲'; color: #333; }
th.sortable[aria-sort="descending"]::after { content: '▼'; color: #333; }
.preview { max-width: 1200px; margin: 0 auto; }
.count { font-weight: bold; margin: 10px 0; }
</style>
</head>
<body>
<div class="preview">
<h1>Review Report ([[START]] ~ [[END]])</h1>

<h2>리뷰 요약 시각화</h2>
<img src="[[SUMMARY_IMG]]" alt="리뷰 요약">

<h2>워드클라우드</h2>
<img src="[[WC_IMG]]" alt="워드클라우드">

<h2>리뷰 상세 <span class="count">(총 [[COUNT]]건)</span></h2>
[[TABLE_HTML]]
</div>
<style>
th.sortable {
    cursor: pointer;
    position: relative;
    padding-right: 24px;
}
th.sortable::after { content: '⇅'; position: absolute; right: 8px; font-size: 12px; color: #888; }
th.sortable[aria-sort="ascending"]::after { content: '▲'; color: #333; }
th.sortable[aria-sort="descending"]::after { content: '▼'; color: #333; }

/* 리뷰 toggle 스타일 */
td.review {
    max-width: 600px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.review-short { display: inline; }
.review-full { display: none; }
.toggle-btn {
    color: blue;
    cursor: pointer;
    font-size: 12px;
    margin-left: 4px;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const table = document.querySelector('table');
    if (!table) return;
    const tbody = table.tBodies[0];
    if (!tbody) return;

    const numRegex = /^\s*-?[\d,]+(\.\d+)?\s*$/;

    // === sortable 기능 ===
    const headers = Array.from(table.querySelectorAll('th.sortable'));
    headers.forEach(th => {
        let asc = true; // 초기 오름차순
        th.addEventListener('click', function() {
            const colIndex = th.cellIndex;
            const rows = Array.from(tbody.querySelectorAll('tr'));

            const headerText = (th.textContent || '').trim().toLowerCase();

            rows.sort((a, b) => {
                const aText = (a.children[colIndex]?.textContent || '').trim();
                const bText = (b.children[colIndex]?.textContent || '').trim();

                if(headerText === 'at') {
                    const da = Date.parse(aText) || 0;
                    const db = Date.parse(bText) || 0;
                    return (da - db) * (asc ? 1 : -1);
                } else if(headerText === 'score') {
                    const aClean = aText.replace(/,/g,'');
                    const bClean = bText.replace(/,/g,'');
                    const aIsNum = numRegex.test(aText);
                    const bIsNum = numRegex.test(bText);
                    if(aIsNum && bIsNum){
                        const na = parseFloat(aClean);
                        const nb = parseFloat(bClean);
                        return (na - nb) * (asc ? 1 : -1);
                    }
                    return aText.localeCompare(bText, 'ko') * (asc ? 1 : -1);
                } else if(headerText === 'review') {
                    // 글자 수 기준 정렬
                    return (aText.length - bText.length) * (asc ? 1 : -1);
                } else {
                    return aText.localeCompare(bText, 'ko') * (asc ? 1 : -1);
                }
            });

            // 정렬된 순서로 tbody에 재삽입
            rows.forEach(r => tbody.appendChild(r));

            // 아이콘 표시
            th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
            headers.forEach(h => { if(h!==th) h.removeAttribute('aria-sort'); });

            // 다음 클릭에서 반대 방향
            asc = !asc;
        });
    });
});
</script>

</body>
</html>
"""

# 치환
html_content = (html_template
                .replace('[[START]]', str(START_DATE))
                .replace('[[END]]', str(END_DATE))
                .replace('[[SUMMARY_IMG]]', os.path.basename(summary_img))
                .replace('[[WC_IMG]]', os.path.basename(wc_img))
                .replace('[[TABLE_HTML]]', table_html)
                .replace('[[COUNT]]', str(total_count))
               )

html_file = os.path.join(OUT_DIR, f"review_report_{START_DATE}_{END_DATE}.html")
with open(html_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML 보고서 생성 완료: {html_file}")
print(f"최근 {DAYS}일간 리뷰 요약 파일 생성 완료!")

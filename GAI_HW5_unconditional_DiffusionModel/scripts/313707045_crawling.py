import requests
from bs4 import BeautifulSoup
import json
import time
from collections import Counter
import requests
import argparse
import re
import random

PTT_URL = "https://www.ptt.cc"
COOKIES = {'over18': '1'}
HEADERS = {'User-Agent': 'Mozilla/5.0'}
START_INDEX = 3350 
MAX_INDEX = 5000
START_HREF = "/bbs/Beauty/M.1704040318.A.E87.html"  # 2023 "/bbs/Beauty/M.1672503968.A.5B5.html"

def get_page(index):
    url = f"{PTT_URL}/bbs/Beauty/index{index}.html"
    res = requests.get(url, cookies=COOKIES, headers=HEADERS)
    if res.status_code != 200:
        return None
    return BeautifulSoup(res.text, 'html.parser')

def crawl():
    index = START_INDEX
    found_start = False
    visited = set()
    prev_month = None

    with open("articles.jsonl", "w", encoding="utf-8") as f_all, \
         open("popular_articles.jsonl", "w", encoding="utf-8") as f_pop:

        while index <= MAX_INDEX:
            print(f"🔎 正在爬 index{index}.html")
            soup = get_page(index)
            if soup is None:
                print("⚠️ 無法取得頁面")
                break

            entries = soup.select('div.r-ent')

            for entry in entries:
                a_tag = entry.select_one("div.title a")
                if not a_tag:
                    continue  # 文章已刪除

                href = a_tag["href"]
                title = a_tag.text.strip()

                # skip 公告
                if '[公告]' in title or 'Fw:[公告]' in title:
                    continue
                if not title:
                    continue

                if not found_start:
                    if href == START_HREF:
                        print(f"🎯 找到起點：{href}")
                        found_start = True
                    else:
                        continue  # 還沒找到第一篇文章

                date_div = entry.select_one("div.meta div.date")
                if not date_div:
                    continue
                date_str = date_div.text.strip()  # e.g., '1/01'
                try:
                    month = int(date_str.split('/')[0])
                    day = int(date_str.split('/')[1])
                    mmdd = f"{month:02d}{day:02d}"
                except:
                    continue

                if prev_month == 12 and month == 1:
                    print("🛑 偵測到跨年（12月 ➜ 1月），結束爬蟲")
                    return
                prev_month = month

                url = PTT_URL + href
                if url in visited:
                    continue
                visited.add(url)

                article = {
                    "date": mmdd,
                    "title": title,
                    "url": url
                }
                f_all.write(json.dumps(article, ensure_ascii=False) + "\n")

                # 檢查推爆
                nrec = entry.select_one("div.nrec span")
                if nrec and nrec.text.strip() == "爆":
                    f_pop.write(json.dumps(article, ensure_ascii=False) + "\n")

            index += 1
            time.sleep(random.uniform(0.2, 0.5))

def parse_push_comments(url):
    try:
        res = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=5)
        if res.status_code != 200:
            return []
        soup = BeautifulSoup(res.text, 'html.parser')
        pushes = soup.select('div.push')
        result = []
        for p in pushes:
            tag = p.select_one('span.push-tag')
            user = p.select_one('span.push-userid')
            if not tag or not user:
                continue
            tag_text = tag.text.strip()
            user_id = user.text.strip()
            if tag_text == '推':
                result.append(('push', user_id))
            elif tag_text == '噓':
                result.append(('boo', user_id))
        return result
    except Exception as e:
        print(f"[Error] parsing pushes from {url}: {e}")
        return []

def ord_sort_key(user_id):
    return tuple(ord(c) for c in user_id)

def top10(counter):
    return sorted(
        [{"user_id": uid, "count": count} for uid, count in counter.items()],
        key=lambda x: (-x["count"], tuple([-ord(c) for c in x["user_id"]]))
    )[:10]


def write_push_output(result, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('{\n')
        for i, key in enumerate(['push', 'boo']):
            f.write(f'  "{key}": {{\n')
            f.write(f'    "total": {result[key]["total"]},\n')
            f.write(f'    "top10": [\n')
            for j, item in enumerate(result[key]["top10"]):
                json_line = json.dumps(item, ensure_ascii=False, separators=(',', ': '))
                comma = ',' if j < len(result[key]["top10"]) - 1 else ''
                f.write(f'      {json_line}{comma}\n')
            f.write('    ]\n')
            f.write('  }')
            if i == 0:
                f.write(',\n')
            else:
                f.write('\n')
        f.write('}\n')

def push(start_date, end_date):
    push_counter = Counter()
    boo_counter = Counter()
    push_total = 0
    boo_total = 0

    with open('articles.jsonl', 'r', encoding='utf-8') as f:
        articles = [json.loads(line) for line in f]

    for article in articles:
        date = article['date']
        if start_date <= date <= end_date:
            url = article['url']
            print(f"分析留言：{url}")
            comments = parse_push_comments(url)
            for ctype, uid in comments:
                if ctype == 'push':
                    push_counter[uid] += 1
                    push_total += 1
                elif ctype == 'boo':
                    boo_counter[uid] += 1
                    boo_total += 1

    result = {
        "push": {
            "total": push_total,
            "top10": top10(push_counter)
        },
        "boo": {
            "total": boo_total,
            "top10": top10(boo_counter)
        }
    }

    filename = f"push_{start_date}_{end_date}.json"
    write_push_output(result, filename)
    print(f"✅ 統計完成，輸出至 {filename}")


def extract_image_urls(soup):
    urls = set()

    # 內文
    for link in soup.select('a'):
        href = link.get('href', '')
        if re.match(r'^https?://.*\.(jpg|jpeg|png)$', href, re.IGNORECASE): #'^https?://.*\.(jpg|jpeg|png|gif)$'
            urls.add(href)

    # 留言
    for push in soup.select('div.push'):
        content = push.get_text()
        matches = re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif)', content, flags=re.IGNORECASE)
        urls.update(matches)

    return list(urls)

def popular(start_date, end_date):
    with open('popular_articles.jsonl', 'r', encoding='utf-8') as f:
        articles = [json.loads(line) for line in f]

    selected = [a for a in articles if start_date <= a['date'] <= end_date]
    image_urls = []

    for article in selected:
        url = article['url']
        print(f"🔍 處理推爆文章：{url}")
        try:
            res = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=5)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, 'html.parser')
            imgs = extract_image_urls(soup)
            image_urls.extend(imgs)
        except Exception as e:
            print(f"⚠️ 解析失敗：{url} | {e}")
            continue

    result = {
        "number_of_popular_articles": len(selected),
        "image_urls": image_urls  # 重複可保留
    }

    filename = f"popular_{start_date}_{end_date}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ 已輸出至 {filename}")

def extract_main_content(raw_text):

    lines = raw_text.split('\n')
    content_lines = []
    in_body = False
    for line in lines:
        if line.startswith('作者'):
            in_body = True
        elif '※ 發信站' in line:
            break
        elif in_body:
            content_lines.append(line)
    return '\n'.join(content_lines) if in_body else None

def keyword(start_date, end_date, word):
    with open('articles.jsonl', 'r', encoding='utf-8') as f:
        all_articles = [json.loads(line) for line in f]
    with open('popular_articles.jsonl', 'r', encoding='utf-8') as f:
        popular_urls = {json.loads(line)['url'] for line in f}

    selected_articles = [a for a in all_articles if start_date <= a['date'] <= end_date]
    matched_images = []
    popular_count = 0

    for article in selected_articles:
        url = article['url']
        try:
            res = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=5)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, 'html.parser')
            main_div = soup.select_one('#main-content')
            if not main_div:
                continue

            # [1] 擷取內文段落
            full_text = main_div.text
            content = extract_main_content(full_text)

            # [2] 擷取 meta 區塊資訊（例如作者、標題、時間）
            meta_info = "\n".join(span.text for span in soup.select('.article-meta-value'))

            # [3] 一併搜尋 keyword
            search_space = f"{meta_info}\n{content}"
            if not word or word not in search_space:
                continue

            if url in popular_urls:
                popular_count += 1

            matched_images.extend(extract_image_urls(soup))

        except Exception as e:
            print(f"⚠️ 解析失敗：{url} | {e}")
            continue

    result = {
        "image_urls": matched_images
    }

    filename = f"keyword_{start_date}_{end_date}_{word}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ 關鍵字統計完成，輸出至 {filename}")

def all_images(start_date, end_date):
    """
    擷取 articles.jsonl 中指定日期範圍內所有文章的內文圖片網址（無論推爆與否）
    """
    with open('articles.jsonl', 'r', encoding='utf-8') as f:
        all_articles = [json.loads(line) for line in f]

    selected_articles = [a for a in all_articles if start_date <= a['date'] <= end_date]
    image_urls = []

    for article in selected_articles:
        url = article['url']
        try:
            res = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=5)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, 'html.parser')
            imgs = extract_image_urls(soup)
            image_urls.extend(imgs)
            print(f"處理文章：{url} | 圖片數量：{len(imgs)}")
        except Exception as e:
            print(f"⚠️ 解析失敗：{url} | {e}")
            continue

    result = {
        "number_of_articles": len(selected_articles),
        "image_urls": image_urls
    }

    filename = f"all_images_{start_date}_{end_date}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ 已輸出所有內文圖片至 {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("crawl")

    push_parser = subparsers.add_parser("push")
    push_parser.add_argument("start_date", type=str)
    push_parser.add_argument("end_date", type=str)


    popular_parser = subparsers.add_parser("popular")
    popular_parser.add_argument("start_date", type=str)
    popular_parser.add_argument("end_date", type=str)

    keyword_parser = subparsers.add_parser("keyword")
    keyword_parser.add_argument("start_date", type=str)
    keyword_parser.add_argument("end_date", type=str)
    keyword_parser.add_argument("keyword", type=str)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("start_date", type=str)
    all_parser.add_argument("end_date", type=str)


    args = parser.parse_args()
    if args.command == "crawl":
        crawl()
    elif args.command == "push":
        push(args.start_date, args.end_date)
    elif args.command == "popular":
        popular(args.start_date, args.end_date)
    elif args.command == "keyword":
        keyword(args.start_date, args.end_date, args.keyword)
    elif args.command == "all":
        all_images(args.start_date, args.end_date)
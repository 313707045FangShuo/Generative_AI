# PTT 表特板爬蟲與分析工具

本專案是一個 Python 腳本，用於爬取 PTT 表特板的文章，並提供三種分析功能：統計推噓文、抓取爆文圖片、以及搜尋關鍵字。

## 功能

  * **`crawl`**：爬取 PTT 表特板，儲存所有文章資訊 (`articles.jsonl`) 與爆文資訊 (`popular_articles.jsonl`)。
  * **`push`**：分析指定日期範圍內的文章，統計「推」和「噓」的總數與前 10 名使用者。
  * **`popular`**：抓取指定日期範圍內，「爆文」中的所有圖片 URL。
  * **`keyword`**：抓取指定日期範圍內，符合「關鍵字」的文章（搜尋標題、作者、內文）中的所有圖片 URL。

## 安裝

安裝所需的 Python 套件：

```bash
pip install requests beautifulsoup4
```

## 設定

在**執行 `crawl` 之前**，你必須手動修改腳本頂部的全域變數：

  * `START_INDEX`：PTT 列表頁的起始 index (例如 `3630`)。
  * `START_HREF`：你希望開始爬取的第一篇文章的 URL (例如 `"/bbs/Beauty/M.1704040318.A.E87.html"`)。

`crawl` 會從 `START_HREF` 這篇文章開始，爬到偵測跨年 (12 月 -\> 1 月) 時自動停止。

## 使用方式

本工具使用 `argparse` 透過指令列執行。

### 1\. 爬取文章 (crawl)

**這是第一步**，必須先執行此指令來產生資料檔。

```bash
python crawler.py crawl
```

> 🔎 正在爬 index3630.html
> 🎯 找到起點：/bbs/Beauty/M.1704040318.A.E87.html
> ...
> 🛑 偵測到跨年（12月 ➜ 1月），結束爬蟲

### 2\. 分析推文 (push)

分析指定日期範圍 (MMDD) 內的推文。

```bash
# 範例：分析 1 月 1 日到 1 月 31 日的推文
python crawler.py push 0101 0131
```

  * **輸出**: `push_0101_0131.json`

### 3\. 抓取爆文圖片 (popular)

抓取指定日期範圍 (MMDD) 內的爆文圖片。

```bash
# 範例：抓取 10 月 1 日到 10 月 31 日的爆文圖片
python crawler.py popular 1001 1031
```

  * **輸出**: `popular_1001_1031.json`

### 4\. 關鍵字搜尋 (keyword)

搜尋指定日期範圍 (MMDD) 內，文章內容或 metadata (作者、標題) 包含特定關鍵字的文章圖片。

```bash
# 範例：搜尋 3 月 1 日到 3 月 15 日，包含「Yuri」的文章
python crawler.py keyword 0301 0315 Yuri
```

  * **輸出**: `keyword_0301_0315_Yuri.json`

### 作業評分:
[![Scoring](GAI_HW4_PTT_Beauty_Crawler/competetion_result.png)](/GAI_HW4_PTT_Beauty_Crawler/competetion_result.png)


# HW1：Few-shot Prompt Engineering with Gemini API

本專案為課堂 Kaggle 作業（HW1），目標是利用 **Large Language Model (LLM)**（Google Gemini 2.0）進行多選題回答任務。
系統使用 **few-shot prompting** 技術，從已知樣本中自動挑選相似題目作為範例，提升模型回答準確率，並將結果自動生成提交檔案。

---

## 專案介紹

作業目標：

* 使用 Google Gemini API 建立自動化多選題解答系統
* 實作 **2-shot prompting**：根據題目類別 (`task`)，從 sample 集取 2 題作為示範
* 控制 **API 使用速率**（Tokens per minute、Requests per minute）以避免超額
* 產生 Kaggle 可提交格式的結果檔案（`mmlu_predictions.csv`）

---

## 專案架構

```
HW_LLM/
├── HW1_prompt/
│   ├── dataset/
│   │   ├── mmlu_sample.csv      # few-shot 範例資料
│   │   ├── mmlu_submit.csv      # 要預測的測資
│   │   └── mmlu_predictions.csv # 模型輸出結果（自動生成）
│   ├── genmini_api.txt          # Google Gemini API Key (本地檔案)
│   └── hw-1-prompt-engineering.zip # 原始壓縮資料
└── ML_HW1_GeminiFewShot.ipynb   # 或 main.py, 主程式
```

---

## API 設定

1. 建立文字檔 `genmini_api.txt`，內容為：

   ```
   your_google_api_key_here
   ```

2. 程式會自動讀取：

   ```python
   file_path = "/mnt/sda1/shuof/HW_LLM/HW1_prompt/genmini_api.txt"
   os.environ["GOOGLE_API_KEY"] = google_api_key
   ```

---

## 執行流程

### 1. 解壓資料集

程式會自動將 `hw-1-prompt-engineering.zip` 解壓至：

```
HW_LLM/HW1_prompt/
```

### 2. 啟動 Gemini LLM

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
```

### 3. Few-shot Prompt 組成

每次根據題目類別 (`task`) 從 sample.csv 中選出 2 題相同任務作為提示：

```python
few_shot_samples = df_sample[df_sample["task"] == task].sample(n=2, random_state=42)
```

Prompt 結構：

```
Task: Answer multiple-choice questions...
Example:
Question: ...
A) ...
B) ...
C) ...
D) ...
Answer: B

Now answer the following question:
Question: ...
A) ...
B) ...
C) ...
D) ...
Answer:
```
---

## 結果輸出

程式最終會輸出結果至：

```
HW_LLM/HW1_prompt/dataset/mmlu_predictions.csv
```

內容格式如下：

| ID | target |
| -- | ------ |
| 0  | C      |
| 1  | A      |
| 2  | D      |

---

## 執行

```bash
python hw1_prompt_gemini.py
```

輸出範例：

```
Processing question 1
"ID": 0, "target": C
Processing question 2
"ID": 1, "target": B
Results saved to HW_LLM/HW1_prompt/dataset/mmlu_predictions.csv
```

---

## 使用技術

* Few-shot Learning（2-shot）
* 自動化 prompt 組合
* 使用 Google Gemini 2.0 Flash 模型
* Token 與 Request 控制機制

[![Competetion Result: 26th of 130](competetion_result.png)](https://github.com/313707045FangShuo/Generative_AI/blob/90b160a1d740484c48a73eca6bc0d2c558a76270/GAI_HW1_Prompt/competetion_result.png)

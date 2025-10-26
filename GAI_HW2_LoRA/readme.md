# LLM HW2 ---- QLoRA

[hackmd version url link](https://hackmd.io/@41eOHZkcS3y8iYNWvrdBig/HJcYteWCkg)

本專案旨在使用 Qwen2-7B 模型，透過 QLoRA 與 FSDP 進行微調，訓練目標為根據論文的 Introduction 自動生成 Abstract。

## 專案結構
- dataset：於 E3 下載。
- `313707045.ipynb`：主要訓練與推論 Notebook，包含資料處理、模型載入、微調與摘要生成等流程。
- `qlora-fsdp.yaml`：Axolotl 訓練框架的設定檔，定義微調參數、LoRA 設定，資料集路徑要改成資料前處理完的 training data path
```
datasets:
  - path: /mnt/sda1/shuof/HW_LLM/HW2/HW2_Introduction/data/axolotl_format_llama.jsonl
```

### 微調模型設定
- base model：Qwen/Qwen2-7B
- bitsandbytes：4-bit 避免 OOM
- Adapter：QLoRA

### LoRA 設定：
- r=32
- alpha=64
- dropout=0.05
- Epoch：4
- lora_target_linear: true
- max sequence len：2048 tokens
- Micro batch：1
- Gradient Accumulation：4
- bf16: auto
- fp16: false
- tf32: true
- Optimizer：adamw_torch_fused + cosine learning rate scheduler

## 資料前處理
- 轉換 train、test 資料成 axolotl 格式
    - axolotl_format_llama.jsonl
    - test_axolotl_format_llama.json
```
instruction_text = 
"You are a professional researcher writing for a top-tier academic conference.
Based on the introduction, generate a concise and logically structured academic abstract based solely on the following introduction.
Ensure the abstract reflects the motivation, methodology, and contributions as implied in the introduction.
Do not use any information beyond what is provided."
```

## 訓練模型
- Input：axolotl_format_llama.jsonl

1. 下載 requirements.txt
2. 調整訓練與驗證資料格式： `.jsonl`
3. 執行 `subprocess.run(["axolotl", "train", "qlora-fsdp.yaml"])`，或在 bash 執行 `accelerate launch -m axolotl train qlora-fsdp.yaml`

## testing data 預測
- Input：test_axolotl_format_llama.json
- Output：output_QWEN2.json
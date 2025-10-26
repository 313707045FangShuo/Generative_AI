---
title: HW5_unconditional_DiffusionModel

---

# HW5-Human Face Generation

## 使用方式
環境設定：`requirements.txt`
1. 執行 `python 313707045_crawling.py all`，儲存所需年分 PTT 表特版中照片的 url 清單 all_images_0101_1231.json
2. 利用 `face_recognition.ipynb` 來辨認及下載 all_images_0101_1231.json 的 url 中之清晰人物正臉
3. 執行 `stable_diffusion_model.py` 來訓練模型 (需先確認 `config.yaml` 中的 im_path 有更改成 dataset)
4. 利用模型訓練出的權重生成 10,000 張照片至 generated_images/

## 內容說明
### 訓練資料下載
#### A. 爬圖片網址
`python 313707045_crawling.py` 的 all 功能會擷取 指定日期區間(一年內)的所有文章內文與留言區中的圖片網址，並統一儲存為一個 JSON 檔 (若需多年份則需執行多次)。

#### B. 臉部辨識及圖片下載
使用人臉偵測模型（MTCNN）對爬下來連結的圖片進行篩選，挑選出「清晰、正臉」的照片，resize 成 64×64 並下載為 .png 格式之檔案，作為後續 Diffusion 模型訓練資料集。
清晰、正臉之判斷依據
```
    # 1. 眼睛是否水平（避免側臉或歪頭）
    if slope > 0.08:  # 原本是 0.15，現在更嚴格
        return False

    # 2. 左右眼距離與臉寬一致性（避免歪頭導致的錯位）
    eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    box_w = face_box[2] - face_box[0]
    ratio = eye_dist / (box_w + 1e-6)
    if ratio < 0.35 or ratio > 0.6:  # 避免太窄或太開（側臉會異常）
        return False

    # 3. 臉大小過小不要
    if box_w < 80 or (face_box[3] - face_box[1]) < 80:
        return False

    # 4. 模糊度檢查
    gray = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 50:
        return False
```

### 訓練方式

#### A. 初始化及設定
讀取 config.yaml 配置檔並設定參數：
- Image Size：64x64
- Channel：3 (RGB)
- Diffusion 步驟：num_timesteps = 1000
- Optimizer：AdamW
- Learning Rate：0.0001
- Scheduler：CosineAnnealingLR
- Batch Size：4

#### B. 資料集切分
使用 train_test_split 切成訓練 (80%) 與驗證 (20%) 資料，圖片預處理：
```
transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
```

#### C. 模型架構建立
模型整體架構是基於 U-Net 架構 的 Pixel-space Diffusion Model。使用自定義的 Unet class：
- 包含 DownBlock → MidBlock → UpBlock
- 每層含 Residual Block + Attention + Time Embedding
- 搭配 Sinusoidal Time Embedding 作為 timestep 輸入

#### D. Diffusion Scheduler
`LinearNoiseScheduler`：模擬 DDPM 的 noise (Gaussian distribution) 的加入與去噪過程：
- 加噪（forward process）：x_t = sqrt(α) * x_0 + sqrt(1-α) * ε
- 去噪（reverse sampling）：x_{t-1} 根據 noise_pred 估算 x_0 再往回推


#### E. 訓練 Loop

1. 隨機選擇 timestep t
2. 為圖片加上高斯噪音：noisy_imgs = add_noise(imgs, noise, t)
3. 用模型預測噪音：noise_pred = model(noisy_imgs, t)
4. 計算 loss = MSE(noise_pred, true_noise)
5. 使用 AMP + Gradient Accumulation 優化訓練效率

#### F. 儲存模型權重
每 100 epoch 存一次 checkpoint，並儲存 loss 最低的權重為 best_checkpoint。

## 資料夾結構
```
hw5_313707045.zip
├── data/                        # 訓練 dataset
├── generated_images/            # 共 10000 張 PNG 格式的生成圖片 (64x64)
│   ├── gen_00000.png
│   ├── ...
│   └── gen_09999.png
├── model/                       # 模型權重
│   ├── checkpoint_epoch_200.pt  # best model checkpoint
├── scripts/                     # 爬蟲、前處理、模型訓練等程式碼與相關設定檔
│   ├── 313707045_crawling.py    # 沿用上次 HW4 的爬蟲，加入 all 指令
│   ├── config.yaml              # model 之架構與參數設定
│   ├── face_recognition.ipynb   # 辨認並儲存爬蟲爬下來的 url 中之人臉
│   ├── generate.py              # 根據 model checkpoint 生成人臉
│   └── stable_diffusion_model.py# 訓練模型
├── README.md
└── requirements.txt
```


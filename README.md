# 😷 口罩偵測系統

使用 MobileNetV2 遷移學習與 OpenCV 實現的即時口罩偵測系統。

## 展示效果

- 🟢 綠色框 → 有戴口罩
- 🔴 紅色框 → 未戴口罩

## 環境需求

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- scikit-learn
- Matplotlib

## 安裝步驟

1. 複製專案
   git clone https://github.com/你的帳號/mask-detection.git
   cd mask-detection

2. 安裝套件
   pip install tensorflow opencv-python numpy matplotlib scikit-learn

3. 下載資料集
   從 Kaggle 下載：https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
   將圖片放入以下資料夾：
   dataset/with_mask/
   dataset/without_mask/

## 使用方式

### 訓練模型

    python3 train.py

### 即時偵測

    python3 detect.py
    按 Q 離開

## 模型資訊

- 基礎模型：MobileNetV2（遷移學習）
- 訓練準確率：99.95%
- 驗證準確率：99.60%

## 專案結構

    mask-detection/
    ├── dataset/
    │   ├── with_mask/
    │   └── without_mask/
    ├── model/
    │   └── mask_model.h5
    ├── train.py
    ├── detect.py
    └── README.md

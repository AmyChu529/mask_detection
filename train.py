import matplotlib.pyplot as plt
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# ---- 設定參數 ----
IMG_SIZE = 224  # 圖片統一縮放成 224x224
DATASET_PATH = "dataset"

# ---- 載入圖片 ----
data = []
labels = []

categories = ["with_mask", "without_mask"]

for label, category in enumerate(categories):
    folder = os.path.join(DATASET_PATH, category)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)

        # 讀取圖片
        img = cv2.imread(img_path)
        if img is None:
            continue  # 跳過讀取失敗的圖片

        # 縮放成統一大小
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # 轉換顏色 BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data.append(img)
        labels.append(label)  # with_mask=0, without_mask=1

# ---- 轉成 numpy 陣列 ----
data = np.array(data, dtype="float32") / 255.0  # 正規化到 0~1
labels = np.array(labels)

print(f"總共載入圖片：{len(data)} 張")
print(f"圖片形狀：{data[0].shape}")

# ---- 切分訓練集 / 測試集 ----
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print(f"訓練集：{len(X_train)} 張")
print(f"測試集：{len(X_test)} 張")

# ---- 載入 MobileNetV2 預訓練模型 ----
baseModel = MobileNetV2(
    weights="imagenet",   # 使用 ImageNet 預訓練權重
    include_top=False,    # 不包含最後的分類層
    input_tensor=Input(shape=(224, 224, 3))
)

# 凍結基礎模型的權重（不讓它被訓練）
baseModel.trainable = False

# ---- 在後面加上我們自己的分類層 ----
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)  # 2類：有/無口罩

# ---- 組合成完整模型 ----
model = Model(inputs=baseModel.input, outputs=headModel)

# ---- 編譯模型 ----
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())
print("模型建立成功！")

# ---- 設定訓練參數 ----
EPOCHS = 20      # 訓練 20 輪
BATCH_SIZE = 32  # 每次餵 32 張圖片

# ---- 開始訓練 ----
print("開始訓練模型...")

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test)
)

print("訓練完成！")

# ---- 儲存模型 ----
os.makedirs("model", exist_ok=True)
model.save("model/mask_model.h5")
print("模型已儲存到 model/mask_model.h5")

# ---- 畫出訓練結果 ----
plt.figure(figsize=(12, 4))

# 準確率圖
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="訓練準確率")
plt.plot(history.history["val_accuracy"], label="驗證準確率")
plt.title("準確率")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# 損失圖
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="訓練損失")
plt.plot(history.history["val_loss"], label="驗證損失")
plt.title("損失")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_result.png")
print("訓練結果圖已儲存！")

from tensorflow.keras.models import load_model
import numpy as np
import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ---- 載入訓練好的模型 ----
model = load_model("model/mask_model.h5")
print("模型載入成功！")

# ---- 載入人臉偵測器 ----
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---- 開啟鏡頭 ----
cap = cv2.VideoCapture(0)
print("鏡頭開啟！按 Q 離開")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # 轉成灰階做人臉偵測
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # 擷取臉部區域
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        # 預測
        prediction = model.predict(face, verbose=0)
        label_index = np.argmax(prediction)
        confidence = prediction[0][label_index] * 100

        # 0 = with_mask, 1 = without_mask
        if label_index == 0:
            label = f"Mask ON ({confidence:.1f}%)"
            color = (0, 255, 0)   # 綠色
        else:
            label = f"No Mask ({confidence:.1f}%)"
            color = (0, 0, 255)   # 紅色

        # 畫框和文字
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Mask Detection", frame)

    # 按 Q 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("程式結束！")

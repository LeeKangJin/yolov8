import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO

# 1. YOLO 모델 불러오기
model = YOLO("runs/detect/moster_detector/weights/best.pt")  # ← 학습된 best.pt 경로

# 2. 스크린샷 찍기
screenshot = pyautogui.screenshot()
screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# 3. YOLO로 추론
results = model(screenshot)

# 4. 추론 결과에서 바운딩 박스 좌표 추출
for box in results[0].boxes:
    cls = int(box.cls[0])  # 클래스 번호
    conf = float(box.conf[0])  # 신뢰도
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표

    print(f"Class: {cls}, Confidence: {conf:.2f}, BBox: ({x1}, {y1}, {x2}, {y2})")

# 5. 시각화 (선택)
results[0].show()
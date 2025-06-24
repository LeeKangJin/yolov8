from ultralytics import YOLO

# 사전학습된 yolov8n 모델 불러오기 (최초 실행 시 자동 다운로드)
model = YOLO("yolov8n.pt")  # 또는 yolov8s.pt, yolov8m.pt 등

# 학습 실행
model.train(
    data="dataset/data.yaml",  # YAML 파일 경로
    epochs=50,                        # 학습 반복 횟수
    imgsz=640,                        # 입력 이미지 크기
    batch=16,                         # 배치 사이즈
    name="monster_detector",          # 결과 저장 폴더 이름 (runs/detect/animal_detector)
    workers=4                        # 데이터 로딩 스레드 수
)
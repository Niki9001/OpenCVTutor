from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
from util import get_car

mot_tracker = Sort()
vehicles = [2, 3, 5, 7]

# 加载模型
coco_model = YOLO('yolov8n.pt')  # 请确保路径正确
license_plate_detector = YOLO('license_plate_detector.pt')  # 请确保路径正确

# 打开视频
cap = cv2.VideoCapture('nscc.mp4')

# 获取视频的基本信息
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 初始化VideoWriter对象
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 进行车辆检测
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])
            # 绘制车辆检测的边界框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # 跟踪车辆（如果需要）
    track_ids = mot_tracker.update(np.asarray(detections_))

    # 检测车牌（这里假设license_plate_detector的输出逻辑和coco_model相同）
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        # 绘制车牌检测的边界框
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # 写入处理后的帧到输出视频
    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
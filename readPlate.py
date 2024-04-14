from ultralytics import YOLO
import cv2
import numpy as np
from imageProcess import process_license_plate
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv,apply_perspective_transform



results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
cap = cv2.VideoCapture(r"FInalDark2.mp4")

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # 调整坐标至局部坐标
                lp_x1, lp_y1, lp_x2, lp_y2 = 0, 0, int(x2) - int(x1), int(y2) - int(y1)
                license_plate_transformed = apply_perspective_transform(license_plate_crop, lp_x1, lp_y1, lp_x2, lp_y2)


                # process license plate
                #license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                # 使用自适应阈值进行二值化
                #license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255,
                                                                  #cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                                 # 11, 4)
               # # pts = np.array([
               #      [x1, y1],  # 左上角
               #      [x2, y1],  # 右上角
               #      [x2, y2],  # 右下角
               #      [x1, y2]  # 左下角
               #  ], dtype="float32")
                #license_plate_transformed = apply_perspective_transform(license_plate_crop, x1, y1, x2, y2)

                license_plate_crop_thresh = process_license_plate(license_plate_transformed)

                #kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
                #license_plate_crop_thresh = cv2.morphologyEx(license_plate_crop_thresh1, cv2.MORPH_CLOSE, kernelX, iterations=3)


                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
write_csv(results, 'test4.csv')
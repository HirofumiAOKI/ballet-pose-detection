import cv2
import sys
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

model_path = "pose_landmarker.task"
if not os.path.exists(model_path):
    print("モデルをダウンロード中...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, model_path)
    print("ダウンロード完了！")

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]

def draw_landmarks_on_image(cv_image, result):
    annotated = cv_image.copy()
    h, w = cv_image.shape[:2]
    for landmarks in result.pose_landmarks:
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
        for connection in POSE_CONNECTIONS:
            start = landmarks[connection[0]]
            end = landmarks[connection[1]]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return annotated

def save_csv(result, csv_path, image_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'person', 'landmark', 'x', 'y', 'z', 'visibility'])
        for person_idx, landmarks in enumerate(result.pose_landmarks):
            for idx, lm in enumerate(landmarks):
                name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"landmark_{idx}"
                writer.writerow([
                    os.path.basename(image_path),
                    person_idx,
                    name,
                    round(lm.x, 4),
                    round(lm.y, 4),
                    round(lm.z, 4),
                    round(lm.visibility, 4)
                ])
    print(f"CSVを保存しました: {csv_path}")

def analyze_image(image_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    image = mp.Image.create_from_file(image_path)
    cv_image = cv2.imread(image_path)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)

    if result.pose_landmarks:
        print(f"{len(result.pose_landmarks)}人の骨格を検出しました！")
        annotated = draw_landmarks_on_image(cv_image, result)
        csv_path = image_path.rsplit('.', 1)[0] + '_landmarks.csv'
        save_csv(result, csv_path, image_path)
    else:
        print("骨格が検出できませんでした")
        annotated = cv_image

    output_path = image_path.rsplit('.', 1)[0] + '_result.jpg'
    cv2.imwrite(output_path, annotated)
    print(f"画像を保存しました: {output_path}")

    cv2.imshow('Result', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python pose_detection.py 画像のパス")
    else:
        analyze_image(sys.argv[1])
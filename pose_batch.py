import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import glob

model_path = "pose_landmarker.task"

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

def draw_landmarks(cv_image, result):
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

def process_folder(folder_path):
    output_folder = os.path.join(folder_path, "results")
    os.makedirs(output_folder, exist_ok=True)
    all_csv_path = os.path.join(output_folder, "all_landmarks.csv")

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    image_files.sort()

    if not image_files:
        print("画像ファイルが見つかりませんでした")
        return

    print(f"{len(image_files)}枚の画像を処理します...")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options)

    header = ['image', 'person', 'landmark', 'x', 'y', 'z', 'visibility']

    with open(all_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for image_path in image_files:
            filename = os.path.basename(image_path)
            print(f"処理中: {filename}")
            try:
                image = mp.Image.create_from_file(image_path)
                cv_image = cv2.imread(image_path)
                with vision.PoseLandmarker.create_from_options(options) as landmarker:
                    result = landmarker.detect(image)
                if result.pose_landmarks:
                    print(f"  -> {len(result.pose_landmarks)}人検出")
                    annotated = draw_landmarks(cv_image, result)
                    out_img = os.path.join(output_folder, filename.rsplit('.', 1)[0] + '_result.jpg')
                    cv2.imwrite(out_img, annotated)
                    for person_idx, landmarks in enumerate(result.pose_landmarks):
                        for idx, lm in enumerate(landmarks):
                            name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"landmark_{idx}"
                            writer.writerow([filename, person_idx, name, round(lm.x, 4), round(lm.y, 4), round(lm.z, 4), round(lm.visibility, 4)])
                else:
                    print(f"  -> 骨格検出できませんでした")
            except Exception as e:
                print(f"  -> エラー: {e}")

    print(f"\n完了！")
    print(f"結果画像: {output_folder}")
    print(f"全データCSV: {all_csv_path}")

if __name__ == "__main__":
    folder_path = r"C:\Users\hiro\Documents\260429_tdac"
    process_folder(folder_path)
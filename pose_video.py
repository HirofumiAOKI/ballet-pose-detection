import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

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

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー：動画が開けません: {video_path}")
        return

    # 動画情報取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"動画情報: {width}x{height}, {fps}fps, 総フレーム数: {total_frames}")

    # 出力ファイルパス
    base = video_path.rsplit('.', 1)[0]
    output_video_path = base + '_result.mp4'
    csv_path = base + '_landmarks.csv'

    # 動画出力設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options)

    frame_count = 0

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'time_sec', 'person', 'landmark', 'x', 'y', 'z', 'visibility'])

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                time_sec = round(frame_count / fps, 3)

                # 進捗表示（100フレームごと）
                if frame_count % 100 == 0:
                    print(f"処理中: {frame_count}/{total_frames}フレーム ({int(frame_count/total_frames*100)}%)")

                # MediaPipe用に変換
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                result = landmarker.detect(mp_image)

                if result.pose_landmarks:
                    annotated = draw_landmarks(frame, result)
                    for person_idx, landmarks in enumerate(result.pose_landmarks):
                        for idx, lm in enumerate(landmarks):
                            name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"landmark_{idx}"
                            writer.writerow([
                                frame_count, time_sec, person_idx, name,
                                round(lm.x, 4), round(lm.y, 4),
                                round(lm.z, 4), round(lm.visibility, 4)
                            ])
                else:
                    annotated = frame

                out.write(annotated)

    cap.release()
    out.release()
    print(f"\n完了！")
    print(f"結果動画: {output_video_path}")
    print(f"CSVデータ: {csv_path}")

if __name__ == "__main__":
    video_path = r"C:\Users\hiro\Documents\260428_tdac\260428_tdac_BW.mp4"
    process_video(video_path)
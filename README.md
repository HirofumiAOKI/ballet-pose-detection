# ballet-pose-detection

バレエおよびダンス研究のためのMediaPipeベースの姿勢検出ツール

A MediaPipe-based pose detection tool for ballet and dance research.

---

## 概要 / Overview

このツールはGoogleのMediaPipeを使用して、バレエ・ダンス・振り付け研究のための骨格検出を行います。

- 画像1枚の骨格検出
- フォルダ内の画像一括処理
- 動画の骨格検出

---

## 動作環境 / Requirements

- Windows 11
- Python 3.12
- MediaPipe 0.10.35
- OpenCV

---

## インストール方法 / Installation

### 1. Minicondaのインストール
以下からMinicondaをダウンロードしてインストール：
https://docs.conda.io/en/latest/miniconda.html

### 2. 専用環境の作成
```bash
conda create -n mediapipe_env python=3.12
conda activate mediapipe_env
pip install mediapipe
```

### 3. MediaPipeモデルのダウンロード
以下のURLからモデルファイルをダウンロードし、スクリプトと同じフォルダに保存：

https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

ファイル名を `pose_landmarker.task` に変更してください。

---

## 使い方 / Usage

### 事前準備（毎回必要）
```bash
conda activate mediapipe_env
cd C:\Users\your_username
```

---

### 1. 画像1枚の骨格検出：pose_detection.py

```bash
python pose_detection.py "画像のパス"
```

**出力：**
- `画像名_result.jpg`：骨格描画済み画像
- `画像名_landmarks.csv`：33個のキーポイント座標データ

---

### 2. 画像一括処理：pose_batch.py

スクリプト内の `folder_path` を対象フォルダに変更してから実行：

```bash
python pose_batch.py
```

**出力（resultsフォルダ内）：**
- 各画像の骨格描画済み画像
- `all_landmarks.csv`：全画像の座標データ

---

### 3. 動画の骨格検出：pose_video.py

スクリプト内の `video_path` を対象動画に変更してから実行：

```bash
python pose_video.py
```

**出力：**
- `動画名_result.mp4`：骨格描画済み動画
- `動画名_landmarks.csv`：全フレームの座標データ

---

## CSVデータの説明 / CSV Data Description

| 列名 | 内容 |
|---|---|
| image / frame | ファイル名またはフレーム番号 |
| person | 検出された人物番号 |
| landmark | キーポイント名（nose, left_shoulderなど） |
| x | 横方向の位置（0〜1） |
| y | 縦方向の位置（0〜1） |
| z | 奥行き |
| visibility | 検出の確信度（0〜1） |

---

## 検出されるキーポイント / Keypoints

MediaPipeは以下の33個のキーポイントを検出します：

| 番号 | 部位 |
|---|---|
| 0 | nose（鼻） |
| 11-12 | left/right shoulder（肩） |
| 13-14 | left/right elbow（肘） |
| 15-16 | left/right wrist（手首） |
| 23-24 | left/right hip（腰） |
| 25-26 | left/right knee（膝） |
| 27-28 | left/right ankle（足首） |
| 31-32 | left/right foot index（つま先） |

---

## 注意事項 / Notes

- ファイル名に日本語が含まれる場合、読み込みエラーになる場合があります
- RTX 50シリーズGPUではCPUモードで動作します
- 長い動画の処理には時間がかかります

---

## 参考文献 / References

- MediaPipe: https://mediapipe.dev
- Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. arXiv:1906.08172

---

## 著者 / Author

HirofumiAOKI

---

## ライセンス / License

MIT License

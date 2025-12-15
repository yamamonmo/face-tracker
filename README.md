# 😉face-tracker

> **大阪ハイテクノロジー専門学校　授業内プロジェクト**  
> Webカメラで顔を認識し、サーボモーターでトラッキングを行うアプリ

---
## 🚀 クイックスタート(RaspberryPi)

### 1. リポジトリのクローン

```bash
git clone https://github.com/yamamoto/face-tracker
cd face-tracker
```


### 2. 仮想環境の作成・有効化

```bash
# Windows (Git Bash)
py -3.11 -m venv .venv
source .venv/Scripts/activate

# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3.環境構築(pigpiod)
サーボモーターの制御精度向上のため pigpio デーモンを使用します。
```bash
sudo apt update
sudo apt upgrade
sudo apt install pigpio
which pigpiod　　（でpigpiodがあるか確認）
sudo pigpiod
```
起動する際は毎回 sudo pigpiodを実行してください

### 4.依存ライブラリのインストール
```bash
pip install opencv-python gpiozero
```
※ OpenCVでエラーが出る場合は sudo apt install libatlas-base-dev 等を試してください。

---
## 🔌 ハードウェア構成・配線

- **Webカメラ**
  - 任意のUSBポートに接続
- **左右サーボモーター**
  - 信号線(PWM): GPIO 18
  - 電源(VCC): 5V
  - GND: GND
- **上下サーボモーター**
  - 信号線(PWM): GPIO 12
  - 電源(VCC): 5V
  - GND: GND

---

## 主な機能
- 🏍️**サーボ機能**
- 🎥**カメラ機能**

---

## ⚙️使用技術

| 区分 | 使用技術 |
|------|-----------|
| **マイコン** | RaspberryPi4 Model-B|
| **カメラ** | USBカメラ |
| **サーボモーター** | TowerPro MG90S ２個|
| **言語** | Python 3.11 |
| **画像処理・顔認識** | OpenCV |
| **PWM制御** | gpiozero - Servo |
| **顔認識モデル** | Haar-Cascade |

--- 
## 🧱 ディレクトリ構成

```
FACE-TRACKER/
├── app/
|   ├── camera.py                   # カメラ機能
|   ├── servo.py                    # サーボ機能
|   ├── main.py                     # 機能統合
|   ├── test_servo.py               # サーボ動作テスト
|   ├── yolo.py                     # 顔認識YOLO版（未完成）
|   └── facial_recognition.py       # 顔認識YOLO版（未完成）
├── face-tracker.py                 # レガシーapp
└── README.md
```
---

## 担当

| 役割 | 担当者 |
| :--- | :--- |
| **リーダー** | シャイン |
| **3DCAD作成** | シャイン |
| **カメラ処理** | 山本、米澤 |
| **サーボ制御** | 米澤 |
| **機能統合** | シャイン、米澤 |
| **ProtoPedia記事** | 山本 |
| **README** | 米澤 |

### YOLOインストール方法
```bash
psudo apt update
sudo apt install python3-pip -y
pip install -U pip
pip install ultralytics[export]
sudo reboot

```

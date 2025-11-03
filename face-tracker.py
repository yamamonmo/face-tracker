import cv2

# 使用するカメラを指定 (0は通常、内蔵または最初に見つかったUSBカメラ)
CAM_ID = 0

# 顔認識用の学習済みモデルファイル（環境に合わせてパスを調整してください）
CASCADE_FILE = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

def main():
    # カメラをキャプチャ
    cap = cv2.VideoCapture(CAM_ID)
    
    # Haar Cascade分類器をロード
    cascade = cv2.CascadeClassifier(CASCADE_FILE)
    
    print("顔認識を開始します... (終了するには'q'キーを押してください)")

    while True:
        # カメラから1フレーム読み込む
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームを読み込めませんでした。")
            break

        # 処理を高速化するため、画像をグレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔を検出
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 検出した顔を矩形で囲む
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 結果をウィンドウに表示
        cv2.imshow('Face Recognition', frame)

        # 'q'キーが押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 後処理
    cap.release()
    cv2.destroyAllWindows()
    print("顔認識を終了します。")

if __name__ == '__main__':
    main()
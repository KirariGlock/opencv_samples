import cv2
from datetime import datetime as dt

def capture_camera(mirror=True, size=None):
    """Capture video from camera"""
    # カメラをキャプチャする
    cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号

    while True:
        # retは画像を取得成功フラグ
        ret, frame = cap.read()

        # 鏡のように映るか否か
        if mirror is True:
            frame = frame[:,::-1]

        # フレームをリサイズ
        # sizeは例えば(800, 600)
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        # フレームを表示する
        cv2.imshow('camera capture', frame)

        k = cv2.waitKey(1) # 1msec待つ
        if k == 27: # ESCキーで終了
            break
        elif k == ord('p'): # pキーで画像保存
            datetime_str = create_now_datetime_str()
            file_name = output_file_path + datetime_str + 'output.png'
            cv2.imwrite(file_name, frame)
            print('capture ok ! file_name = ' + file_name)

    # キャプチャを解放する
    cap.release()
    cv2.destroyAllWindows()

def create_now_datetime_str():
    """現在日時（ミリ秒も含む）を文字列で返します。"""
    return dt.now().strftime('%Y%m%d%H%M%S') + "%04d" % (dt.now().microsecond // 1000)

output_file_path = '../output/';
capture_camera()

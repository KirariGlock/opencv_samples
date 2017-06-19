# -*- coding: utf-8 -*-

# 画像から顔認識をするサンプル
# 参考：http://www.takunoko.com/blog/pythonで遊んでみる-part1-opencvで顔認識/

import cv2

cascade_path = "../haarcascade_frontalface_alt.xml"
image_path = "../picture/trump.png"
replace_face_image_path = "../picture/morikubo_face.png"

# color white
color = (255, 255, 255)


# 画像の読み込み
image = cv2.imread(image_path)
replace_face_image = cv2.imread(replace_face_image_path)

# グレースケール変換
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

cascade = cv2.CascadeClassifier(cascade_path)

# 顔認識の実行
facerect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

if len(facerect) > 0:
    # 検出した顔を囲む矩形の作成
    for rect in facerect:
        # rect[0:2] = x,y の座標 rect[2:4] = 高さ・幅
        cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
else:
    print("no face")

# 認識結果の表示
cv2.imshow("detected.jpg", image)

# 何かキーが押されたら終了
while(1):
    if cv2.waitKey(10) > 0:
        cv2.destroyAllWindows()
        break

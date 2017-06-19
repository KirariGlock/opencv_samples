# -*- coding: utf-8 -*-
# 参考：http://qiita.com/k_sui_14/items/bb9dc8395da85400e518

import cv2
import numpy as np
from PIL import Image

def resize_image(image, height, width):
    u"""渡された画像をheightとwidthの大きい方に合わせてリサイズします。
    アスペクト比の変更はしません。
    """
    # 元々のサイズを取得
    org_height, org_width = image.shape[:2]

    # レートが大きい方のサイズに合わせてリサイズ
    height_ratio = float(height)/org_height
    width_ratio = float(width)/org_width

    if height_ratio > width_ratio:
        resized = cv2.resize(image,(int(org_height*height_ratio),int(org_width*height_ratio)))
    else:
        resized = cv2.resize(image,(int(org_height*width_ratio),int(org_width*width_ratio)))

    return resized

def overlayOnPart(src_image, overlay_image, posX, posY):

    # オーバレイ画像のサイズを取得
    ol_height, ol_width = overlay_image.shape[:2]

    # OpenCVの画像データをPILに変換
    #　BGRAからRGBAへ変換
    src_image_RGBA = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    overlay_image_RGBA = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)

    #　PILに変換
    src_image_PIL = Image.fromarray(src_image_RGBA)
    overlay_image_PIL = Image.fromarray(overlay_image_RGBA)

    # 合成のため、RGBAモードに変更
    src_image_PIL = src_image_PIL.convert('RGBA')
    overlay_image_PIL = overlay_image_PIL.convert('RGBA')

    # 同じ大きさの透過キャンパスを用意
    tmp = Image.new('RGBA', src_image_PIL.size, (255, 255,255, 0))
    # 用意したキャンパスに上書き
    tmp.paste(overlay_image_PIL, (posX, posY), overlay_image_PIL)
    # オリジナルとキャンパスを合成して保存
    result = Image.alpha_composite(src_image_PIL, tmp)

    return  cv2.cvtColor(np.asarray(result), cv2.COLOR_RGBA2BGRA)




# 認識対象ファイルの読み込み
image_path = "../picture/trump2.png"
image = cv2.imread(image_path)

# 上書きする画像の読み込み
ol_imgae_path = "../picture/morikubo_face.png"
ol_image = cv2.imread(ol_imgae_path,cv2.IMREAD_UNCHANGED)   # アルファチャンネル(透過)も読みこむようにIMREAD_INCHANGEDを指定

# グレースケールに変換
#image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔認識用特徴量のファイル指定
cascade_path = "../haarcascade_frontalface_alt.xml"
# カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

# 顔認識の実行
minsize = (int(image.shape[0] * 0.1), int(image.shape[1] * 0.1))
facerecog = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=minsize)


if len(facerecog) > 0:
    # 認識した顔全てに画像を上書きする
    result_array = np.array([])
    for rect in facerecog:

        # 認識結果を表示
        print ("認識結果")
        print ("(x,y)=(" + str(rect[0]) + "," + str(rect[1])+ ")" + \
            "  高さ："+str(rect[2]) + \
            "  幅："+str(rect[3]))

        # 認識範囲にあわせて画像をリサイズ
        resized_ol_image = resize_image(ol_image, rect[2], rect[3])

        # 上書きした画像の作成
        image = overlayOnPart(image, resized_ol_image, rect[0], rect[1])

# 認識結果の出力
cv2.imwrite("result.png", image)

import cv2
import dlib
import sys
import numpy as np

from re import A
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import time

import math

import mss

# スクリーンショット
def SCT(bbox):
    with mss.mss() as sct:
        img = sct.grab(bbox)
    return img

while True:
    # 開始時間
    start = time.time()

    # [TO DO]ウィンドウサイズを抽出して絶対値で指定する ※上のタブも入らないように！！！
    original_sct = SCT((1400, 200, 1900, 450))
    original = np.asarray(original_sct)

    original = cv2.cvtColor(original, cv2.COLOR_RGBA2RGB)

    cv2.imshow('img', original)

    # 終了時間
    end = time.time()

    # Time elapsed
    seconds = end - start
    print("経過時間: {0} seconds".format(seconds))

    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)
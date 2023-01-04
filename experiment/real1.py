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
  
    # fps計測用：開始時間
    start = time.time()

    # pc2においてピン止めして右モニターの右端に寄せる
    # (Rectangle.left, Rectangle.top, Rectangle.right, Rectangle.bottom)
    original_sct = SCT((980, 110, 1600, 460))
    original = np.asarray(original_sct)

    original = cv2.cvtColor(original, cv2.COLOR_RGBA2RGB)

    cv2.imshow('img', original)

    # fps計測用：終了時間
    end = time.time()

    # fps計測用：経過時間
    seconds = end - start
    print("経過時間: {0} seconds".format(seconds))

    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)
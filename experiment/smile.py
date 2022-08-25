import cv2
import dlib
import sys
import numpy as np

from re import A
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import torch    # Install PyTorch first: https://pytorch.org/get-started/locally/
    from img_utils_pytorch import (
        mls_rigid_deformation as mls_rigid_deformation_pt,
    )
    # device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
except ImportError as e:
    print(e)

from PIL import Image

import time

import math

#Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

np.random.seed(1234)

import mss

def SCT(bbox):
    with mss.mss() as sct:
        img = sct.grab(bbox)
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

while True:
    # 開始時間
    start = time.time()

    # [TO DO?]ウィンドウサイズを抽出して絶対値で指定する ※上のタブも入らないように！！！
    # この数値はpcAにおいてピン止めして被験者を右に寄せたウィンドウを右のモニターの右上端に寄せる
    # (Rectangle.left, Rectangle.top, Rectangle.right, Rectangle.bottom)
    original = SCT((980, 135, 1600, 450))
    image = torch.from_numpy(original).to(device)

    # [TO DO?]指定したらコメントアウト
    height, width, _ = image.shape

    # detect faces
    faces = detector(original)

    #例外処理　顔が検出されなかった時
    if len(faces) == 0:
        print('no faces')
        aug1 = original

    for face in faces:

        # landmark
        dlib_shape = landmark_predictor(original,face)
        landmark = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        for j in range(68):
            landmark[j][0], landmark[j][1] = landmark[j][1], landmark[j][0]
        
        x = landmark[57][0] - landmark[66][0]

        p_array = [landmark[48], landmark[49], landmark[50], landmark[51], landmark[52], landmark[53], 
            landmark[54], landmark[55], landmark[56], landmark[57], landmark[58], landmark[59]]
        
        # [TO DO?]ウィンドウサイズを抽出して絶対値で指定する
        p_array_append = p_array.append
        for m in range(math.ceil(height/100)):
            p_array_append([width, m * 100])

        # [TO DO?]ウィンドウサイズを抽出して絶対値で指定する
        for n in range(math.ceil(width/100)):
            p_array_append([n * 100, height])

        a = landmark[48] + np.array([-3*x/4, -2*x/5])
        b = landmark[49] + np.array([-3*x/8, -x/10])
        c = landmark[50] + np.array([-3*x/8, 0])
        d = landmark[51] + np.array([-3*x/8, 0])
        e = landmark[52] + np.array([-3*x/8, 0])
        f = landmark[53] + np.array([-3*x/8, x/10])
        g = landmark[54] + np.array([-3*x/4, 2*x/5])
        h = landmark[55] + np.array([-3*x/10, 0])
        i = landmark[56] + np.array([-3*x/10, 0])
        j = landmark[57] + np.array([-3*x/10, 0])
        k = landmark[58] + np.array([-3*x/10, 0])
        l = landmark[59] + np.array([-3*x/10, 0])
                
        q_array = [a, b, c, d, e, f, g, h, i, j, k, l]

        # [TO DO?]ウィンドウサイズを抽出して絶対値で指定する
        q_array_append = q_array.append
        for m in range(math.ceil(height/100)):
            q_array_append([width, m * 100])

        # [TO DO?]ウィンドウサイズを抽出して絶対値で指定する
        for n in range(math.ceil(width/100)):
            q_array_append([n * 100, height])

        p1 = torch.from_numpy(np.array(p_array)).to(device)
        q1 = torch.from_numpy(np.array(q_array)).to(device)

        # Define deformation grid
        # [TO DO?]ウィンドウサイズを抽出して絶対値で指定する
        gridX = torch.arange(width, dtype=torch.int16).to(device)
        gridY = torch.arange(height, dtype=torch.int16).to(device)
        vy, vx = torch.meshgrid(gridX, gridY)
        # !!! Pay attention !!!: the shape of returned tensors are different between numpy.meshgrid and torch.meshgrid
        vy, vx = vy.transpose(0, 1), vx.transpose(0, 1)

        rigid1 = mls_rigid_deformation_pt(vy, vx, p1, q1, alpha=1)
        aug1 = torch.ones_like(image).to(device)
        aug1[vx.long(), vy.long()] = image[tuple(rigid1)]
        aug1 = aug1.to('cpu').detach().numpy()

    cv2.imshow('img', aug1)
    cv2.imshow("original", original)

    # 終了時間
    end = time.time()

    # Time elapsed
    seconds = end - start
    print("経過時間: {0} seconds".format(seconds))

    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)
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

# import math

#Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

np.random.seed(1234)

cap = cv2.VideoCapture(0)


while True:
    # 開始時間
    start = time.time()
    
    # read frame buffer from video
    ret, original = cap.read()
    original = cv2.resize(original, (int(original.shape[1]*0.5), int(original.shape[0]*0.5)))
    image = torch.from_numpy(original).to(device)
    # height, width, _ = image.shape

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue

    # print(height, width)

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

        # for s in landmark:
        #     cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # cv2.circle(image, center=landmark[48], radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        for j in range(68):
            landmark[j][0], landmark[j][1] = landmark[j][1], landmark[j][0]

        
        x = landmark[57][0] - landmark[66][0]

        p_array = [landmark[48], landmark[49], landmark[50], landmark[51], landmark[52], landmark[53], 
            landmark[54], landmark[55], landmark[56], landmark[57], landmark[58], landmark[59]]
        
        p_array_append = p_array.append
        for m in range(4):
            p_array_append([240, m * 100])


        for n in range(3):
            p_array_append([n * 100, 320])


        a = landmark[48] + np.array([-x/2, -2*x/5])
        b = landmark[49] + np.array([-x/4, -x/10])
        c = landmark[50] + np.array([-x/4, 0])
        d = landmark[51] + np.array([-x/4, 0])
        e = landmark[52] + np.array([-x/4, 0])
        f = landmark[53] + np.array([-x/4, x/10])
        g = landmark[54] + np.array([-x/2, 2*x/5])
        h = landmark[55] + np.array([-x/5, 0])
        i = landmark[56] + np.array([-x/5, 0])
        j = landmark[57] + np.array([-x/5, 0])
        k = landmark[58] + np.array([-x/5, 0])
        l = landmark[59] + np.array([-x/5, 0])
                
        q_array = [a, b, c, d, e, f, g, h, i, j, k, l]

        q_array_append = q_array.append
        for m in range(4):
            q_array_append([240, m * 100])

        for n in range(3):
            q_array_append([n * 100, 320])

        p1 = torch.from_numpy(np.array(p_array)).to(device)
        q1 = torch.from_numpy(np.array(q_array)).to(device)

        # Define deformation grid
        gridX = torch.arange(320, dtype=torch.int16).to(device)
        gridY = torch.arange(240, dtype=torch.int16).to(device)
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
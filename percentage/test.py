from re import A
import cv2
import numpy as np

import openpyxl

# PC3の場合
# book = openpyxl.load_workbook("C:\\Users\\Misaki Sato\\Desktop\\result\\smile_percentage.xlsx")
book = openpyxl.load_workbook("D:\\Misaki Sato\\master\\result\\smile_percentage.xlsx")
sheet = book["test2"]

name = "hina"

# capture = cv2.VideoCapture("C:\\Users\\Misaki Sato\\Desktop\\recording\\hon\\mkv\\%s.mkv" % name)
capture = cv2.VideoCapture("D:\\Misaki Sato\\master\\recording\\yobi\\%s.mp4" % name)
capture.set(3,640)# 320 320 640 720
capture.set(4,480)# 180 240  360 405

video_frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) # フレーム数を取得する
video_fps = capture.get(cv2.CAP_PROP_FPS)                 # フレームレートを取得する
video_len_sec = video_frame_count / video_fps         # 長さ（秒）を計算する

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('./haarcascade_smile.xml')

i = 1
j = 1
left_sum = 0
left_average = 0
right_sum = 0
right_sum = 0
frame = 0
left_face_count = 0
right_face_count = 0
left_smile_count = 0
right_smile_count = 0
left_roi = 0
right_roi = 0

while capture.isOpened():
    ret, img = capture.read()

    if ret == True:
        # img = cv2.flip(img,1)#鏡表示にするため．
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
        for (x,y,w,h) in faces:
            _, width, _ = img.shape
            # 左の人の顔の計算
            if x < width/2:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0),2) # blue
                #Gray画像から，顔領域を切り出す．
                roi_gray = gray[y:y+h, x:x+w]
                left_face_count += 1

                #サイズを縮小
                roi_gray = cv2.resize(roi_gray,(100,100))
                #cv2.imshow("roi_gray",roi_gray) #確認のためサイズ統一させた画像を表示

                # 輝度で規格化
                lmin = roi_gray.min() #輝度の最小値
                lmax = roi_gray.max() #輝度の最大値
                # print("lmax:" + str(lmax))
                # print("lmin:" + str(lmin))
                for index1, item1 in enumerate(roi_gray):
                    for index2, item2 in enumerate(item1) :
                        roi_gray[index1][index2] = int((item2 - lmin)/(lmax-lmin) * item2)
                        left_roi += float((item2 - lmin)/(lmax-lmin))
                        # print(left_roi)
                # cv2.imshow("roi_gray12",roi_gray)  #確認のため輝度を正規化した画像を表示

                smiles= smile_cascade.detectMultiScale(roi_gray,scaleFactor= 1.2, minNeighbors=0, minSize=(20, 20))#笑顔識別
                if len(smiles) >0 : # 笑顔領域がなければ以下の処理を飛ばす．#if len(smiles) <=0 : continue でもよい．その場合以下はインデント不要
                    left_smile_count += 1
                    # サイズを考慮した笑顔認識
                    smile_neighbors = len(smiles)
                    # print("smile_neighbors=",smile_neighbors) #確認のため認識した近傍矩形数を出力
                    left_intensityZeroOne = smile_neighbors
                    # print(intensityZeroOne) #確認のため強度を出力
                    left_sum += left_intensityZeroOne
                    left_average = left_sum / i
                    i += 1
                    for(sx,sy,sw,sh) in smiles:
                        cv2.circle(img,(int(x+(sx+sw/2)*w/100),int(y+(sy+sh/2)*h/100)),int(sw/2*w/100), (255*(1.0-left_intensityZeroOne), 0, 255*left_intensityZeroOne),2)#red
                        
            # 右の人の顔の計算
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0),2) # blue
                #Gray画像から，顔領域を切り出す．
                roi_gray = gray[y:y+h, x:x+w] 
                right_face_count += 1

                #サイズを縮小
                roi_gray = cv2.resize(roi_gray,(100,100))
                #cv2.imshow("roi_gray",roi_gray) #確認のためサイズ統一させた画像を表示

                # 輝度で規格化
                lmin = roi_gray.min() #輝度の最小値
                lmax = roi_gray.max() #輝度の最大値
                for index1, item1 in enumerate(roi_gray):
                    for index2, item2 in enumerate(item1) :
                        roi_gray[index1][index2] = int((item2 - lmin)/(lmax-lmin) * item2)
                        right_roi += float((item2 - lmin)/(lmax-lmin))
                        # print(right_roi)
                # cv2.imshow("roi_gray2",roi_gray)  #確認のため輝度を正規化した画像を表示

                smiles= smile_cascade.detectMultiScale(roi_gray,scaleFactor= 1.2, minNeighbors=0, minSize=(20, 20))#笑顔識別
                if len(smiles) >0 : # 笑顔領域がなければ以下の処理を飛ばす．#if len(smiles) <=0 : continue でもよい．その場合以下はインデント不要
                    right_smile_count += 1
                    # サイズを考慮した笑顔認識
                    smile_neighbors = len(smiles)
                    # print("smile_neighbors=",smile_neighbors) #確認のため認識した近傍矩形数を出力
                    right_intensityZeroOne = smile_neighbors
                    # print(intensityZeroOne) #確認のため強度を出力
                    right_sum += right_intensityZeroOne
                    right_average = right_sum / j
                    j += 1
                    for(sx,sy,sw,sh) in smiles:
                        cv2.circle(img,(int(x+(sx+sw/2)*w/100),int(y+(sy+sh/2)*h/100)),int(sw/2*w/100), (255*(1.0-right_intensityZeroOne), 0, 255*right_intensityZeroOne),2)#red                

        cv2.imshow('img',img)

        # key Operation
        key = cv2.waitKey(5) 
        if key ==27 or key ==ord('q'): #escまたはeキーで終了
            print("left:" + str(left_average))
            print("right:" + str(right_average))
            maxRow = sheet.max_row + 1
            print(maxRow)
            sheet.cell(maxRow,1).value = name
            sheet.cell(maxRow,2).value = left_average
            sheet.cell(maxRow,3).value = right_average
            book.save("D:\\Misaki Sato\\master\\result\\smile_percentage.xlsx")
            break
        
        frame += 1

    else:
        maxRow = sheet.max_row + 1
        sheet.cell(maxRow,1).value = name # ファイル名
        sheet.cell(maxRow,2).value = left_average # 左の人の笑顔度合
        sheet.cell(maxRow,3).value = left_face_count*100/frame # 左の人の顔認識率
        sheet.cell(maxRow,4).value = left_smile_count*100/frame # 左の人の笑顔認識率
        sheet.cell(maxRow,6).value = left_roi/(frame*10000) # 左の正規化輝度平均
        sheet.cell(maxRow,7).value = right_average # 右の人の笑顔度合い
        sheet.cell(maxRow,8).value = right_face_count*100/frame # 右の人の顔認識率
        sheet.cell(maxRow,9).value = right_smile_count*100/frame  # 右の人の笑顔認識率
        sheet.cell(maxRow,11).value = right_roi/(frame*10000) # 右の正規化輝度平均
        sheet.cell(maxRow,12).value = video_len_sec # 動画秒数
        sheet.cell(maxRow,13).value = frame/video_len_sec # fps(計算)
        sheet.cell(maxRow,14).value = video_fps # fps(正式)
        # book.save("C:\\Users\\Misaki Sato\\Desktop\\result\\smile_percentage.xlsx")
        book.save("D:\\Misaki Sato\\master\\result\\smile_percentage.xlsx")

        break

capture.release()
cv2.destroyAllWindows()

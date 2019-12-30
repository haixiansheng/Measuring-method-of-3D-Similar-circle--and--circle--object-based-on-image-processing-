
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from matplotlib import pyplot as plt
import numpy as np
# import argparse
import imutils
import cv2

def reshape_image(image):
    '''归一化图片尺寸：短边800，长边不超过1600，短边800，长边超过1600以长边1600为主'''
    width,height=image.shape[1],image.shape[0]
    min_len=width
    scale=width*1.0/800
    new_width=800
    new_height=int(height/scale)
    if new_height>1600:
        new_height=1600
        scale=height*1.0/1600
        new_width=int(width/scale)
    out=cv2.resize(image,(int(new_width),int(new_height)))
    return out
#QR-code
# WIDTH = 80.7

#USB
WIDTH = 24.99
H0 = 42.24

egg_image = cv2.imread(r"D:\images\long_dist_h42.24.jpg")
egg_image = reshape_image(egg_image)
# egg_image =cv2.resize(egg_image,(int(egg_image.shape[0]/RESIZE_RATE),int(egg_image.shape[1]/RESIZE_RATE)))
gray_img = cv2.cvtColor(egg_image,cv2.COLOR_BGR2GRAY)
gray_img = cv2.GaussianBlur(gray_img,(7,7),0)

# bw_img1 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,5)
# bw_img2  = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,5)
edged = cv2.Canny(gray_img, 50, 10) #cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# cv2.namedWindow("edged",0)
# cv2.resizeWindow("edged",480,640)
# cv2.imshow("edged",bw_img1)
cv2.imshow("edged", edged)
# cv2.waitKey(0)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None
# result_img = cv2.drawContours(egg_image, cnts, -1, (0, 0, 255), 2)

# 单独循环轮廓

dst = np.zeros((edged.shape))
dst = np.array(dst,np.uint8)
rad = []
for c in cnts:
    # 如果轮廓不够大，请忽略它
    if cv2.contourArea(c) < 1800:
        continue
    # 计算轮廓的旋转边界框
    dst = cv2.drawContours(dst,[c],-1, 255, thickness=cv2.FILLED)
    maxdist = 0
    dist = 0
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            dist = cv2.pointPolygonTest(c, (j, i), True)
            if (dist > maxdist):
                maxdist = dist
                center = (j, i)
    maxdist = int(abs(maxdist))
    cv2.circle(egg_image, center, maxdist, (255, 0, 0),2, cv2.LINE_8, 0)
    cv2.circle(egg_image, center, 2, (255, 0, 0),2, cv2.LINE_8, 0)
    if pixelsPerMetric is None:
        pixelsPerMetric = (maxdist*2) / WIDTH
    #
    # # 计算对象的大小
    dim = (maxdist*2) / pixelsPerMetric
    # dimE = dB / pixelsPerMetric
    #
    # # 在图像上绘制对象大小

    print("内切圆直径："+str(dim))
    rad.append(dim)
    # print("内切圆直径："+str(rad))

h = H0*rad[1]/(rad[1]-rad[0])
R = h*rad[-1]/(2*h+rad[-1])
print("修正后直径："+str(2*R)+"mm")

cv2.putText(egg_image, "{:.1f}mm".format(R*2),
        (center), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 0, 255), 1)
    # 显示输出图像
    # resImg = cv2.resize(image, (10, 10), interpolation=cv2.INTER_CUBIC)
cv2.imshow("dst", dst)
# cv2.namedWindow("resImg",0)
cv2.imshow("resImg", egg_image)
cv2.waitKey(0)

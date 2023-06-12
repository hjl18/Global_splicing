import os
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import time
import my_RANSAC
import my_Warp


def display(img):
    plt.imshow(cv2.cvtColor(np.float32(img / 255), cv2.COLOR_BGR2RGB))
    plt.title("Splicing")
    plt.show()


# 保存图片
def save(img, name):
    cv2.imwrite("..//Result//" + str(name) + ".jpg", img)


# 将A，B大小变为一致(添加黑色边框)
def same_size(A, B):
    h1, w1, _ = A.shape
    h2, w2, _ = B.shape
    if w1 > w2:
        B = cv2.copyMakeBorder(B,
                               0,
                               0,
                               0,
                               w1 - w2,
                               cv2.BORDER_CONSTANT,
                               value=(0, 0, 0))
    else:
        A = cv2.copyMakeBorder(A,
                               0,
                               0,
                               w2 - w1,
                               0,
                               cv2.BORDER_CONSTANT,
                               value=(0, 0, 0))
    return (A, B)


# 圆柱投影
# f为圆柱半径，每次匹配需要调节f
def cylindrical_projection(img, f):
    rows = img.shape[0]
    cols = img.shape[1]

    blank = np.zeros_like(img)
    center_x = int(cols / 2)
    center_y = int(rows / 2)

    for y in range(rows):
        for x in range(cols):
            theta = math.atan((x - center_x) / f)
            point_x = int(f * math.tan((x - center_x) / f) + center_x)
            point_y = int((y - center_y) / math.cos(theta) + center_y)

            if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
                pass
            else:
                blank[y, x, :] = img[point_y, point_x, :]
    return blank


# 去除黑色边框
def remove_the_blackborder(img):
    image = img
    img = cv2.medianBlur(img, 5)
    b = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
    binary_image = b[1]
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

    edges_y, edges_x = np.where(binary_image == 255)
    bottom = min(edges_y)
    top = max(edges_y)
    #height = top - bottom

    left = min(edges_x)
    right = max(edges_x)
    height = top - bottom
    width = right - left
    res_image = image[bottom:bottom + height, left:left + width]
    return res_image


def Warp(A, B):
    img = same_size(A, B)
    srcImg, dstImg = img[0], img[1]
    #转换为灰度图
    img1gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    img2gray = cv2.cvtColor(dstImg, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT().create()

    kp1, des1 = sift.detectAndCompute(img1gray, None)
    kp2, des2 = sift.detectAndCompute(img2gray, None)

    # 用于匹配特征之的算法(画出匹配图)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None,
                              **draw_params)
    #cv2.imshow("FLANN", img3)
    #rows, cols = srcImg.shape[:2]
    #寻找单应性矩阵
    MIN_MATCH_COUNT = 20
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
        H = my_RANSAC.RANSAC(src_pts, dst_pts)
        #print("单应性矩阵H为：")
        #print(H)
    #todo: implement this function manually.
    # warpImg = cv2.warpPerspective(dstImg, np.linalg.inv(H), (dstImg.shape[1] + srcImg.shape[1], dstImg.shape[0]))

        warpImg = my_Warp.myWarpPerspective(
            dstImg, np.linalg.inv(H),
            (dstImg.shape[0], dstImg.shape[1] + srcImg.shape[1]))
        rows, cols = srcImg.shape[:2]

        #display(warpImg)
        for col in range(0, cols):
            if srcImg[:, col].any() and warpImg[:, col].any():
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if srcImg[:, col].any() and warpImg[:, col].any():
                right = col
                break

        res = np.zeros([rows, cols, 3], np.uint8)
        #阿尔法拼接方法
        for row in range(0, rows):
            for col in range(0, cols):
                if not dstImg[row, col].any():
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = srcImg[row, col]
                else:
                    dstImgLen = float(abs(col - left))
                    srcImgLen = float(abs(col - right))
                    alpha = dstImgLen / (dstImgLen + srcImgLen)
                    res[row, col] = np.clip(
                        srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha,
                        0, 255)

        warpImg[0:srcImg.shape[0], 0:srcImg.shape[1]] = res
        warpImg = remove_the_blackborder(warpImg)
        #display(warpImg)
        return warpImg
    else:
        print("MAP: Not  enough  matches are found!")
        exit(0)

def Splice(dir_name):
    if dir_name == "..//data//source004":
        img1 = cv2.imread(dir_name+'//panorama03_03.jpg')
        img2 = cv2.imread(dir_name+'//panorama03_04.jpg')
        img3 = cv2.imread(dir_name+'//panorama03_07.jpg')
        imgnew1 = Warp(img1, img2)
        imgnew2 = Warp(imgnew1, img3)
        display(imgnew2)
        save(imgnew2, 'source004')
    elif dir_name == '..//data//source005':
        img1 = cv2.imread(dir_name+'//file0001.jpg')
        img2 = cv2.imread(dir_name+'//file0002.jpg')
        img3 = cv2.imread(dir_name+'//file0003.jpg')
        img4 = cv2.imread(dir_name+'//file0004.jpg')
        img5 = cv2.imread(dir_name+'//file0005.jpg')
        imgnew1 = Warp(img5, img1)
        imgnew2 = Warp(img2, img3)
        imgnew3 = Warp(imgnew1, img4)
        imgnew4 = Warp(imgnew3, imgnew2)
        display(imgnew4)
        save(imgnew4, 'source005')
    elif dir_name == '..//data//source006':
        img1 = cv2.imread(dir_name+'//yosemite1.jpg')
        img2 = cv2.imread(dir_name+'//yosemite2.jpg')
        img3 = cv2.imread(dir_name+'//yosemite3.jpg')
        imgnew1 = Warp(img1, img2)
        imgnew2 = Warp(imgnew1, img3)
        display(imgnew2)
        save(imgnew2, 'source006')
    elif dir_name == '..//data//source007':
        img1 = cv2.imread(dir_name+'//0.jpg')
        img2 = cv2.imread(dir_name+'//1.jpg')
        img3 = cv2.imread(dir_name+'//2.jpg')
        img4 = cv2.imread(dir_name+'//3.jpg')
        imgnew1 = Warp(img1, img2)
        imgnew2 = Warp(img3, img4)
        imgnew3 = Warp(imgnew1, imgnew2)
        display(imgnew3)
        save(imgnew3, 'source007')
    elif dir_name == '..//data//source008':
        img1 = cv2.imread(dir_name+'//01.jpg')
        img2 = cv2.imread(dir_name+'//02.jpg')
        img3 = cv2.imread(dir_name+'//03.jpg')
        imgnew1 = Warp(img1, img2)
        imgnew2 = Warp(imgnew1, img3)
        display(imgnew2)
        save(imgnew2, 'source008')
    elif dir_name == '..//data//my_source':
        img1 = cv2.imread(dir_name + '//DJI_0278.jpg')
        img2 = cv2.imread(dir_name + '//DJI_0279.jpg')
        img3 = cv2.imread(dir_name + '//DJI_0280.jpg')
        imgnew1 = Warp(img1, img2)
        imgnew2 = Warp(imgnew1, img3)
        display(imgnew2)
        save(imgnew2, 'my_source')
    elif dir_name == '..//data//test':
        img1 = cv2.imread(dir_name + '//01.jpg')
        img2 = cv2.imread(dir_name + '//02.jpg')
        img3 = cv2.imread(dir_name + '//03.jpg')
        imgnew1 = Warp(img1, img2)
        imgnew2 = Warp(imgnew1, img3)
        display(imgnew2)
        save(imgnew2, 'test')

if __name__ == '__main__':
    dir_name = '..//data'
    dir_2_name = os.listdir(dir_name)
    print("图像处理中...")
    for img_dir in dir_2_name:
        img_path = dir_name +'//'+ img_dir
        Splice(img_path)
    print("图像处理完成，请在Result文件夹查看！")
    '''
    dir_name = '..//data//source005'
    img1 = cv2.imread(dir_name + '//file0001.jpg')
    img2 = cv2.imread(dir_name + '//file0002.jpg')
    img3 = cv2.imread(dir_name + '//file0003.jpg')
    img4 = cv2.imread(dir_name + '//file0004.jpg')
    img5 = cv2.imread(dir_name + '//file0005.jpg')
    imgnew1 = Warp(img5, img1)
    imgnew2 = Warp(img2, img3)
    imgnew3 = Warp(imgnew1, img4)
    imgnew4 = Warp(imgnew3, imgnew2)
    display(imgnew4)
    '''

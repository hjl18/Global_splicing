import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import time


#通过最小二乘法求解单应矩阵H
def findH(fp, tp):
    A = np.zeros((8, 8))
    B = np.zeros((8, 1))
    for i in range(4):
        A[2 * i] = [fp[0][i], fp[1][i], 1, 0, 0, 0,
                    -tp[0][i] * fp[0][i], -tp[0][i] * fp[1][i]]
        A[2 * i + 1] = [0, 0, 0, fp[0][i], fp[1][i], 1,
                        -tp[1][i] * fp[0][i], -tp[1][i] * fp[1][i]]
        B[2 * i] = [tp[0][i]]
        B[2 * i + 1] = [tp[1][i]]

    k = np.linalg.lstsq(A, B, rcond=None)[0]
    H = np.zeros((3, 3))
    H[2][2] = 1
    for i in range(8):
        H[i // 3][i % 3] = k[i][0]
    H = H / np.linalg.norm(H)
    return H

#将源图像中的一个点A通过单应矩阵H映射到目标图像
#它接受两个参数：H和A，分别表示单应矩阵和源图像中的一个点的坐标。它返回一个长度为2的一维数组，表示映射后的坐标。
def change(H,A):
    b = np.array([[A[0]],[A[1]],[1]])
    a = np.dot(H,b)
    tmp = a[2][0]
    B = np.array([a[0][0]/tmp,a[1][0]/tmp])
    return B

def RANSAC(A,B):
    random.seed(time.time())
    min_e = 999999
    e = 5
    p = 0.9
    n = len(A)
    H = False
    for ite in range(2000):
        flag = 0
        m = 0
        count = 0
        #随机选择4个对应点，计算它们之间的单应矩阵H
        s = random.sample(range(0,n),4)
        i1,i2,i3,i4 = s[0],s[1],s[2],s[3]
        fp = np.array([[A[i1][0],A[i2][0],A[i3][0],A[i4][0]],
                       [A[i1][1],A[i2][1],A[i3][1],A[i4][1]]])
        tp = np.array([[B[i1][0],B[i2][0],B[i3][0],B[i4][0]],
                       [B[i1][1],B[i2][1],B[i3][1],B[i4][1]]])
        try:
            h = findH(fp,tp)
        except:
            flag = 1
        if flag == 0:
            for i in range(n):
                #计算B和B_h之间的误差e_h，将所有误差值相加得到m
                B_h = change(h,A[i])
                e_h = abs(B[i][0]-B_h[0]) + abs(B[i][1]-B_h[1])
                m = m+e_h
                if e_h < e:
                    count += 1
            #如果m小于min_e，则更新min_e和H
            if m < min_e:
                min_e = m
                H = h
            #如果有超过p比例的点满足e_h小于e，则认为这个H是一个好的估计

            if count/n > p:
                break

    return H


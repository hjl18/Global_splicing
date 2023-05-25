import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import time

def warpPerspective(srcimg,M,dsize):
    rows = dsize[1]#360
    cols = dsize[0]#960
    warpimg = np.zeros([rows, cols, 3], np.uint8)
    #cols = srcimg.shape[1]#480
    #rows = srcimg.shape[0]#360
    for x in range(rows):
        for y in range(cols):
            X = int((M[0][0]*x+M[0][1]*y+M[0][2]) // (M[2][0]*x+M[2][1]*y+M[2][2]))
            Y = int((M[1][0]*x+M[1][1]*y+M[1][2]) // (M[2][0]*x+M[2][1]*y+M[2][2]))
            #print(X,Y)
            if X >= srcimg.shape[0] or Y >= srcimg.shape[1] or X < 0 or Y < 0:
                warpimg[x][y] = 0
                continue
            warpimg[x][y] = srcimg[X][Y]

    return warpimg
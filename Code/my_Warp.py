import numpy as np
import random
from matplotlib import pyplot as plt
import time
import typing


def myWarpPerspective(srcMat: np.ndarray, homoMat: np.ndarray,
                      dstSize: typing.Tuple[int, int]):
    """
    my implementation of cv2.warpPerspective Function
    :param srcMat: source image matrix.
    :param homoMat: homography matrix.
    :param dstSize: target size of the output matrix. **Attention: the size of the matrix is the transpose of the image size.**
    :returm dstMat: transformed image matrix.
    """
    srcShape = srcMat.shape
    # if len(srcShape) != 2:
    #     raise Exception("Input Image is not a gray image.")
    srcH, srcW = srcShape[0], srcShape[1]
    dstH, dstW = dstSize[0], dstSize[1]
    dstShape = np.array(srcShape)
    dstShape[0], dstShape[1] = dstH, dstW
    if dstH < srcH or dstW < srcW:  # if the output image size is smaller than the input image size, then return original
        return srcMat
    dst = np.zeros(dstShape, np.uint8)  #创建一张空白的图，尺寸和目标尺寸相同
    invHomoMat = np.linalg.inv(homoMat)
    # Using matrix multiplication to move pixels from srcMat to dst
    # no cv2.warpPerspective is allowed here.
    for y in range(dstSize[0]):
        for x in range(dstSize[1]):
            #从dst向src进行定位
            srcPixel = np.dot(invHomoMat, np.array([x, y, 1]))
            srcX, srcY = srcPixel[0] / srcPixel[2], srcPixel[1] / srcPixel[2]
            if srcX < 0 or srcX >= srcW or srcY < 0 or srcY >= srcH:
                continue
            #双线性插值
            x1, y1 = int(srcX), int(srcY)
            x2, y2 = x1 + 1, y1 + 1
            fx, fy = srcX - x1, srcY - y1
            p1 = srcMat[y1, x1]
            p2 = srcMat[y2, x1] if y2 < srcH else p1
            p3 = srcMat[y1, x2] if x2 < srcW else p1
            p4 = srcMat[
                y2,
                x2] if y2 < srcH and x2 < srcW else p2 if y2 < srcH else p3 if x2 < srcW else p1
            dst[y, x] = (1 - fy) * ((1 - fx) * p1 + fx * p3) + fy * (
                (1 - fx) * p2 + fx * p4)
    return dst
import cv2 as cv
import numpy as np

def canny_edge(src, use_cvfun=True):
    if use_cvfun:
        edge = cv.Canny(src, 100, 300)
    else:
        edge = np.zeros_like(src)[:,:,0]
    return edge

if __name__ == '__main__':
    img = cv.imread(r'week5\lenna.png')
    edge = cv.Canny(img, 100, 300)
    cv.imshow("lenna", edge)
    cv.waitKey(0)

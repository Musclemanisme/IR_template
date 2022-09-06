import cv2
import numpy as np

def laplacian_pyramid(src, level_num=4):
    '''
    This is an python implementation of Gaussian Pyramid and Laplacian Pyramid through OpenCV
    :param src: numpy.ndarray
    :param level_num: the number of pyramid levels
    :return: Gaussian Pyramid and Laplacian Pyramid that consists
    '''
    G_pyramid = []
    L_pyramid = []
    G_pyramid.append(src)
    for i in range(level_num-1):
        G_pyramid.append(cv2.pyrDown(G_pyramid[i]))
    for j in range(level_num-1):
        L_diff = cv2.subtract(G_pyramid[j], cv2.pyrUp(G_pyramid[j+1]))
        L_pyramid.append(L_diff)
    L_pyramid.append(G_pyramid[-1])
    # Currently item in Pyramid with smaller index has larger shape.
    # After reverse operation, item with smaller index has smaller shape.
    G_pyramid.reverse()
    L_pyramid.reverse()
    return G_pyramid, L_pyramid


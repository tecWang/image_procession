import cv2
import matplotlib.pyplot as plt
import numpy as np
from tec.tools import *

def simple_threshold(img):
    process_list = {}
    ret, simple_img1 = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    ret, simple_img1_trunc = cv2.threshold(img, 20, 255, cv2.THRESH_TRUNC)
    ret, simple_img2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    ret, simple_img2_trunc = cv2.threshold(img, 100, 255, cv2.THRESH_TRUNC)
    ret, simple_img3 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    ret, simple_img3_trunc = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)
    ret, simple_img4 = cv2.threshold(img, 500, 255, cv2.THRESH_BINARY)
    ret, simple_img4_trunc = cv2.threshold(img, 500, 255, cv2.THRESH_TRUNC)
    process_list["simple_img1_20"] = simple_img1
    process_list["simple_img1_20_trunc"] = simple_img1_trunc
    process_list["simple_img1_100"] = simple_img2
    process_list["simple_img1_100_trunc"] = simple_img2_trunc
    process_list["simple_img1_200"] = simple_img3
    process_list["simple_img1_200_trunc"] = simple_img3_trunc
    process_list["simple_img1_500"] = simple_img4
    process_list["simple_img1_500_trunc"] = simple_img4_trunc

    return process_list

def adaptive_threshold(img):
    process_list = {}
    adaptive_img_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 5)
    adaptive_img_gussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 5)
    process_list["adaptive_img1"] = adaptive_img_mean 
    process_list["adaptive_img_gussian"] = adaptive_img_gussian 

    return process_list

def otsu_threshold(img, fig_name=None):
    process_list = {}
    ret, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, otsu_img_inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if fig_name is not None:
        process_list[fig_name + "_otsu_img"] = otsu_img 
        process_list[fig_name + "_otsu_img_inv"] = otsu_img_inv
    else:
        process_list["otsu_img"] = otsu_img 
        process_list["otsu_img_inv"] = otsu_img_inv

    return process_list

if __name__ == "__main__":
    # img = cv2.imread("/home/tec/code/image_processing/lung_gray.png")
    # img = cv2.imread("E:\Code\image_processing\lung.png")
    img = cv2.imread("E:\Code\image_processing\HPY.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    process_list_simple = simple_threshold(img)
    process_list_adap = adaptive_threshold(img)
    process_list_otsu = otsu_threshold(img)

    roi_area = img[100:300, 100:300]
    process_list_roi = otsu_threshold(roi_area, fig_name="roi")

    process_list = {}
    process_list["source"] = img
    process_list = {**process_list, **process_list_simple, **process_list_adap, **process_list_otsu, **process_list_roi}
    plot_multi(process_list, cols=8, savefig=True, figdir="images_hpy")
    
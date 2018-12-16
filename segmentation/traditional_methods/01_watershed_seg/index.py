import cv2
import numpy as np
import matplotlib.pyplot as plt
from tec.tools import *

def main(img):
    process_list = {}
    
    #####################################################################################
    # origin image
    process_list["source"] = img
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    process_list["gray"] = gray
    
    #####################################################################################
    # threshold image
    handle_threshold = 127     # 55 bad result, 127 great result
    ret, handle_thres = cv2.threshold(gray,handle_threshold,255,cv2.THRESH_BINARY_INV)
    print("ret", ret)   # 127.0
    process_list["handle_thres"] = handle_thres
    # adaptive threshold
    adaptive_thres_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 33, 5)
    adaptive_thres_gussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 5)
    process_list["adaptive_thres_mean"] = adaptive_thres_mean
    process_list["adaptive_thres_gussian"] = adaptive_thres_gussian
    # otsu threshold
    ret, otsu_thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    def plot_hist(gray):
        hist = cv2.calcHist([gray], [0], None, [256], [0.0, 255.0])  # calc hist
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)  
        print(minVal, maxVal, minLoc, maxLoc)
        histImg = np.zeros([256,256,1], np.uint8)  
        hpt = int(0.9* 256);  # lower the height of the hist
        for h in range(256):  
            intensity = int(hist[h]* (hpt/maxVal))  
            cv2.line(histImg,(h,256), (h,256-intensity), [255]) 
        return histImg.reshape(256, 256)
        
    histImg = plot_hist(gray)
    process_list["histImg"] = histImg
    print("ret", ret)   # 162.0
    process_list["otsu_thresh"] = otsu_thresh

    #####################################################################################
    # remove small noises by morphological features
    kernel = np.ones((3, 3),np.uint8)
    erosion = cv2.erode(handle_thres, kernel, iterations=2)
    dilate = cv2.dilate(handle_thres, kernel, iterations=2)
    opening = cv2.morphologyEx(handle_thres, cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(handle_thres, cv2.MORPH_CLOSE,kernel, iterations = 2)
    process_list["erosion"] = erosion
    process_list["dilate"] = dilate
    process_list["opening"] = opening
    process_list["closing"] = closing

    #####################################################################################
    # Finding sure background area(black area) and foreground area(white area)
    sure_bg = cv2.dilate(erosion,kernel,iterations=3)
    process_list["sure_background"] = sure_bg

    dist_transform1 = cv2.distanceTransform(erosion,cv2.DIST_L1,5)
    dist_transform2 = cv2.distanceTransform(erosion,cv2.DIST_L2,5)
    process_list["dist_transform_l1"] = dist_transform1
    process_list["dist_transform_l2"] = dist_transform2
    ret, sure_fg0 = cv2.threshold(dist_transform1, 0.9*dist_transform1.max(),255,0)
    ret, sure_fg = cv2.threshold(dist_transform1, 0.7*dist_transform1.max(),255,0)
    ret, sure_fg2 = cv2.threshold(dist_transform1, 0.5*dist_transform1.max(),255,0)
    process_list["sure_foreground_0.9"] = sure_fg0
    process_list["sure_foreground_0.7"] = sure_fg
    process_list["sure_foreground_0.5"] = sure_fg2

    #####################################################################################
    # Finding unknown region
    sure_fg = np.uint8(sure_fg2)
    # extract pixels outside the foreground and background area in the image
    unknown = cv2.subtract(sure_bg,sure_fg)
    process_list["unknown"] = unknown

    #####################################################################################
    # Marker labelling
    ret, con_markers = cv2.connectedComponents(sure_fg)
    # print(ret)  # the number of connected areas
    process_list["con-markers"] = con_markers

    #####################################################################################
    # Add one to all labels so that sure background is not 0, but 1
    markers_1 = con_markers+10
    process_list["markers+1"] = markers_1
    print(markers_1)
    # Now, mark the region of unknown with zero
    markers_unknown = markers_1.copy()
    markers_unknown[unknown > 0] = 0
    # markers_unknown[unknown > 0] = 11
    process_list["markers-unknown"] = markers_unknown

    markers_res = cv2.watershed(img,markers_unknown)
    process_list["markers_res"] = markers_res
    img2 = img.copy()
    img2[markers_res == -1] = [255,0,0]
    process_list["final"] = img2
    return img, process_list

if __name__ == "__main__":
    
    imgpath = "E:\Code\image_processing\coins.jpg"
    # imgpath = "/home/tec/code/image_processing/lung.png"
    img = cv2.imread(imgpath)

    res, process_list = main(img)
    plot_multi(process_list, cols=8, savefig=True)
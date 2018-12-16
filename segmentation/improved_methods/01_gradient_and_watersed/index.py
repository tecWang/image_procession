import cv2
import numpy as np
import matplotlib.pyplot as plt
from tec.tools import *

def watershed(img, thres=20, max_value=255, thres_type=cv2.THRESH_BINARY, prefix="type"):
    process_list = {}

    # convert to binary image
    fg = img.copy()
    ret, handle_thres = cv2.threshold(fg, thres, max_value, thres_type)
    process_list[prefix + "-handle_thres"] = handle_thres

    # erosion 
    kernel = np.ones((2, 2),np.uint8)
    erosion = cv2.erode(handle_thres, kernel, iterations=3)
    process_list[prefix + "-erosion"] = erosion

    # confirm sure_bg and sure_fg
    sure_bg = cv2.dilate(erosion,kernel,iterations=3)
    process_list[prefix + "-sure_bg"] = sure_bg
    dist_transform1 = cv2.distanceTransform(erosion,cv2.DIST_L2,5)
    process_list[prefix + "-dist_transform_l2"] = dist_transform1
    ret, sure_fg = cv2.threshold(dist_transform1, 0.3*dist_transform1.max(),255,0)
    process_list[prefix + "-sure_fg_0.3"] = sure_fg

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    process_list[prefix + "-unknown"] = unknown

    # form gradient for the image
    ret, con_markers = cv2.connectedComponents(sure_fg)
    process_list[prefix + "-con-markers"] = con_markers

    # make bg_val != 0
    markers_1 = con_markers+1
    process_list[prefix + "-markers+1"] = markers_1

    # mark the region of unknown with zero
    markers_unknown = markers_1.copy()  
    markers_unknown[unknown > 0] = 0    
    process_list[prefix + "-markers_unknown"] = markers_unknown

    # convert colorspace for tec_watershed
    src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    process_list[prefix + "-src"] = src
    markers_res = cv2.watershed(src, markers_unknown)

    # produce final result
    res = src.copy()
    res[markers_res == -1] = [255,0,0]
    process_list[prefix + "-src-final"] = res

    return src, process_list

def grdient_transform(img, dtype="sobel"):
    process_list = {}

    if(dtype == "sobel"):
      # sobel
        # x means horizontal and y means vertical
        # def Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)   # convert negative values to positive values
        absY = cv2.convertScaleAbs(y)
        sobel_dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)        # addWeighted(img1, weight1, img2, weight2, bias)
        process_list["sobel_x"] = absX
        process_list["sobel_y"] = absY
        process_list["sobel_dst"] = sobel_dst
        # xy = cv2.Sobel(img, cv2.CV_16S, 1, 1)
        # absXY = cv2.convertScaleAbs(xy)
        # process_list["sobel_xy"] = absXY
        
        return sobel_dst, process_list
    elif(dtype == "scharr"):
        # scharr operator
        # def Scharr(src, ddepth, dx, dy, dst, scale, delta, borderType)
        scharr_x = cv2.Scharr(img,cv2.CV_16S,1,0)
        scharr_y = cv2.Scharr(img,cv2.CV_16S,0,1)
        scharr_absx = cv2.convertScaleAbs(scharr_x) 
        scharr_absy = cv2.convertScaleAbs(scharr_y)
        scharr_dst = cv2.addWeighted(scharr_absx, 0.5, scharr_absy, 0.5, 0)
        process_list["scharr_x"] = scharr_x
        process_list["scharr_y"] = scharr_y
        process_list["scharr_dst"] = scharr_dst

        return scharr_dst, process_list

    elif(dtype == "lap"):
        # laplacian operator
        gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
        lap_dst = cv2.convertScaleAbs(gray_lap) 
        process_list["laplacian"] = lap_dst
        return lap_dst, process_list

if __name__ == "__main__":
    img = cv2.imread("E:\Code\image_processing\lung.png")
    # img = cv2.imread("/home/tec/code/image_proce1ssing/lung_gray.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thres = 20
    max_value = 255
    thres_type = cv2.THRESH_BINARY

    # origin
    _, origin_watershed_result = watershed(img, thres=thres, max_value=max_value, thres_type=thres_type, prefix="origin")

    # sobel
    sobel_dst, sobel_result = grdient_transform(img)
    _, sobel_watershed_result = watershed(sobel_dst, thres=thres, max_value=max_value, thres_type=thres_type, prefix="sobel")

    # scharr
    scharr_dst, scharr_result = grdient_transform(img, "scharr")
    _, scharr_watershed_result = watershed(scharr_dst, thres=thres, max_value=max_value, thres_type=thres_type, prefix="scharr")

    # laplacian
    lap_dst, lap_result = grdient_transform(img, "lap")
    _, lap_watershed_result = watershed(lap_dst, thres=thres, max_value=max_value, thres_type=thres_type, prefix="lap")

    # preview result
    # process_list = {**origin_watershed_result,
    #                 **sobel_result, **sobel_watershed_result, 
    #                 **scharr_result, **scharr_watershed_result,
    #                 **lap_result, **lap_watershed_result}
    process_list = {
                    **sobel_result, **sobel_watershed_result}
    plot_multi(process_list, cols=8, savefig=True)

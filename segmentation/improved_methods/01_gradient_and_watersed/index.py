import cv2
import numpy as np
import matplotlib.pyplot as plt
from tec.tools import *
from tec.image_processor import *

img = cv2.imread("E:\Code\image_processing\lung.png")
# img = cv2.imread("/home/tec/code/image_proce1ssing/lung_gray.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thres_low = 20
thres_high = 40
max_value = 255
thres_type = cv2.THRESH_BINARY

# origin
# _, origin_watershed_result = watershed(img, thres=thres_low, max_value=max_value, thres_type=thres_type, prefix="origin")

# sobel
sobel_dst, sobel_result = tec_grdient_transform(img)
# _, sobel_watershed_result = tec_watershed(sobel_dst, thres=thres_low, max_value=max_value, thres_type=thres_type, dist_percentage=0.1, prefix="sobel_low")
_, sobel_watershed_result2 = tec_watershed(sobel_dst, thres=thres_high, max_value=max_value, thres_type=thres_type, dist_percentage=0.1, prefix="sobel_high")


#####################################################################################
# region_grow
grow_img = sobel_watershed_result2["sobel_high-final-single"]
sobel_watershed_result2["grow_img"] = grow_img

A,B = np.nonzero(grow_img)
points = zip(A, B)
point_list = []
for i in points:
    point_list.append(region_grow.Point(i[0], i[1], sobel_dst[i[0], i[1]]))
print(len(point_list))
sa = region_grow.Seed_Area(sobel_dst, point_list)
grow_res = sa.grow(thres=1)
sobel_watershed_result2["grow_res"] = grow_res

process_list = {
                # **origin_watershed_result}
                **sobel_result,
                **sobel_watershed_result2}
                # **sobel_watershed_result2}
plot_multi(process_list, cols=8, savefig=True, showtitle=True)

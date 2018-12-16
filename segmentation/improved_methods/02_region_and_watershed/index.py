import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tec.tools.image_tools as imtool
import tec.image_processor.region_grow as region_grow
import tec.image_processor.segmentation as segmentation

# img = cv2.imread("/home/tec/code/image_processing/lung.png")
img = cv2.imread("E:\Code\image_processing\lung.png")
# img = cv2.imread("/home/tec/code/image_processing/coins.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res, res_list = segmentation.tec_grdient_transform(img, "sobel", ksize=3)

print(pd.DataFrame(img).describe())
print(pd.DataFrame(res).describe())

# watershed
low_res, low_water_list = segmentation.tec_watershed(res, thres=10, max_value=255, thres_type=cv2.THRESH_BINARY, prefix="low")
mid_res, mid_water_list = segmentation.tec_watershed(res, thres=70, max_value=255, thres_type=cv2.THRESH_BINARY, prefix="mid")
high_res, high_water_list = segmentation.tec_watershed(res, thres=150, max_value=255, thres_type=cv2.THRESH_BINARY, prefix="high")

# region_grow
p1 = region_grow.Point(400, 200, res[400, 200])
sa = region_grow.Seed_Area(res, [p1])
grow_res = sa.grow(thres=13)
process_list = {}
process_list["grow_res"] = grow_res


# final[]
final = mid_water_list["mid-handle_thres"].copy()
final[grow_res==1] = 0
process_list["final"] = final

imtool.plot_multi({
            # **low_water_list,
            **mid_water_list,
            # **high_water_list, 
            **process_list
            }, cols=8, savefig=True, showtitle=False)


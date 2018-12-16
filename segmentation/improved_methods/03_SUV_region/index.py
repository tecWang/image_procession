import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt


ct_name = "/home/tec/code/image_processing/NSCLC Radiogenomics/AMC-001/04-30-1994-PETCT Lung Cancer-74760/3-CT FUSION-97864/000013.dcm"
try:
    ds = pydicom.read_file(ct_name)
except Exception:
    print(ct_name)
    exit()

img = ds.pixel_array 
plt.imshow(img)
plt.show()

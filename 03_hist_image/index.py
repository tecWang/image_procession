import cv2
import numpy as np

# load img
img = cv2.imread("/home/tec/code/image_processing/coins.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])

print("len hist:", len(hist))   # 256
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)  
print(minVal, maxVal, minLoc, maxLoc)
histImg = np.zeros([256,256,1], np.uint8)  
hpt = int(0.9* 256);  # lower the height of the hist
      
for h in range(256):  
    intensity = int(hist[h]* (hpt/maxVal))  
    cv2.line(histImg,(h,256), (h,256-intensity), [255]) 

cv2.imshow("histImg", histImg)  
cv2.imwrite("./hist_image.png", histImg)
cv2.waitKey(10000)  
cv2.destroyAllWindows()
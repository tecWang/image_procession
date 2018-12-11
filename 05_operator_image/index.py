import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../lung_gray.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sobel operator
# in the case of 8-bit input images it will result in truncated derivatives, so use cv_16s to change to 16bit
x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)

# convert negative values to positive values
# back to uint8
absX = cv2.convertScaleAbs(x) 
absY = cv2.convertScaleAbs(y)
# addWeighted(img1, weight1, img2, weight2, bias)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


# scharr operator
x2 = cv2.Scharr(img,cv2.CV_16S,1,0)
y2 = cv2.Scharr(img,cv2.CV_16S,0,1)
absX2 = cv2.convertScaleAbs(x2) 
absY2 = cv2.convertScaleAbs(y2)
dst2 = cv2.addWeighted(absX2, 0.5, absY2, 0.5, 0)

# laplacian operator
gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
dst3 = cv2.convertScaleAbs(gray_lap) 
print(dst3.shape, gray_lap.shape)

# canny operator
def CannyThreshold(gray, lowThreshold):
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3

    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold*ratio, apertureSize = kernel_size)
    # dst = cv2.bitwise_and(img, img, mask = detected_edges)  # just add some colours to edges from original image.
    return detected_edges

img = cv2.GaussianBlur(img,(3,3),0)
# canny = CannyThreshold(img, 0)
lowThreshold = 0
max_lowThreshold = 100
cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)


# plot 
def plot(data, rows=4, cols=3):
    plt.figure()
    for x in range(rows):
        for y in range(cols):
            plt.subplot(rows, cols, x*cols+y+1)
            plt.xticks([])
            plt.yticks([])
            try:
                plt.imshow(data[list(data.keys())[x]][y])
                plt.title(list(data.keys())[x])
            except Exception:
                plt.imshow(np.ones(shape=(512, 512)))
                plt.title("Non image")
    plt.tight_layout()
    plt.savefig("./combine_image.png", dpi=300)
    plt.show()


plot({
    "sobel": [absX, absY, dst],
    "scharr": [absX2, absY2, dst2],
    "laplacian": [gray_lap, dst3],
    "canny": [canny]
})

cv2.imwrite("./label_absX.png", absX)
cv2.imwrite("./label_absY.png", absY)
cv2.imwrite("./label_dst.png", dst)
cv2.imwrite("./scharr_absX2.png", absX2)
cv2.imwrite("./scharr_absY2.png", absY2)
cv2.imwrite("./scharr_dst2.png", dst2)
cv2.imwrite("./lap_gray_lap.png", gray_lap)
cv2.imwrite("./lap_dst3.png", dst3)
cv2.imwrite("./canny.png", canny)

cv2.waitKey(10000)
cv2.destroyAllWindows()
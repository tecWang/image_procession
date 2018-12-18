import cv2
import matplotlib.pyplot as plt
import numpy as np
from tec.tools import *

class Seed_Area():
    
    def __init__(self, image, seed_points):
        self.seed_points = seed_points
        self.image = image

    def calc_gray_diff(self, p1, p2):
        # print(p1.get_gray())
        # print(p2.get_gray())
        return float(p1.get_gray()) - float(p2.get_gray())

    def get_gray_value(self, x, y):
        return self.image[x, y]

    def get_connects(self, seed_x, seed_y):
        if(seed_x == 0 or seed_y == 0 or seed_x == self.image.shape[0]-1 or seed_y == self.image.shape[1]-1):
            #####################################################################################
            # four corner
            # left upper
            if seed_x == 0 and seed_y == 0:
                # print("left upper")
                return [Point(seed_x+1, seed_y, self.get_gray_value(seed_x+1, seed_y)),
                        Point(seed_x+1, seed_y+1, self.get_gray_value(seed_x+1, seed_y+1)),
                        Point(seed_x, seed_y+1, self.get_gray_value(seed_x, seed_y+1))]
            # right upper
            elif seed_x == 0 and seed_y == self.image.shape[1]-1:
                # print("right upper")
                return [Point(seed_x+1, seed_y, self.get_gray_value(seed_x+1, seed_y)),
                        Point(seed_x+1, seed_y-1, self.get_gray_value(seed_x+1, seed_y-1)),
                        Point(seed_x, seed_y-1, self.get_gray_value(seed_x, seed_y-1))]
            # left lower
            elif seed_x == self.image.shape[0]-1 and seed_y == 0:
                # print("left lower")
                return [Point(seed_x, seed_y+1, self.get_gray_value(seed_x, seed_y+1)),
                        Point(seed_x-1, seed_y, self.get_gray_value(seed_x-1, seed_y)),
                        Point(seed_x-1, seed_y+1, self.get_gray_value(seed_x-1, seed_y+1))]
            # right lower
            elif seed_x == self.image.shape[0]-1 and seed_y == self.image.shape[1]-1:
                # print("right lower")
                return [Point(seed_x, seed_y-1, self.get_gray_value(seed_x, seed_y-1)),
                        Point(seed_x-1, seed_y, self.get_gray_value(seed_x-1, seed_y)),
                        Point(seed_x-1, seed_y-1, self.get_gray_value(seed_x-1, seed_y-1))]
            #####################################################################################
            # four border
            # upper line
            elif seed_x == 0 and seed_y != 0 and seed_y != self.image.shape[1]-1:
                # print("upper line")
                return [Point(seed_x, seed_y-1, self.get_gray_value(seed_x, seed_y-1)),
                        Point(seed_x, seed_y+1, self.get_gray_value(seed_x, seed_y+1)),
                        Point(seed_x+1, seed_y-1, self.get_gray_value(seed_x+1, seed_y-1)),
                        Point(seed_x+1, seed_y, self.get_gray_value(seed_x+1, seed_y)),
                        Point(seed_x+1, seed_y+1, self.get_gray_value(seed_x+1, seed_y+1))]
            # lower line
            elif seed_x == self.image.shape[0]-1 and seed_y != 0 and seed_y != self.image.shape[1]-1:    
                # print("lower line")
                return [Point(seed_x, seed_y-1, self.get_gray_value(seed_x, seed_y-1)),
                        Point(seed_x, seed_y+1, self.get_gray_value(seed_x, seed_y+1)),
                        Point(seed_x-1, seed_y-1, self.get_gray_value(seed_x-1, seed_y-1)),
                        Point(seed_x-1, seed_y, self.get_gray_value(seed_x-1, seed_y)),
                        Point(seed_x-1, seed_y+1, self.get_gray_value(seed_x-1, seed_y+1))]
            # left line
            elif seed_y == 0 and seed_x != 0 and seed_x != self.image.shape[0]-1:
                # print("left line")
                return [Point(seed_x-1, seed_y, self.get_gray_value(seed_x-1, seed_y)),
                        Point(seed_x+1, seed_y, self.get_gray_value(seed_x+1, seed_y)),
                        Point(seed_x-1, seed_y+1, self.get_gray_value(seed_x-1, seed_y+1)),
                        Point(seed_x, seed_y+1, self.get_gray_value(seed_x, seed_y+1)),
                        Point(seed_x+1, seed_y+1, self.get_gray_value(seed_x+1, seed_y+1))]
            # right line
            elif seed_y == self.image.shape[1]-1 and seed_x != 0 and seed_x != self.image.shape[0]-1:
                # print("right line")
                return [Point(seed_x-1, seed_y, self.get_gray_value(seed_x-1, seed_y)),
                        Point(seed_x+1, seed_y, self.get_gray_value(seed_x+1, seed_y)),
                        Point(seed_x-1, seed_y-1, self.get_gray_value(seed_x-1, seed_y-1)),
                        Point(seed_x, seed_y-1, self.get_gray_value(seed_x, seed_y-1)),
                        Point(seed_x+1, seed_y-1, self.get_gray_value(seed_x+1, seed_y-1))]
        else:
            return [Point(seed_x-1,  seed_y-1,  self.get_gray_value(seed_x-1, seed_y-1)),
                    Point(seed_x-1,  seed_y,  self.get_gray_value(seed_x-1, seed_y)),
                    Point(seed_x-1,  seed_y+1,  self.get_gray_value(seed_x-1, seed_y+1)),
                    Point(seed_x,  seed_y-1,  self.get_gray_value(seed_x, seed_y-1)),
                    Point(seed_x,  seed_y+1,  self.get_gray_value(seed_x, seed_y+1)),
                    Point(seed_x+1,  seed_y-1,  self.get_gray_value(seed_x+1, seed_y-1)),
                    Point(seed_x+1,  seed_y,  self.get_gray_value(seed_x+1, seed_y)),
                    Point(seed_x+1,  seed_y+1,  self.get_gray_value(seed_x+1, seed_y+1))]

    def grow(self, thres):
        seed_flag = np.zeros(shape=self.image.shape)
        # set existing seed point = 1
        for seed in self.seed_points:
            seed_flag[seed.get_x(), seed.get_y()] = 1


        init_flag = False
        while(len(self.seed_points) > 0):
            # remove current seed point from seed_list  
            # print('=====================================================================================')
            # print('')
            # print("len(self.seed_points)", len(self.seed_points))
            # print('')
            # print('=====================================================================================')
            if init_flag is True:
                self.seed_points.pop(0)

            for seed in self.seed_points:
                # find neareast 8 points through the position of seed point
                # construct 8 Point objects
                # print(seed.get_x(), seed.get_y())
                seed_x = seed.get_x()
                seed_y = seed.get_y()
                connects = self.get_connects(seed_x, seed_y)
                # print("seed_x", seed_x, "seed_y", seed_y, "connects", connects)
                for target in connects:
                    # print("target", (target.get_x(), target.get_y()), "seed", (seed.get_x(), seed.get_y()))
                    diff = self.calc_gray_diff(seed, target)
                    # print("diff", diff)
                    if(diff < thres and seed_flag[target.get_x(), target.get_y()]==0):
                        self.seed_points.append(target)
                        # print(target.get_x())
                        seed_flag[target.get_x(), target.get_y()] = 1
                if init_flag is not True:
                    self.seed_points.pop(0)
                    init_flag = True
            return seed_flag
                
class Point(object):
    
    def __init__(self, x, y, gray):
        self.x = x
        self.y = y
        self.gray = gray

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
    
    def get_gray(self):
        return self.gray

if __name__ == "__main__":
    img = cv2.imread("E:\Code\image_processing\lung.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float")
    # roi = img[:100, :100]
    roi = img
    p1 = Point(x=250, y=280, gray=roi[250, 280])
    # p2 = Point(x=250, y=460, gray=roi[250, 460])
    sa = Seed_Area(roi, [p1])

    thres = 1
    res = sa.grow(thres=thres)

    process_list = {}
    process_list["input"] = roi
    process_list["res"] = res

    plot_multi(process_list, savefig=True, cols=2)
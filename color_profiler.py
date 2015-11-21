import cv2
import numpy as np

class ColorProfiler(object):
    def __init__(self, centers, windowSize, hsvRange, parent):
        self.centers = centers
        self.windowSize = windowSize
        self.hsvRange = hsvRange
        self.hsvColors = np.zeros((centers.shape[0],3), dtype=np.uint8)
        self.hsvRanges = np.zeros((centers.shape[0],2,3), dtype=np.uint8)
        self.parent = parent

    def run(self, imhsv):
        medianHSV = cv2.medianBlur(imhsv,self.windowSize)
        for i in range(self.centers.shape[0]):
            centerHSV = medianHSV[self.centers[i][1],self.centers[i][0]]
            self.hsvColors[i] = centerHSV
            
            minH,maxH = self.find_color_range(centerHSV[0], self.hsvRange[0], 0, 179)
            minS,maxS = self.find_color_range(centerHSV[1], self.hsvRange[1], 0, 255)
            minV,maxV = self.find_color_range(centerHSV[2], self.hsvRange[2], 0, 255)
            hsvMint = np.array((minH, minS, minV))
            hsvMaxt = np.array((maxH, maxS, maxV))
            self.hsvRanges[i] = np.array([hsvMint, hsvMaxt])

    def find_color_range(self, centerHSV, hsvRange, minimum=0, maximum=255):
        if (maximum-centerHSV) < hsvRange/2:
            retMax = int(maximum)
            retMin = int(max(retMax-hsvRange, minimum))
        elif (centerHSV-minimum) < hsvRange/2:
            retMin = int(minimum)
            retMax = int(min(retMin+hsvRange, maximum))
        else:
            retMax = int(min(centerHSV+hsvRange/2, maximum))
            retMin = int(max(centerHSV-hsvRange/2, minimum))
        return retMin, retMax

    def draw_color_windows(self, image):
        #colorFlag = np.zeros(centers.shape[0])
        #imhsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for c in self.centers:
            cv2.rectangle(image, tuple(np.subtract(c,np.array([(self.windowSize-1)/2,(self.windowSize-1)/2]))), tuple(np.add(c,np.array([(self.windowSize-1)/2,(self.windowSize-1)/2]))), [0,0,255])
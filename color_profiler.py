import cv2
import numpy as np

class ColorProfiler(object):
    def __init__(self, centers, bgCenters, windowSize, hsvRange, parent):
        self.centers = centers
        self.bgCenters = bgCenters
        self.windowSize = windowSize
        self.hsvRange = hsvRange
        self.hsvColors = np.zeros((centers.shape[0],3), dtype=np.uint8)
        self.hsvBgColors = np.zeros((bgCenters.shape[0],3), dtype=np.uint8)
        self.hsvRanges = np.zeros((centers.shape[0],2,3), dtype=np.uint8)
        self.parent = parent

    def run(self):
        for i in range(self.centers.shape[0]):
            centerHSV = self.hsvColors[i]
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

    def draw_color_windows(self, image, imhsv):
        medianHSV = cv2.medianBlur(imhsv,self.windowSize)
        for i in range(self.centers.shape[0]):
            self.hsvColors[i] = medianHSV[self.centers[i][1],self.centers[i][0]]
        centerHues = self.hsvColors.T[0]
        #hueVariance = np.sqrt(np.var(centerHues))
        #cv2.putText(image, str(3*hueVariance/2), (self.parent.parent.parent.imWidth/20,self.parent.parent.parent.imHeight/4), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
        globalMedianHue = np.median(self.hsvColors, axis=0)[0] * np.ones((centerHues.shape[0]))
        windowFlags = np.logical_and(centerHues >= globalMedianHue-25, centerHues <= globalMedianHue+25)
        for i,c in enumerate(self.centers):
            if not windowFlags[i]:
                cv2.rectangle(image, tuple(np.subtract(c,np.array([(self.windowSize-1)/2,(self.windowSize-1)/2]))), tuple(np.add(c,np.array([(self.windowSize-1)/2,(self.windowSize-1)/2]))), [0,0,255])
            else:
                cv2.rectangle(image, tuple(np.subtract(c,np.array([(self.windowSize-1)/2,(self.windowSize-1)/2]))), tuple(np.add(c,np.array([(self.windowSize-1)/2,(self.windowSize-1)/2]))), [0,255,0])

#        for i in range(self.bgCenters.shape[0]):
#            self.hsvBgColors[i] = medianHSV[self.bgCenters[i][1],self.bgCenters[i][0]]
#        bgCenterHues = self.hsvBgColors.T[0]
#        globalMedianHue = np.median(self.hsvColors, axis=0)[0] * np.ones((bgCenterHues.shape[0]))
#        bgWindowFlags = np.logical_or(bgCenterHues < globalMedianHue-35, bgCenterHues > globalMedianHue+35)
#        for i,c in enumerate(self.bgCenters):
#            if not bgWindowFlags[i]:
#                cv2.rectangle(image, tuple(np.subtract(c,np.array([(self.windowSize-1)/2,(self.windowSize-1)/2]))), tuple(np.add(c,np.array([(self.windowSize-1)/2,(self.windowSize-1)/2]))), [0,0,255])
#            else:
#                cv2.rectangle(image, tuple(np.subtract(c,np.array([(self.windowSize-1)/2,(self.windowSize-1)/2]))), tuple(np.add(c,np.array([(self.windowSize-1)/2,(self.windowSize-1)/2]))), [255,0,0])
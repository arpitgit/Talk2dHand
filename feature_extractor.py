import cv2
import numpy as np

class FeatureExtractor(object):
    def __init__(self, type, parent):
        self.type = type
        if type == 'orb':
            self.model = cv2.ORB()
        elif type == 'sift':
            self.model = cv2.SIFT()
        elif type == 'surf':
            self.model = cv2.SURF()
        else:
            self.model = None
        self.parent = parent

    def get_keypoints_and_descriptors(self, image):
        kp = self.model.detect(image, None)
        kp, des = self.model.compute(image, kp)
        return kp,des

    def draw_keypoints(self, image, kp):
        cv2.drawKeypoints(image, kp, image, color=(0,255,0), flags=0)

class Collector(object):
    def __init__(self, parent):
        self.desList = []
        self.handTracker = HandTracker(kernelSize=7, thresholdAngle=0.4, defectDistFromHull=30, parent=self)
        self.featureExtractor = featureExtractor
        self.parent = parent
        self.minDescriptorsPerFrame = self.parent.parent.minDescriptorsPerFrame

    def collect_descriptors_from_video(self, minDescriptorsPerFrame):
        gestureID = 1
        frameNum = 0
        gestureFrameID = 0
        captureFlag = False
        vc = self.parent.parent.vc
        while(vc.isOpened()):
            ret,im = vc.read()
            im = cv2.flip(im, 1)
            imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            binaryIm = self.handTracker.get_binary_image(imhsv)
            cnt,hull,centroid,defects = self.handTracker.get_contour(binaryIm)
            imCopy = 1*im
            if cnt is not None:
                cropImage = self.handTracker.get_cropped_image(im, cnt)
                cropImageGray = self.handTracker.get_cropped_image(imgray, cnt)
                kp,des = self.featureExtractor.get_keypoints_and_descriptors(cropImageGray)
                if des is not None and des.shape[0] >= 0:
                    self.featureExtractor.draw_keypoints(cropImage, kp)
                if captureFlag:
                    if des is not None and des.shape[0] >= minDescriptorsPerFrame:
                        self.desList.append(des)
                        self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,255,0))
                        frameNum += 1
                    else:
                        self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,0,255))
                    if frameNum >= self.numFramesPerGesture:
                        if gestureID >= self.numGestures:
                            break
                        else:
                            captureFlag = False
                            gestureID += 1
                            frameNum = 0
                else:
                    self.handTracker.draw_on_image(imCopy, cnt=False)
                cv2.imshow(handWindowName, cropImage)
            cv2.imshow(windowName,imCopy)
            k = cv2.waitKey(1)
            if not captureFlag:
                print "Press <space> for new gesture <{0}>!".format(gestureID)
                if k == 32:
                    captureFlag = True
                elif k == 27:
                    sys.exit(0)
            else:
                if k == 27:
                    sys.exit(0)
        descriptors = self.desList[0]
        for des in self.desList:
            descriptors = np.vstack((descriptors, des))
        return descriptors

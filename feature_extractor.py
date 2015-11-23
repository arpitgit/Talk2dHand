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

    def get_keypoints(self, image):
        kp = self.model.detect(image, None)
        return kp

    def compute_descriptors(self, image, kp):
        kp, des = self.model.compute(image, kp)
        return kp,des

    def draw_keypoints(self, image, kp):
        cv2.drawKeypoints(image, kp, image, color=(0,255,0), flags=0)

    def get_keypoints_in_contour(self, kps, cnt):
        kpList = []
        for kp in kps:
            if cv2.pointPolygonTest(cnt,kp.pt,False) != -1:
                kpList.append(kp)
        return kpList

import sys
import os
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn import svm
from feature_extractor import FeatureExtractor
from hand_tracker import HandTracker

class Trainer(object):
    def __init__(self, numGestures, numFramesPerGesture, minDescriptorsPerFrame, numWords, descType, kernel, numIter, parent):
        self.numGestures = numGestures
        self.numFramesPerGesture = numFramesPerGesture
        self.numWords = numWords
        self.minDescriptorsPerFrame = minDescriptorsPerFrame
        self.parent = parent
        self.desList = []
        self.voc = None
        self.classifier = None
        self.windowName = "Training preview"
        self.handWindowName = "Cropped hand"
        self.binaryWindowName = "Binary frames"
        self.handTracker = HandTracker(kernelSize=7, thresholdAngle=0.4, defectDistFromHull=30, parent=self)
        self.featureExtractor = FeatureExtractor(type=descType, parent=self)
        self.kernel = kernel
        self.numIter = numIter
        self.numDefects = None
        self.firstFrameList = []
        self.trainLabels = []

    def extract_descriptors_from_images(self, gestureDirList, parentDirPath):
        #self.numFramesPerGestureList = []
        for i,gestureDir in enumerate(gestureDirList):
            gestureDirPath = os.path.join(parentDirPath, gestureDir)
            imgList = []
            for dirpath, dirnames, filenames in os.walk("%s" % (gestureDirPath), topdown=True, followlinks=True):
                for f in filenames:
                    if f.endswith(".jpg"):
                        imgList.append(os.path.join(dirpath, f))
            #self.numFramesPerGestureList.append(len(imgList))
            gestureID = i+1
            for j,f in enumerate(imgList):
                cropImage = cv2.imread(f)
                cropImage = cv2.flip(cropImage, 1)
                cropImageGray = cv2.cvtColor(cropImage, cv2.COLOR_BGR2GRAY)
                kp,des = self.featureExtractor.get_keypoints_and_descriptors(cropImageGray)
                if des is not None and des.shape[0] >= 0:
                    self.featureExtractor.draw_keypoints(cropImage, kp)
                    self.desList.append(des)
                    self.trainLabels.append(gestureID)
                if j == 0:
                    self.firstFrameList.append(cropImage)
                cv2.imshow(self.handWindowName, cropImage)
                k = cv2.waitKey(1)
                if k == 27:
                    sys.exit(0)
        cv2.destroyAllWindows()

    def extract_descriptors_from_video(self):
        vc = self.parent.vc
        while(vc.isOpened()):
            ret,im = vc.read()
            im = cv2.flip(im, 1)
            imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            self.handTracker.colorProfiler.draw_color_windows(im)
            cv2.imshow(self.windowName, im)
            k = cv2.waitKey(1)
            if k == 32: # space
                break
            elif k == 27:
                sys.exit(0)

        self.handTracker.colorProfiler.run(imhsv)
        binaryIm = self.handTracker.get_binary_image(imhsv)
        cnt,hull,centroid,defects = self.handTracker.initialize_contour(binaryIm)
        self.numDefects = np.zeros((self.numGestures,self.numFramesPerGesture), "uint8")

        #self.numFramesPerGestureList = [self.numFramesPerGesture] * self.numGestures
        gestureID = 1
        frameNum = 0
        captureFlag = False
        while(vc.isOpened()):
            ret,im = vc.read()
            im = cv2.flip(im, 1)
            imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            binaryIm = self.handTracker.get_binary_image(imhsv)
            cnt,hull,centroid,defects = self.handTracker.get_contour(binaryIm)
            imCopy = 1*im
            if cnt is not None:
                cropImage,cropPoints = self.handTracker.get_cropped_image_from_cnt(im, cnt, 0.05)
                cropImageGray = self.handTracker.get_cropped_image_from_points(imgray, cropPoints)
                kp,des = self.featureExtractor.get_keypoints_and_descriptors(cropImageGray)
                if des is not None and des.shape[0] >= 0:
                    self.featureExtractor.draw_keypoints(cropImage, kp)
                if captureFlag:
                    if frameNum == 0:
                        self.firstFrameList.append(im)
                    if des is not None and des.shape[0] >= self.minDescriptorsPerFrame and self.is_hand(defects):
                        self.desList.append(des)
                        self.trainLabels.append(gestureID)
                        self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,255,0))
                        self.numDefects[gestureID-1][frameNum] = defects.shape[0]
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
                cv2.imshow(self.handWindowName, cropImage)
            cv2.imshow(self.binaryWindowName, binaryIm)
            cv2.imshow(self.windowName,imCopy)
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
        cv2.destroyAllWindows()

    def kmeans(self):
        print "Running k-means clustering with {0} iterations...".format(self.numIter)
        descriptors = self.desList[0]
        for des in self.desList:
            descriptors = np.vstack((descriptors, des))
        if descriptors.dtype != "float32":
            descriptors = np.float32(descriptors)
        self.voc,variance = kmeans(descriptors, self.numWords, self.numIter)
        return variance

    def bow(self):
        print "Extracting bag-of-words features for {0} visual words...".format(self.numWords)
        self.trainData = np.zeros((len(self.trainLabels), self.numWords), "float32")
        for i in range(len(self.trainLabels)):
            words, distance = vq(self.desList[i], self.voc)
            for w in words:
                self.trainData[i][w] += 1

    def svm(self):
        print "Training SVM classifier with {0} kernel...".format(self.kernel)
        if self.kernel == "linear":
            clf = svm.LinearSVC()
        elif self.kernel == "hist":
            from sklearn.metrics.pairwise import additive_chi2_kernel
            clf = svm.SVC(kernel=additive_chi2_kernel, decision_function_shape='ovr')
        else:
            clf = svm.SVC(kernel=self.kernel, decision_function_shape='ovr')
        valScore = self.leave_one_out_validate(clf)
        clf.fit(self.trainData, self.trainLabels)
        self.classifier = clf
        self.classifier.voc = self.voc
        if self.numDefects is not None:
            self.classifier.medianDefects = np.median(self.numDefects, axis=1)
        else:
            self.classifier.medianDefects = None
        return valScore

    def leave_one_out_validate(self, clf):
        fullTrainData = self.trainData
        fullTrainLabels = self.trainLabels
        accuracy = np.zeros(len(fullTrainLabels))
        for i in range(len(fullTrainLabels)):
            testData = fullTrainData[i]
            testLabels = fullTrainLabels[i]
            trainData = np.append(fullTrainData[:i], fullTrainData[i+1:], axis=0)
            trainLabels = np.append(fullTrainLabels[:i], fullTrainLabels[i+1:])
            #clf = svm.LinearSVC()
            clf.fit(trainData, trainLabels)
            prediction = clf.predict(testData.reshape(1,-1))
            #score = clf.decision_function(testData.reshape(1,-1))
            if prediction != testLabels:
                accuracy[i] = 0
            else:
                accuracy[i] = 1
        return np.mean(accuracy)

    def predict(self, testData):
        prediction = self.classifier.predict(testData.reshape(1,-1))
        score = self.classifier.decision_function(testData.reshape(1,-1))
        return prediction[0], score[0]

    def is_hand(self, defects):
        if defects.shape[0] > 4:
            return False
        else:
            return True

import sys
import cv2
import numpy as np
from scipy.cluster.vq import vq
from sklearn import svm
from feature_extractor import FeatureExtractor
from hand_tracker import HandTracker

class Tester(object):
    def __init__(self, numGestures, minDescriptorsPerFrame, numWords, descType, numPredictions, parent):
        self.numGestures = numGestures
        self.numWords = numWords
        self.minDescriptorsPerFrame = minDescriptorsPerFrame
        self.parent = parent
        self.classifier = None
        self.windowName = "Testing preview"
        self.handWindowName = "Cropped hand"
        self.binaryWindowName = "Binary frames"
        self.predictionList = [-1]*numPredictions;
        self.handTracker = HandTracker(kernelSize=7, thresholdAngle=0.4, defectDistFromHull=30, parent=self)
        self.featureExtractor = FeatureExtractor(type=descType, parent=self)
        self.numSideFrames = 10
        self.prevFrameList = np.zeros((self.numSideFrames,self.parent.imHeight/self.numSideFrames,self.parent.imWidth/self.numSideFrames,3), "uint8")
        self.numPrevFrames = 0
        self.predictionScoreThreshold = 0.2
        self.learningRate = 0.01
        self.numReinforce = 1

    def initialize(self, clf):
        self.classifier = clf
        self.numWords = self.classifier.voc.shape[0]
        self.prevStates = np.zeros((self.numSideFrames, self.numWords), "float32")
        self.prevLabels = [0]*self.numSideFrames
        self.prevScores = [0]*self.numSideFrames

    def test_on_video(self):
        vc = self.parent.vc
        while(vc.isOpened()):
            ret,im = vc.read()
            im = cv2.flip(im, 1)
            imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            self.handTracker.colorProfiler.draw_color_windows(im, imhsv)
            cv2.imshow(self.windowName, im)
            k = cv2.waitKey(1)
            if k == 32: # space
                break
            elif k == 27:
                sys.exit(0)

        self.handTracker.colorProfiler.run()
        binaryIm = self.handTracker.get_binary_image(imhsv)
        cnt,hull,centroid,defects = self.handTracker.initialize_contour(binaryIm)
        cv2.namedWindow(self.binaryWindowName)
        cv2.namedWindow(self.handWindowName)
        cv2.namedWindow(self.windowName)
        cv2.setMouseCallback(self.windowName, self.reinforce)

        while(vc.isOpened()):
            ret,im = vc.read()
            im = cv2.flip(im, 1)
            imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            binaryIm = self.handTracker.get_binary_image(imhsv)
            cnt,hull,centroid,defects = self.handTracker.get_contour(binaryIm)
            imCopy = 1*im
            testData = None
            prediction = -1
            score = -1
            update = False
            if cnt is not None:
                numDefects = defects.shape[0]
                cropImage,cropPoints = self.handTracker.get_cropped_image_from_cnt(im, cnt, 0.05)
                cropImageGray = self.handTracker.get_cropped_image_from_points(imgray, cropPoints)
                #cv2.fillPoly(binaryIm, cnt, 255)
                #cropImageBinary = self.handTracker.get_cropped_image_from_points(binaryIm, cropPoints)
                #cropImageGray = self.apply_binary_mask(cropImageGray, cropImageBinary, 5)
                #kp,des = self.featureExtractor.get_keypoints_and_descriptors(cropImageGray)
                kp = self.featureExtractor.get_keypoints(cropImageGray)
                cropCnt = self.handTracker.get_cropped_contour(cnt, cropPoints)
                kp = self.featureExtractor.get_keypoints_in_contour(kp, cropCnt)
                kp,des = self.featureExtractor.compute_descriptors(cropImageGray, kp)
                if des is not None and des.shape[0] >= 0:
                    self.featureExtractor.draw_keypoints(cropImage, kp)
                if des is not None and des.shape[0] >= self.minDescriptorsPerFrame and self.is_hand(defects):
                    words, distance = vq(des, self.classifier.voc)
                    testData = np.zeros(self.numWords, "float32")
                    for w in words:
                        testData[w] += 1
                    normTestData = np.linalg.norm(testData, ord=2) * np.ones(self.numWords)
                    testData = np.divide(testData, normTestData)
                    prediction,score = self.predict(testData)
                    sortedScores = np.sort(score)
                    #if max(score) > self.predictionScoreThreshold:
                    if sortedScores[-1]-sortedScores[-2] >= self.predictionScoreThreshold:
                        self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,255,0))
                    else:
                        self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(255,0,0))
                        prediction = -1
                    update = True
                else:
                    self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,0,255))
                    prediction = -1
                cv2.imshow(self.handWindowName,cropImage)
            else:
                prediction = -1
            #self.insert_to_prediction_list(prediction)
            #prediction,predictionCount = self.most_common(self.predictionList)
            #if prediction>=0:
            writtenVal = '-'
            if prediction > 0:
                #if self.classifier.medianDefects is not None and numDefects>=self.classifier.medianDefects[prediction-1]-1 and numDefects<=self.classifier.medianDefects[prediction-1]+1:
                #    #print prediction
                #    writtenVal = str(prediction)
                #    update = True
                #elif self.classifier.medianDefects is None:
                    #print prediction
                writtenVal = str(prediction)
            self.write_on_image(imCopy, writtenVal)
            cv2.imshow(self.binaryWindowName, binaryIm)
            imCopy = self.add_prev_frames_to_image(imCopy, testData, prediction, score, update)
            cv2.imshow(self.windowName,imCopy)
            k = cv2.waitKey(1)
            if k == 27: # space
                break

    def test_on_descriptors(self, desList):
        testLabels = []
        for i,des in enumerate(desList): 
            if des is not None and des.shape[0] >= self.minDescriptorsPerFrame:
                words, distance = vq(des, self.classifier.voc)
                testData = np.zeros(self.numWords, "float32")
                for w in words:
                    testData[w] += 1
                normTestData = np.linalg.norm(testData, ord=2) * np.ones(self.numWords)
                testData = np.divide(testData, normTestData)
                prediction,score = self.predict(testData)
                sortedScores = np.sort(score)
                    #if max(score) > self.predictionScoreThreshold:
                if sortedScores[-1]-sortedScores[-2] >= self.predictionScoreThreshold:
                    pass
                else:
                    prediction = -1
            else:
                prediction = -1
            testLabels.append(prediction)
        return testLabels

    def predict(self, testData):
        prediction = self.classifier.predict(testData.reshape(1,-1))
        score = self.classifier.decision_function(testData.reshape(1,-1))
        return prediction[0], score[0]

    def insert_to_prediction_list(self, prediction):
        self.predictionList.append(prediction)
        self.predictionList = self.predictionList[1:]

    def most_common(self, lst):
        for i in range(1,len(lst)-1):
            if lst[i] != lst[i-1] and lst[i] != lst[i+1]:
                lst[i] = -1
        e = max(set(lst), key=lst.count)
        return e,lst.count(e)

    def is_hand(self, defects):
        if defects.shape[0] > 5:
            return False
        else:
            return True

    def write_on_image(self, image, text):
        cv2.putText(image, text, (self.parent.imWidth/20,self.parent.imHeight/4), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 5)

    def get_prev_frames_image(self):
        image = self.prevFrameList[0]
        for i in range(1,len(self.prevFrameList)):
            image = np.append(image, self.prevFrameList[i], axis=0)
        return image

    def apply_binary_mask(self, image, mask, kernelSize):
        kernel = np.ones((kernelSize,kernelSize),np.uint8)
        dilatedMask = cv2.dilate(mask,kernel,iterations=1)
        maskedImage = cv2.bitwise_and(image, image, mask=dilatedMask)
        return maskedImage

    def add_prev_frames_to_image(self, image, testData, testLabel, testScore, update=False):
        shrinkIm = cv2.resize(image, None, fx=float(1)/self.numSideFrames, fy=float(1)/self.numSideFrames)
        prevFramesIm = self.get_prev_frames_image()
        image = np.append(image, prevFramesIm, axis=1)
        if update:
            if self.numPrevFrames < self.numSideFrames:
                self.prevFrameList[self.numPrevFrames] = shrinkIm
                self.prevStates[self.numPrevFrames] = testData
                self.prevLabels[self.numPrevFrames] = testLabel
                self.prevScores[self.numPrevFrames] = testScore
                self.numPrevFrames += 1
            else:
                self.prevFrameList = np.append(self.prevFrameList, np.array([shrinkIm]), axis=0)
                self.prevFrameList = self.prevFrameList[1:]
                self.prevStates = np.append(self.prevStates, np.array([testData]), axis=0)
                self.prevStates = self.prevStates[1:]
                self.prevLabels.append(testLabel)
                self.prevLabels = self.prevLabels[1:]
                self.prevScores.append(testScore)
                self.prevScores = self.prevScores[1:]
        return image

    def reinforce(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x > self.parent.imWidth:
                prevFrameID = int(np.floor(y*self.numSideFrames/self.parent.imHeight))
                self.prevFrameList[prevFrameID] = cv2.cvtColor(self.prevFrameList[prevFrameID], cv2.COLOR_BGR2HSV)
                if isinstance(self.classifier, svm.LinearSVC):
                    self.perceptron_update(prevFrameID, False)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if x > self.parent.imWidth:
                prevFrameID = int(np.floor(y*self.numSideFrames/self.parent.imHeight))
                self.prevFrameList[prevFrameID] = cv2.cvtColor(self.prevFrameList[prevFrameID], cv2.COLOR_BGR2YCR_CB)
                if isinstance(self.classifier, svm.LinearSVC):
                    self.perceptron_update(prevFrameID, True)

    def perceptron_update(self, prevFrameID, flag):
        weights = self.classifier.coef_
        if not flag:
            wrongData = self.prevStates[prevFrameID]
            #normData = np.linalg.norm(wrongData, ord=2) * np.ones(self.numWords)
            #wrongData = np.divide(wrongData, normData)
            wrongLabel = self.prevLabels[prevFrameID]
            wrongScores = self.prevScores[prevFrameID]
            wrongScore = max(wrongScores)
            if wrongLabel > 0:
                wrongWeights = weights[wrongLabel-1]
                newWeights = np.subtract(wrongWeights, (self.learningRate/self.numReinforce)*wrongData)
                weights[wrongLabel-1] = newWeights
            else:
                k = cv2.waitKey(-1)
                rightLabel = k - 48
                if rightLabel > 0 and rightLabel <= weights.shape[0]:
                    wrongWeights = weights[rightLabel-1]
                    newWeights = np.add(wrongWeights, (self.learningRate/self.numReinforce)*wrongData)
                    weights[rightLabel-1] = newWeights
        else:
            rightData = self.prevStates[prevFrameID]
            #normData = np.linalg.norm(rightData, ord=2) * np.ones(self.numWords)
            #rightData = np.divide(rightData, normData)
            rightLabel = self.prevLabels[prevFrameID]
            rightScores = self.prevScores[prevFrameID]
            rightScore = max(rightScores)
            if rightLabel > 0:
                rightWeights = weights[rightLabel-1]
                newWeights = np.add(rightWeights, (self.learningRate/self.numReinforce)*rightData)
                weights[rightLabel-1] = newWeights
        #self.numReinforce += 1
        self.classifier.coef_ = weights
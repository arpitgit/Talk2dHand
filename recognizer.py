import numpy as np
from trainer import Trainer
from tester import Tester

class Recognizer(object):
    def __init__(self, vc, opts):
        self.vc = vc
        ret,im = vc.read()
        self.numGestures = opts.num
        self.imHeight,self.imWidth,self.channels = im.shape
        self.trainer = Trainer(numGestures=opts.num, numFramesPerGesture=opts.frames, minDescriptorsPerFrame=opts.desc, numWords=opts.words, descType=opts.type, kernel=opts.kernel, numIter=opts.iter, parent=self)
        self.tester = Tester(numGestures=opts.num, minDescriptorsPerFrame=opts.desc, numWords=opts.words, descType=opts.type, numPredictions=7, parent=self)

    def train_from_video(self):
        self.trainer.extract_descriptors_from_video()
        variance = self.trainer.kmeans()
        self.trainer.bow()
        score = self.trainer.svm()
        return score

    def train_from_descriptors(self, desList, trainLabels):
        self.trainer.desList = desList
        self.trainer.trainLabels = trainLabels
        #numFramesPerGesture = trainLabels.count(1)
        #self.trainer.desList = desList[:numFramesPerGesture*self.numGestures]
        #self.trainer.trainLabels = trainLabels[:numFramesPerGesture*self.numGestures]

        variance = self.trainer.kmeans()
        self.trainer.bow()
        score = self.trainer.svm()
        return score

    def train_from_images(self, gestureDirList, parentDirPath, trainMask, maskParentDirPath):
        self.trainer.extract_descriptors_from_images(gestureDirList, parentDirPath, trainMask, maskParentDirPath)
        variance = self.trainer.kmeans()
        self.trainer.bow()
        score = self.trainer.svm()
        return score

    def test_on_video(self, clf):
        #print clf.coef_
        self.tester.initialize(clf)
        self.tester.test_on_video()
    
    def test_on_descriptors(self, clf, descList, trueLabels):
        #numFramesPerGesture = trueLabels.count(1)
        #descList = descList[:numFramesPerGesture*self.numGestures]
        #trueLabels = trueLabels[:numFramesPerGesture*self.numGestures]
        self.tester.initialize(clf)
        testLabels = self.tester.test_on_descriptors(descList)
        matchList = [i for i, j in zip(trueLabels, testLabels) if i == j]
        score = float(len(matchList))/len(trueLabels)
        return score
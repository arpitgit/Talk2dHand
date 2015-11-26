import sys
import os
import cv2
import numpy as np
from scipy.cluster.vq import vq
from recognizer import Recognizer
import cPickle as pickle

def get_new_directory(numGestures, descType):
    i = 0
    while(1):
        trainDirName = "{0}_{1}_train{2}".format(numGestures, descType, i)
        trainDirPath = get_traindir_path(trainDirName)
        if not os.path.exists(trainDirPath):
            return trainDirName
        i += 1

def get_traindir_path(dirName):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainData", dirName)

def get_gesture_parentdir_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "GestureImages")

def get_gesture_mask_parentdir_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "GestureMasks")

def pickle_files(outTrainDir, trainer):
    outTrainDirPath = get_traindir_path(outTrainDir)
    if not os.path.exists(outTrainDirPath):
        os.makedirs(outTrainDirPath)
    descriptorFile = os.path.join(outTrainDirPath, "descriptors.pkl")
    classifierFile = os.path.join(outTrainDirPath, "classifier.pkl")
    with open(descriptorFile, 'wb') as output:
        desList = trainer.desList
        pickle.dump(desList, output, pickle.HIGHEST_PROTOCOL)
        trainLabels = trainer.trainLabels
        pickle.dump(trainLabels, output, pickle.HIGHEST_PROTOCOL)
    with open(classifierFile, 'wb') as output:
        clf = trainer.classifier
        pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
    save_first_frames(outTrainDirPath, recognizer.trainer.firstFrameList)
    return clf

def get_full_descriptors_and_labels(dirNamelist):
    desList = []
    trainLabelsList = []
    for dirName in dirNamelist:
        dirPath = get_traindir_path(dirName)
        descFile = os.path.join(dirPath, "descriptors.pkl")
        if not os.path.exists(descFile):
            print "Descriptor file not present for directory {0}".format(dirName)
            continue
        with open(descFile, 'rb') as input:
            des = pickle.load(input)
            trainLabels = pickle.load(input)
        desList += des
        trainLabelsList += trainLabels
    return desList,trainLabelsList

def save_first_frames(dirPath, frameList):
    for i,im in enumerate(frameList):
        filepath = os.path.join(dirPath, "gesture_{0}.jpg".format(i+1))
        cv2.imwrite(filepath, im)

def cmd_parser():
    """
    Parse user's command line and return opts object and args
    """
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-n", "--num",        help="Number of gestures", default=3, type="int")
    parser.add_option("-f", "--frames",     help="Number of frames to train on per gesture", default=100, type="int")
    parser.add_option("-w", "--words",      help="Number of visual words", default=100, type="int")
    parser.add_option("-d", "--desc",       help="Minimum number of descriptors per frame", default=100, type="int")
    parser.add_option("-t", "--type",       help="Descriptor type", action="store", type="string", default="surf")
    parser.add_option("-k", "--kernel",     help="Kernel type", action="store", type="string", default="linear")
    parser.add_option("-i", "--iter",       help="Number of iterations for k-means clustering", type="int", default=30)
    parser.add_option(      "--doc",        help="Print the docstring", action="store_true", default=False)
    parser.add_option(      "--notrain",    help="Whether to train the system", action="store_true", default=False)
    parser.add_option(      "--nocollect",  help="Whether to collect train descriptors", action="store_true", default=False)
    parser.add_option(      "--notest",     help="Whether to run the system in test mode", action="store_true", default=False)  
    parser.add_option(      "--traindir",   help="Training directory(ies)", action="store", type="string")
    parser.add_option(      "--testdir",    help="Test directory", action="store", type="string")
    parser.add_option(      "--trainmask",  help="Type of masking for image data", default=0, type="int")
    return parser.parse_args()

def process_opts(opts):
    if opts.notrain:
        opts.nocollect = True
    if opts.notrain:
        if opts.traindir is None:
            print "Specify training directory with --traindir"
            exit(0)
        inTrainDirs = opts.traindir.split(',')[0]
        inputMode = None
    elif opts.nocollect:
        if opts.traindir is None:
            print "Specify training directory with --traindir"
            exit(0)
        inTrainDirs = opts.traindir.split(',')
        inputMode = "descriptors"
    else:
        if opts.traindir is None:
            inTrainDirs = None
            inputMode = "video"
        else:
            inTrainDirs = opts.traindir.split(',')
            inputMode = "images"
    return inputMode,inTrainDirs

def process_test_opts(opts):
    if opts.testdir is None:
        outTestDirs = None
        outputMode = "video"
    else:
        outTestDirs = opts.testdir.split(',')
        outputMode = "descriptors"
    return outputMode,outTestDirs

def get_relevant_objects(inputMode, inTrainDirs, opts):
    if inputMode == "video":
        return
    elif inputMode == "descriptors":
        descList,trainLabels = get_full_descriptors_and_labels(inTrainDirs)
        trueNumGestures = len(set(trainLabels))
        if opts.num != trueNumGestures:
            print "Taking number of gestures = {0}".format(trueNumGestures)
            opts.num = trueNumGestures
        return descList,trainLabels
    elif inputMode == "images":
        gestureParentDirPath = get_gesture_parentdir_path()
        trueNumGestures = len(inTrainDirs)
        if opts.num != trueNumGestures:
            print "Taking number of gestures = {0}".format(trueNumGestures)
            opts.num = trueNumGestures
        if opts.trainmask != 0:
            gestureMaskParentDirPath = get_gesture_mask_parentdir_path()
        else:
            gestureMaskParentDirPath = None
        return gestureParentDirPath,gestureMaskParentDirPath,opts.trainmask
    else:
        return
            
#########################
### Main script entry ###
#########################
if __name__ == "__main__":
    opts,args = cmd_parser()
    inputMode,inTrainDirs = process_opts(opts)    
    vc = cv2.VideoCapture(0)
    try:
        if inputMode == "video":
            recognizer = Recognizer(vc=vc, opts=opts)
            score = recognizer.train_from_video()
            print "Training score = {0}".format(score)
            outTrainDir = get_new_directory(opts.num, opts.type)
            clf = pickle_files(outTrainDir, recognizer.trainer)
        elif inputMode == "descriptors":
            descList,trainLabels = get_relevant_objects(inputMode, inTrainDirs, opts)
            recognizer = Recognizer(vc=vc, opts=opts)
            score = recognizer.train_from_descriptors(descList, trainLabels)
            print "Training score = {0}".format(score)
            if len(inTrainDirs) == 1:
                outTrainDir = inTrainDirs[0]
            else:
                outTrainDir = get_new_directory(opts.num, opts.type)
            clf = pickle_files(outTrainDir, recognizer.trainer)
        elif inputMode == "images":
            parentDirPath, maskParentDirPath, trainMask = get_relevant_objects(inputMode, inTrainDirs, opts)
            recognizer = Recognizer(vc=vc, opts=opts)
            score = recognizer.train_from_images(inTrainDirs, parentDirPath, trainMask, maskParentDirPath)
            print "Training score = {0}".format(score)
            outTrainDir = get_new_directory(opts.num, opts.type)
            clf = pickle_files(outTrainDir, recognizer.trainer)
        else:
            recognizer = Recognizer(vc=vc, opts=opts)
            outTrainDir = inTrainDirs
            inTrainDirPath = get_traindir_path(inTrainDirs)
            classifierFile = os.path.join(inTrainDirPath, "classifier.pkl")
            if not os.path.exists(classifierFile):
                print "Trained classifier not present for directory {0}".format(inTrainDirs[0])
                exit(0)
            with open(classifierFile, 'rb') as input:
                clf = pickle.load(input)

        if not opts.notest:
            outputMode,outTestDirs = process_test_opts(opts)
            if outputMode == "video": 
                recognizer.test_on_video(clf)
            else:
                descList,trueLabels = get_relevant_objects(outputMode, outTestDirs, opts)
                score = recognizer.test_on_descriptors(clf, descList, trueLabels)
                #matchList = [i for i, j in zip(trueLabels, testLabels) if i == j]
                #score = float(len(matchList))/len(trueLabels)
                print "Test score = {0}".format(score)
    except:
        vc.release()
        import traceback
        traceback.print_exc(file=sys.stdout)
    
    if 'outTrainDir' in locals():
        print "Train directory = {0}".format(outTrainDir)
    
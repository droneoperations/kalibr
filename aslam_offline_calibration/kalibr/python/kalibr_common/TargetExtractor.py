import sm

import numpy as np
import sys
import multiprocessing
try:
   import queue
except ImportError:
   import Queue as queue # python 2.x
import time
import copy
import cv2

def multicoreExtractionWrapper(detector, taskq, resultq, clearImages, noTransformation):    
    while True:
        try:
            task = taskq.get(timeout=5)
        except queue.Empty:
            return
        idx = task[0]
        stamp = task[1]
        image = task[2]
        
        if noTransformation:
            success, obs = detector.findTargetNoTransformation(stamp, np.array(image))
        else:
            success, obs = detector.findTarget(stamp, np.array(image))
            
        if clearImages:
            obs.clearImage()
        if success:
            resultq.put( (obs, idx) )

def extractCornersFromDataset(dataset, detector, multithreading=False, numProcesses=None, clearImages=True, noTransformation=False):
    print("Extracting calibration target corners")    
    targetObservations = []
    numImages = dataset.numImages()
    
    # prepare progess bar
    iProgress = sm.Progress2(numImages)
    iProgress.sample()
    
    if multithreading:   
        if not numProcesses:
            numProcesses = max(1, multiprocessing.cpu_count())
            clearImages = True
        try:      
            resultq = multiprocessing.Queue()
            taskq = multiprocessing.Queue(1)
                
            plist=list()
            for pidx in range(0, numProcesses):
                p = multiprocessing.Process(target=multicoreExtractionWrapper, args=(detector, taskq, resultq, clearImages, noTransformation, ))
                p.start()
                plist.append(p)
            
            for idx, (timestamp, image) in enumerate(dataset.readDataset()):
                taskq.put( (idx, timestamp, image) )
            
            #wait for results
            last_done=0
            while 1:
                if all([not p.is_alive() for p in plist]):
                    time.sleep(0.1)
                    break
                done = numImages-taskq.qsize()
                sys.stdout.flush()
                if (done-last_done) > 0:
                    iProgress.sample(done-last_done)
                last_done = done
                time.sleep(0.5)
            resultq.put('STOP')
        except Exception as e:
            raise RuntimeError("Exception during multithreaded extraction: {0}".format(e))
        
        #get result sorted by time (=idx)
        if resultq.qsize() > 1:
            targetObservations = [[]]*(resultq.qsize()-1)
            for lidx, data in enumerate(iter(resultq.get, 'STOP')):
                obs=data[0]; time_idx = data[1]
                targetObservations[lidx] = (time_idx, obs)
            targetObservations = list(zip(*sorted(targetObservations, key=lambda tup: tup[0])))[1]
        else:
            targetObservations=[]
    
    #single threaded implementation
    else:
        for timestamp, image in dataset.readDataset():
            if noTransformation:
                success, observation = detector.findTargetNoTransformation(timestamp, np.array(image))
            else:
                success, observation = detector.findTarget(timestamp, np.array(image))
            if clearImages:
                observation.clearImage()
            if success == 1:
                targetObservations.append(observation)
            iProgress.sample()

    if len(targetObservations) == 0:
        print("\r")
        sm.logFatal("No corners could be extracted for camera {0}! Check the calibration target configuration and dataset.".format(dataset.topic))
    else:    
        print("\r  Extracted corners for %d images (of %d images)                              " % (len(targetObservations), numImages))

    #close all opencv windows that might be open
    cv2.destroyAllWindows()
    
    return targetObservations

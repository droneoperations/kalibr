import sm

import numpy as np
import multiprocessing
try:
   import queue
except ImportError:
   import Queue as queue # python 2.x
import time

from tqdm import tqdm

def multicoreExtractionWrapper(detector, taskq, resultq, clearImages, noTransformation, finish):
    while not finish.is_set():
        try:
            task = taskq.get(timeout=1)
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
    clearImages = True
    if multithreading:
        if not numProcesses:
            numProcesses = max(1, multiprocessing.cpu_count())
        try:
            resultq = multiprocessing.Queue()
            taskq = multiprocessing.Queue(1)
            plist=list()
            finish = multiprocessing.Event()
            for pidx in range(0, numProcesses):
                p = multiprocessing.Process(target=multicoreExtractionWrapper, args=(detector, taskq, resultq, clearImages, noTransformation, finish))
                p.start()
                plist.append(p)
            
            for idx, (timestamp, image) in tqdm(enumerate(dataset.readDataset()), total=numImages):
                taskq.put((idx, timestamp, image))
            finish.set()
            time.sleep(2)
            resultq.put('STOP')
        except Exception as e:
            raise RuntimeError("Exception during multithreaded extraction: {0}".format(e))
        
        print("I finished processing corners")
        #get result sorted by time (=idx)
        if resultq.qsize() > 1:
            targetObservations = []
            for data in iter(resultq.get, 'STOP'):
                obs=data[0]; time_idx = data[1]
                targetObservations.append((time_idx, obs))
            targetObservations = list(zip(*sorted(targetObservations, key=lambda tup: tup[0])))[1]
        else:
            targetObservations=[]
    
    #single threaded implementation
    else:
        for timestamp, image in tqdm(dataset.readDataset()):
            if noTransformation:
                success, observation = detector.findTargetNoTransformation(timestamp, np.array(image))
            else:
                success, observation = detector.findTarget(timestamp, np.array(image))
            if clearImages:
                observation.clearImage()
            if success == 1:
                targetObservations.append(observation)

    if len(targetObservations) == 0:
        print("\r")
        sm.logFatal("No corners could be extracted for camera {0}! Check the calibration target configuration and dataset.".format(dataset.topic))
    else:    
        print("\r  Extracted corners for %d images (of %d images)                              " % (len(targetObservations), numImages))

    #close all opencv windows that might be open
    return targetObservations

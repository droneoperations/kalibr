import sm
import numpy as np
import sys
import multiprocessing
import time
import copy
import cv2
import queue

def multicoreExtractionWrapper(detector, taskq, resultq, clearImages, noTransformation):    
    while True:
        try:
            task = taskq.get(timeout=1)  # Wait for tasks with a timeout
            if task == 'STOP':
                break
        except queue.Empty:
            continue  # Continue if the queue is temporarily empty

        idx, stamp, image = task
        success, obs = (detector.findTargetNoTransformation(stamp, np.array(image)) 
                        if noTransformation else detector.findTarget(stamp, np.array(image)))
        if clearImages:
            obs.clearImage()
        resultq.put((success, obs, idx))

def task_feeder(dataset, taskq):
    for idx, (timestamp, image) in enumerate(dataset.readDataset()):
        taskq.put((idx, timestamp, image))
    for _ in range(multiprocessing.cpu_count()):
        taskq.put('STOP')  # Signal to workers that there are no more tasks

def extractCornersFromDataset(dataset, detector, multithreading=False, numProcesses=None, clearImages=True, noTransformation=False):
    print("Extracting calibration target corners")    
    targetObservations = []
    numImages = dataset.numImages()

    # Initialize progress
    iProgress = sm.Progress2(numImages)
    iProgress.sample()
            
    if multithreading:
        if not numProcesses:
            numProcesses = max(1, multiprocessing.cpu_count())
        try:
            # Queues for task distribution and result collection
            resultq = multiprocessing.Queue()
            taskq = multiprocessing.Queue(numProcesses)  # Limit task queue size for lazy loading
            
            # Start task feeder in a separate process to avoid preloading all data
            feeder = multiprocessing.Process(target=task_feeder, args=(dataset, taskq))
            feeder.start()
            
            # Start worker processes
            plist = []
            for _ in range(numProcesses):
                p = multiprocessing.Process(target=multicoreExtractionWrapper, args=(detector, taskq, resultq, clearImages, noTransformation))
                p.start()
                plist.append(p)
            
            # Collect results
            for _ in range(numImages):
                success, obs, idx = resultq.get()
                if success:
                    targetObservations.append((idx, obs))
                iProgress.sample()
            
            # Wait for task feeder and worker processes to finish
            feeder.join()
            for p in plist:
                p.join()
            
            # Sort observations by index
            targetObservations.sort(key=lambda tup: tup[0])
            targetObservations = [obs for _, obs in targetObservations]
        
        except Exception as e:
            raise RuntimeError(f"Exception during multithreaded extraction: {e}")

    else:
        # Single-threaded version with lazy loading
        for timestamp, image in dataset.readDataset():
            success, observation = (detector.findTargetNoTransformation(timestamp, np.array(image)) 
                                    if noTransformation else detector.findTarget(timestamp, np.array(image)))
            if clearImages:
                observation.clearImage()
            if success:
                targetObservations.append(observation)
            iProgress.sample()

    if not targetObservations:
        print("\r")
        sm.logFatal(f"No corners could be extracted for camera {dataset.topic}! Check the calibration target configuration and dataset.")
    else:    
        print(f"\r  Extracted corners for {len(targetObservations)} images (of {numImages} images)")

    # Close any OpenCV windows that might be open
    cv2.destroyAllWindows()
    
    return targetObservations

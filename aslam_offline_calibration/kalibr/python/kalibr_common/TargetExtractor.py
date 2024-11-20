import sm
import numpy as np
import multiprocessing
import pickle
import hashlib
import os
import time
import cv2
import queue

CACHE_FILENAME = "calibration_cache.pkl"

def compute_hash(dataset):
    """Compute a hash of the dataset and detector configuration."""
    hash_obj = hashlib.sha256()
    # Assuming `dataset` and `detector` have identifiable attributes for hashing
    hash_obj.update(str(dataset.numImages()).encode())
    hash_obj.update(str(dataset.topic).encode())
    # Include any other dataset or detector-specific configurations
    return hash_obj.hexdigest()

def save_cache(cache_path, metadata, observations):
    """Save observations and metadata to a pickle file."""
    with open(cache_path, "wb") as f:
        pickle.dump({"metadata": metadata, "observations": observations}, f)

def load_cache(cache_path):
    """Load observations and metadata from a pickle file."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)

def multicoreExtractionWrapper(detector, taskq, resultq, clearImages, noTransformation):    
    while True:
        try:
            task = taskq.get(timeout=1)
            if task == 'STOP':
                break
        except queue.Empty:
            continue

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
        taskq.put('STOP')

def extractCornersFromDataset(dataset, detector, multithreading=False, numProcesses=None, clearImages=True, noTransformation=False):
    print("Extracting calibration target corners")
    cache_path = CACHE_FILENAME
    metadata = {"hash": compute_hash(dataset), "timestamp": time.time()}

    # Check for cached results
    # if os.path.exists(cache_path):
    #     try:
    #         cached_data = load_cache(cache_path)
    #         if cached_data["metadata"]["hash"] == metadata["hash"]:
    #             print("Cache found and matches the current configuration. Loading results...")
    #             print(f"Loaded corners for {len(cached_data['observations'])} images")
    #             return cached_data["observations"]
    #         else:
    #             print("Cache found but does not match the current configuration. Recomputing...")
    #     except Exception as e:
    #         print(f"Failed to load cache: {e}. Recomputing...")

    targetObservations = []
    numImages = dataset.numImages()

    iProgress = sm.Progress2(numImages)
    iProgress.sample()
    clearImages = True
    if multithreading:
        if not numProcesses:
            numProcesses = max(1, multiprocessing.cpu_count())
        try:
            resultq = multiprocessing.Queue()
            taskq = multiprocessing.Queue(numProcesses)
            
            feeder = multiprocessing.Process(target=task_feeder, args=(dataset, taskq))
            feeder.start()
            
            plist = []
            for _ in range(numProcesses):
                p = multiprocessing.Process(target=multicoreExtractionWrapper, args=(detector, taskq, resultq, clearImages, noTransformation))
                p.start()
                plist.append(p)
            
            for _ in range(numImages):
                success, obs, idx = resultq.get()
                if success:
                    targetObservations.append((idx, obs))
                iProgress.sample()
            
            feeder.join()
            for p in plist:
                p.join()
            
            targetObservations.sort(key=lambda tup: tup[0])
            targetObservations = [obs for _, obs in targetObservations]
        
        except Exception as e:
            raise RuntimeError(f"Exception during multithreaded extraction: {e}")

    else:
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

    # Save results to cache
    # save_cache(cache_path, metadata, targetObservations)

    # Close OpenCV windows
    cv2.destroyAllWindows()

    return targetObservations

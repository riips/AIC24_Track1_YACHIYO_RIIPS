import os
import numpy as np


def load_tracking(file_name):
    tracking_file = open(file_name)
    result = {}
    for line in tracking_file:
        line = line.rstrip().split(',')
        frame_id = int(line[1])
        track_id = int(line[2])
        bbox = [float(line[3]), float(line[4]), float(line[5]), float(line[6]), 1.0]
        if frame_id not in result.keys():
            result[frame_id] = []
        result[frame_id].append({'bbox': np.array(bbox)})
    return result

def load_tracking_id(file_name):
    tracking_file = open(file_name)
    result = {}
    for line in tracking_file:
        line = line.rstrip().split(',')
        frame_id = int(line[1])
        track_id = int(line[2])
        if frame_id not in result.keys():
            result[frame_id] = []
        result[frame_id].append({'track_id': track_id})
    return result
    
if __name__ == '__main__':
    print("run load_tracking")
    #load_tracking('')

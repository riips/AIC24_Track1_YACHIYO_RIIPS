import os
import numpy as np
import json
import glob


class DetectedObjects:
    """
    Represents whole detected objects to track.
    Object dict is built by frame_id as a key and its entity contains a list of all Detected objects of the frame. 
    """
    def __init__(self):
        self.num_objects = 0
        self.objects = {}
        self._objects_registered = {}
        #self.scene_id = scene_id
        #self.camera_id = -1
        self.camera_projection_matrix = None
        self.homography_matrix = None

    def __str__(self):
        return f"DetectedObjects: scene_id:{self.scene_id}, camera_id:{self.camera_id}, num_objects:{self.num_objects}"

    def load_from_directory(self, feature_root, calibration_path="Calibration"):
        if not os.path.isdir(feature_root):
            raise Exception(f'There is no directory to read from. {feature_root}')
        npys = sorted(glob.glob(os.path.join(feature_root, "**/*.npy"), recursive=True))
        scene_id = None
        camera_id = None
        path_list = feature_root.split("/")
        for dir in path_list:  
            if dir.startswith("scene_"):  
                scene_id = int(dir.replace("scene_",""))
            if dir.startswith("camera_"):  
                camera_id = int(dir.replace("camera_",""))
        if scene_id is not None and camera_id is not None:
            calibration_path = f"Original/scene_{scene_id:03d}/camera_{camera_id:04d}/calibration.json"
            self.load_calibration(calibration_path)
        else:
            print(f'\033[33mwarning\033[0m : failed to get scene_id and camera_id from feature path.')
            print(f'\033[33mwarning\033[0m : world coordinate calculations are ignored.')


        # Below is to parse camera id from the path, we're probably not going to use it though.
        #camera_id = None
        #dirs = npys[0].split("/")
        #if len(dirs) < 2:
        #    print(f"Cannot prop camera id from input path. {feature_path}")
        #else:
        #    camera_id = dirs[-1]
        #    if "Camera" in camera_id:
        #        self.camera_id = int(camera_id[len("Camera"):])
        
        for f in npys:
            self.add_object_from_image_path(f)

    def add_object(self, frame_id, coordinate, world_coordinate, confidence, feature_path, image_path=None):
        if isinstance(frame_id, str):
            frame_id = int(frame_id)

        # Check if coordinate is reasonable
        if coordinate.x1 >= coordinate.x2 or coordinate.y1 >= coordinate.y2:
            print(f"Unnatural coordinate found in frame {frame_id}: {coordinate}")
            return

        detected_obj = DetectedObject(object_id=self.num_objects, frame_id=frame_id, coordinate=coordinate, worldcoordinate=world_coordinate,
                                      confidence=confidence, feature_path=feature_path)
        key = f"{coordinate.x1}_{coordinate.y1}_{coordinate.x2}_{coordinate.y2}"
        if frame_id in self.objects:
            if not key in self._objects_registered[frame_id]:
                objects_per_frame = self.objects[frame_id].append(detected_obj)
                self._objects_registered[frame_id].append(key)
            else:
                print(f"Duplicate coord found in frame {frame_id}: {coordinate}")
                return
        else:
            objects_per_frame = self.objects[frame_id] = [detected_obj]
            self._objects_registered[frame_id] = [key]
        self.num_objects += 1

    def add_object_from_image_path(self, feature_path, image_path=None, calibration_path="Calibration"):
        file_path = os.path.basename(feature_path)
        if file_path.startswith("feature_"):
            _, frame_id, serial_no, x1, x2, y1, y2, conf = os.path.splitext(file_path)[0].split("_")
            conf = conf if len(conf) == 1 else conf[0]+"."+conf[1:]
        else:
            serial_no, frame_id, x1, x2, y1, y2 = os.path.splitext(file_path)[0].split("_")
            x1, x2, y1, y2 = int(x1.replace("x","")), int(x2), int(y1.replace("y","")), int(y2)
            conf = 0.98765 # Dummy
        World_coordinate = None
        if self.homography_matrix is not None:
            w_x, w_y = self.convert_coordinates_2world((int(float(x1)) + int(float(x2))) / 2, int(float(y2)))
            World_coordinate = WorldCoordinate(w_x, w_y)

        self.add_object(frame_id=int(frame_id), coordinate=Coordinate(x1, y1, x2, y2), world_coordinate=World_coordinate,
                        confidence=float(conf), feature_path=feature_path, image_path=image_path)

    def get_objects_of_frames(self, start_frame, end_frame):
        if start_frame > self.num_frames() or end_frame > self.num_frames():
            return None
        object_dict = {}
        for frame_id in range(start_frame, end_frame):
            if frame_id in self.objects:
                object_dict[frame_id] = self[frame_id]
            #else:
            #    print(f"There is no such frame in the DetectedObjects, will be ignored. frame_id: {frame_id}")
        return object_dict

    def get_object_ids_of_frames(self, start_frame, end_frame):
        """
        Returns a list of detected object IDs that appeared within the specified frame window.
        """
        if start_frame > self.num_frames() or end_frame > self.num_frames():
            return None
        object_ids = []
        for frame_id in range(start_frame, end_frame):
            if frame_id in self.objects:
                for det in self[frame_id]:
                    object_ids.append(det.object_id)
        return sorted(object_ids)

    def __getitem__(self, frame_id):
        if frame_id in self.objects:
            return self.objects[frame_id]
        else:
            return None

    def num_frames(self):
        """
        Returns number of frames that currently holding.
        """
        return len(self.objects)

    def last_frame_id(self):
        """
        Returns the last frame id.
        """
        return max(self.objects.keys())

    def to_trackingdict(self):
        """
        Compatibility function to convert detections in TrackingDict format.
        """
        track_dict = {}
        for frame_id in self.objects:
            for detected_object in self.objects[frame_id]:
                serial_no = detected_object.object_id
                coordinate = json.loads(detected_object.coordinate.__str__())
                if detected_object.worldcoordinate.__str__() != "None":
                    world_coordinate = json.loads(detected_object.worldcoordinate.__str__())
                else:
                    world_coordinate = None
                new_object = { "Frame": frame_id, "NpyPath": detected_object.feature_path,
                               "Coordinate": coordinate, "WorldCoordinate": world_coordinate,  "OfflineID": -1 } #"ClusterID": None,
                track_dict[serial_no] = new_object
        return track_dict

    def load_calibration(self, calib_path):
        if os.path.isfile(calib_path):
            with open(calib_path, 'r') as file:
                data = json.load(file)
                self.camera_projection_matrix = np.array(data["camera projection matrix"])
                self.homography_matrix =  np.array(data["homography matrix"])
        else:
            print(f'\033[33mwarning\033[0m : not found Calibration File.')
            print(f'\033[33mwarning\033[0m : world coordinate calculations are ignored.')

    def convert_coordinates_2world(self, x, y):
        vector_xyz = np.array([x, y, 1]) # z=1
        vector_xyz_3d = np.dot(np.linalg.inv(self.homography_matrix), vector_xyz.T)
        return vector_xyz_3d[0] / vector_xyz_3d[2], vector_xyz_3d[1] / vector_xyz_3d[2]
        
class DetectedObject:
    """
    Represents individual detected object to track.
    """
    def __init__(self, object_id, frame_id, coordinate, confidence, worldcoordinate, feature_path, image_path=None):
        self.object_id = f"{object_id:08d}" # AKA serial number
        self.frame_id = frame_id
        self.feature_path = feature_path
        self.confidence = confidence
        self.image_path = image_path
        if isinstance(coordinate, Coordinate):
            self.coordinate = coordinate
        elif isinstance(coordinate, (list, tuple)) and len(coordinate) == 4:
            self.coordinate = Coordinate(*coordinate)
        else:
            raise Exception(f"Unknown coordinate format: {coordinate}")

        if isinstance(worldcoordinate, WorldCoordinate):
            self.worldcoordinate = worldcoordinate
        elif isinstance(worldcoordinate, (list, tuple)) and len(worldcoordinate) == 4:
            self.worldcoordinate = WorldCoordinate(*worldcoordinate)
        else:
            self.worldcoordinate = None

class Coordinate:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(float(x1))
        self.y1 = int(float(y1))
        self.x2 = int(float(x2))
        self.y2 = int(float(y2))

    def __str__(self):
        return(f'{{"x1":{self.x1}, "y1":{self.y1}, "x2":{self.x2}, "y2":{self.y2}}}')

class WorldCoordinate:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return(f'{{"x":{self.x}, "y":{self.y}}}')

class TrackingCluster:
    def __init__(self, camera_id, offline_id):
        self.camera_id = camera_id
        self.offline_id = 0
        self.global_offline_id = -1
        self.clusters = {}
        self.serials = []

    def add(self, serial):
        if serial in self.serials:
            raise Exception("DUP!")
        self.serials.append(serial)
        

class TrackingClusters:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.clusters = []
        self.offline_ids = []

    def add(self, cluster: TrackingCluster):
        cl_id = cluster.offline_id
        if cl_id in self.offline_ids:
            raise Exception("DUP!")
        else:
            self.clusters.append(cluster)

    def get(self, cluster_id):
        if not cluster_id in self.offline_ids:
            raise Exception("No cluster_id registered. {cluster_id}")
        else:
            return self.clusters[offline_ids.index(cluster_id)]

class feature_vector_shed:
    def __init__(self):
        self.features = {}

    def add_vector(self, camera_id, serial_no, npy_path):
        key = camera_id + "_" + serial_no
        if key in self.features:
            print(f"Feature vector of camera ID '{camera_id}' and serial no '{serial_no}' is already exist. ")
            return
            
        if not os.path.isfile(npy_path):
            print(f"The feature vector file '{npy_path}' does not exist. ")
            return
        feature = np.load(npy_path)
        self.features[key] = feature

    def get(self, camera_id, serial_no):
        key = camera_id + "_" + serial_no
        return self.features[key]

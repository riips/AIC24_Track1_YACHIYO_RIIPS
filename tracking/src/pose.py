import os
import numpy as np
import json
import cv2

class PoseKeypoints:
    def __init__(self, keypoint_json):
        self.kp_indice_foot = [15, 16] # ankles
        self.kp_indice_torso = [5, 6, 11, 12, 13, 14] # shoulders, hips, knees
        self.kp_indice_torso_legs = [5, 6, 11, 12, 13, 14, 15, 16] # shoulders, hips, knees, ankles

        self._parse_keypoint_json(keypoint_json)
        self.serial_dict = {}

    def _parse_keypoint_json(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            self.keypoints = data
        else:
            raise Exception(f"Keypoint json file '{file_path}' does not exist.")

    def filter(self, keypoints=None, score_thr=0.3, target_parts="torso_legs", max_frames=0):
        filtered = {}
        if keypoints == None:
            keypoints = self.keypoints
        for i, frame in enumerate(keypoints):
            if max_frames != 0 and i >= max_frames:
                break
            detections = keypoints[frame]
            target_indices = self.kp_indice_torso if target_parts == "torso" else self.kp_indice_torso_legs
            for det in detections:
                kps = det["keypoints"]
                confidences = [k for i2, k in enumerate(kps) if i2 in target_indices and k[2] >= score_thr]
                if len(confidences) < (len(target_indices)):
                    continue

                pose_entity = [det["bbox"], ]
                if int(frame) in filtered:
                    filtered[frame].append(det)
                else:
                    filtered[frame] = [det]
        print(f"Num of filtered results: {len(filtered)}")
        return filtered

    def summary(self): # Just show top_n data
        if len(self.keypoints) <= 0:
            print(f"Empty keypoints")
            return
        print(f"Number of frames: {len(self.keypoints)}")

    def get_keypoints(self, serial:str):
        """
        This must be called after assign_serial_from_tracking_dict() was called,
        as it builds a dictionary with "serial" number as keys.
        
        serial: zero-filled 8-digit string
        """
        if isinstance(serial, int):
            serial = f"{serial:08d}"
        elif isinstance(serial, str) and len(serial) != 8:
            serial = f"{int(serial):08d}"
        if len(self.serial_dict) <= 0:
            raise Exception(f"Serial based dictionary is not built yet.")
        if serial in self.serial_dict:
            return self.serial_dict[serial]
        else:
            return None

    def _build_serial_dict(self, keypoints=None):
        """
        This must be called after assign_serial_from_tracking_dict() was called,
        as it builds a dictionary with "serial" number as keys.
        """
        if len(keypoints) == 0:
            return None
        serial_dict = {}
        if keypoints == None:
            keypoints = self.keypoints
        for i, frame in enumerate(keypoints):
            detections = keypoints[frame]
            for det in detections:
                if "serial" in det: 
                    serial = det["serial"]
                    if serial in serial_dict:
                        print(f"DUP in serial numbers!!")
                    else:
                        serial_dict[serial] = {"bbox": det["bbox"], "Keypoints": det["keypoints"]}
        self.serial_dict = serial_dict

        return self.serial_dict

    def assign_serial_from_tracking_dict(self, tracking_dict, keypoints=None):
        """
        tracking_dict: dictionary of tracking_dict or path to tracking_dict json file.
        """
        if keypoints == None:
            keypoints = self.keypoints
        if isinstance(tracking_dict, str):
            if os.path.isfile(tracking_dict):
                with open(tracking_dict) as f:
                    tracking_dict = json.load(f)
        tracking_coord = {}
        for serial in tracking_dict:
            td_coord = tracking_dict[serial]["Coordinate"]
            td_frame = tracking_dict[serial]["Frame"]
            key = f"{td_frame}_{td_coord['x1']}_{td_coord['y1']}_{td_coord['x2']}_{td_coord['y2']}"
            if key in tracking_coord:
                continue #raise Exception(f"DUP! {key}")
            tracking_coord[key] = serial
        for frame in keypoints:
            detections = keypoints[frame]
    
            for det in detections:
                bbox = det["bbox"]
                key = f"{int(frame)}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"
                if key in tracking_coord:
                    det["serial"] = tracking_coord[key]
                else:
                    #print(f"No tracking found for bbox: {key}, {bbox}")
                    pass

        # Build dict with serial as key
        return self._build_serial_dict(keypoints=keypoints)

    def show_footpoints(self, keypoints=None, frame_img_root="Frames", output_mp4=None, score_thr=0.3, target_parts="torso_legs", max_frames=0): # Generate mp4
        # Creating mp4
        if output_mp4 == None:
            output_mp4 = f"foot_points.mp4"
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_wtr  = cv2.VideoWriter(output_mp4, fourcc=fourcc, fps=30.0, frameSize=(1280, 960))
        if not video_wtr.isOpened():
            print(f"Cannot open video writer.")
            return

        filtered = self.filter(keypoints=keypoints, score_thr=score_thr, target_parts=target_parts, max_frames=max_frames)
        for frame in filtered:
            # Read frame image file
            frame_img_path = os.path.join(frame_img_root, f"{int(frame):06d}.jpg")
            frame_img = cv2.imread(frame_img_path)
            detections = self.keypoints[frame]
            target_indices = self.kp_indice_torso if target_parts == "torso" else self.kp_indice_torso_legs
            for det in detections:
                keypoints = det["keypoints"]
                left_ankle, right_ankle = keypoints[self.kp_indice_foot[0]], keypoints[self.kp_indice_foot[1]]
                if float(left_ankle[2]) >= score_thr:
                    color = (0, 255, 0)
                else:
                    #print(f"Low confidence on KP[15]: {float(fp1[2])}")
                    color = (0, 0, 255)
                cv2.circle(frame_img, (int(left_ankle[0]), int(left_ankle[1])), 5, color, 3)
                
                if float(right_ankle[2]) >= score_thr:
                    color = (0, 255, 0)
                else:
                    #print(f"Low confidence on KP[16]: {float(fp2[2])}")
                    color = (0, 0, 255)
                cv2.circle(frame_img, (int(right_ankle[0]), int(right_ankle[1])), 5, color, 3)
            frame_img = cv2.resize(frame_img, (1280, 960))
            video_wtr.write(frame_img)
        video_wtr.release()
        print(f"Saved video file: {output_mp4}\n")

    def show_footpoints_custom(self, frame_img_root="Frames", output_mp4=None, score_thr=0.3, target_parts="torso_legs"): # Generate mp4
        # Creating mp4
        if output_mp4 == None:
            output_mp4 = f"foot_points.mp4"
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_wtr  = cv2.VideoWriter(output_mp4, fourcc=fourcc, fps=30.0, frameSize=(1280, 960))
        if not video_wtr.isOpened():
            print(f"Cannot open video writer.")
            return

        for i, frame in enumerate(self.keypoints):
            if i >= 300: # only 10-sec, just for debug
                break
            # Read frame image file
            frame_img_path = os.path.join(frame_img_root, f"{int(frame):06d}.jpg")
            frame_img = cv2.imread(frame_img_path)
            detections = self.keypoints[frame]
            target_indices = self.kp_indice_torso if target_parts == "torso" else self.kp_indice_torso_legs
            for det in detections:
                keypoints = det["keypoints"]
                confidences = [k for i2, k in enumerate(keypoints) if i2 in target_indices and k[2] >= score_thr]
                if len(confidences) < (len(target_indices)):
                    # Show bbox in red if doesn't meet the criteria
                    bbox = det["bbox"]
                    cv2.rectangle(frame_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness=2)

                left_ankle, right_ankle = keypoints[self.kp_indice_foot[0]], keypoints[self.kp_indice_foot[1]]
                if float(left_ankle[2]) >= score_thr:
                    color = (0, 255, 0)
                else:
                    #print(f"Low confidence on KP[15]: {float(fp1[2])}")
                    color = (0, 0, 255)
                cv2.circle(frame_img, (int(left_ankle[0]), int(left_ankle[1])), 5, color, 3)
                
                if float(right_ankle[2]) >= score_thr:
                    color = (0, 255, 0)
                else:
                    #print(f"Low confidence on KP[16]: {float(fp2[2])}")
                    color = (0, 0, 255)
                cv2.circle(frame_img, (int(right_ankle[0]), int(right_ankle[1])), 5, color, 3)
            frame_img = cv2.resize(frame_img, (1280, 960))
            video_wtr.write(frame_img)
        video_wtr.release()
        print(f"Saved video file: {output_mp4}\n")

    def draw_keypoints(self, frame_img, frame_id, out_file="kp_img.jpg"):
        def draw_line(img, s1, s2, bbox):
            color = (255, 0, 0) # Blue
            cv2.line(img, (int(s1[0]), int(s1[1])),
                (int(s2[0]), int(s2[1])), color, thickness=2)

        def draw_dot(img, src, bbox):
            color = (0, 255, 0) # Green
            cv2.circle(img, (int(src[0]), int(src[1])), 5, color, 2)

        frame_id = str(frame_id)
        if not frame_id in self.keypoints:
            print(f"There's no record asssiate with frame {frame_id} in the keypoint data.")
            return

        # Read frame image file
        if os.path.isfile(frame_img):
            img = cv2.imread(frame_img)
        else:
            print(f"There's no such image file {frame_img}.")
            return
        
        detections = self.keypoints[str(frame_id)]
        for det in detections:
            keypoints = det["keypoints"]
            bbox = det["bbox"]

            # draw lines
            # 0 to 1, 2
            draw_line(img, keypoints[0], keypoints[1], bbox)
            draw_line(img, keypoints[0], keypoints[2], bbox)
            # 1 to 2, 3
            draw_line(img, keypoints[1], keypoints[2], bbox)
            draw_line(img, keypoints[1], keypoints[3], bbox)
            # 2 to 4
            draw_line(img, keypoints[2], keypoints[4], bbox)
            # 3 to 5
            draw_line(img, keypoints[3], keypoints[5], bbox)
            # 4 to 6
            draw_line(img, keypoints[4], keypoints[6], bbox)
            # 5 to 6, 7, 11
            draw_line(img, keypoints[5], keypoints[6], bbox)
            draw_line(img, keypoints[5], keypoints[7], bbox)
            draw_line(img, keypoints[5], keypoints[11], bbox)
            # 6 to 8, 12
            draw_line(img, keypoints[6], keypoints[8], bbox)
            draw_line(img, keypoints[6], keypoints[12], bbox)
            # 7 to 9
            draw_line(img, keypoints[7], keypoints[9], bbox)
            # 8 to 10
            draw_line(img, keypoints[8], keypoints[10], bbox)
            # 11 to 12, 13
            draw_line(img, keypoints[11], keypoints[12], bbox)
            draw_line(img, keypoints[11], keypoints[13], bbox)
            # 12 to 14
            draw_line(img, keypoints[12], keypoints[14], bbox)
            # 13 to 15
            draw_line(img, keypoints[13], keypoints[15], bbox)
            # 14 to 16
            draw_line(img, keypoints[14], keypoints[16], bbox)

            # Draw dots
            for kp in keypoints:
                draw_dot(img, (int(kp[0]), int(kp[1])), bbox)

        cv2.imwrite(out_file, img)
        print(f"Saved keypoint file: {out_file}")

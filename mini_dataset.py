
import cv2
import glob
import os
from typing import List
from dotmap import DotMap
import torch
import numpy as np
from utils import cameraSync_framerate, get_immediate_subdirectories, read_image_PIL
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple

DatasetRange = namedtuple('DatasetRange', ['start_index', 'stop_index', 'base_camera'])

class MiniDataset:
    '''
        Mini Dataset for experimenting with a smaller dataset.
        The dataset takes as input the path to the dataset.
        It assumes a folder structure for the dataset path.
        dataset_path
            |
            |-> images
                |-> 00
                .
                .
                .
                |-> 19
            |-> intrinsics
            |-> cam_poses
            |-> synchronization

    '''
    def __init__(self, path: str, dataset_range: DatasetRange, ignored_cameras: List = []):
        self.dataset_path=path
        self.ignored_cameras = ignored_cameras
        self.start_index = dataset_range.start_index
        self.stop_index = dataset_range.stop_index
        self.base_cameara = dataset_range.base_camera
        self.read_images()
        self._compute_cameras_present()
        self.K_dict = self.read_calibration(self.cameras_present, self.image_ids)
        self.pose_dict = self.read_camera_poses(self.cameras_present, self.image_ids)
        self.read_camerasync_information()
        self._sanity_check()

    def read_camerasync_information(self):

        initial_synchronization_file = os.path.join(self.dataset_path, 'InitSync.txt')

        if os.path.exists(initial_synchronization_file):
            self.diff, self.fps = cameraSync_framerate(initial_synchronization_file)
        else:
            raise IOError("The InitSync.txt file does not exist.")

    def read_images(self) -> None:
        self.images_folder = os.path.join(self.dataset_path,'images')
        camera_ids = get_immediate_subdirectories(self.images_folder)
        self.camera_sequence_paths = []
        self.image_paths, self.image_ids = DotMap(), DotMap()
        for cam_id in camera_ids:
            image_ids_for_camera = []
            if cam_id not in self.ignored_cameras:
                seq_path = os.path.join(self.images_folder, cam_id)
                self.camera_sequence_paths.append(seq_path)
                image_files = sorted(glob.glob(os.path.join(seq_path,'*.jpg')))
                image_ids_for_camera = [os.path.split(img_file)[1].split('.')[0] for img_file in image_files]
                self.image_paths[cam_id] = image_files
                self.image_ids[cam_id] = image_ids_for_camera
        

    def read_average_calibration(self, cameras_present: List[str]) -> None:
        avg_intrinsics_file = os.path.join(self.dataset_path,'AvgDevicesIntrinsics.txt')
        with open(avg_intrinsics_file) as f:
            lines = f.readlines()
        index, K = [], []
        K_dict = DotMap()
        for line in lines:
            tokens = line.split(' ')
            index_element = int(tokens[0].split('.')[0])
            Intrinsics = np.array(tokens[5:10]).astype(float)
            K_computed = np.asarray([[Intrinsics[0],0.0,Intrinsics[3]],[0.0,Intrinsics[1],Intrinsics[4]],[0.0,0.0,1.0]], dtype = float)
            K_dict[index_element] = K_computed
        return K_dict
    
    def read_calibration(self, cameras_present: List[int], image_ids: DotMap) -> None:
        intrinsics_path = os.path.join(self.dataset_path,'intrinsics')
        K_dict = DotMap()
        # print(f"Cameras Present")
        # print(cameras_present)
        
        for cam in cameras_present:
            # print(f"cam: {cam}")
            intrinsics_file = os.path.join(intrinsics_path,'vIntrinsic_' + str(cam).zfill(4) + '.txt')
            # print(f"{intrinsics_file}")
            if os.path.exists(intrinsics_file): 
                K, K_index = self.read_intrinsics(intrinsics_file)
                K_index = [int(elem) for elem in K_index]
                K_dict[cam] = DotMap()
                for img_index in  image_ids[cam]:
                    try:
                        K_found_index = K_index.index(int(img_index))
                        K_dict[cam][int(img_index)] = K[K_found_index]
                    except ValueError as e:
                        # print(f"{e} | ")
                        continue
                print(f"cam_id: {cam} num intrinsics found: {len(K_dict[cam])}")
        return K_dict

    def read_intrinsics(self, filename: str) -> tuple:
        '''
        
        '''
        with open(filename) as f:
            lines = f.readlines()
        index, K = [], []
        for line in lines:
            index.append(line.split(' ')[0])
            Int = np.array(line.split(' ')[5:10]).astype(float)
            K_computed = np.asarray([[Int[0],0.0,Int[3]],[0.0,Int[1],Int[4]],[0.0,0.0,1.0]], dtype = float)
            K.append(K_computed)
        return K, index

    def _compute_cameras_present(self) -> None:
        '''
            Get cameras present in the mini-dataset.
            This is a temporary function to deal with 
        '''
        self.cameras_present = list(self.image_paths.keys())
        
    def read_camera_poses(self, cameras_present: List[int], image_ids: DotMap) -> None:
        '''
            Read camera poses.
            
            Input: 
                cameras_present [List[int]]: List of cameras present.
                image_ids [DotMap]: Image paths for which to read cameras

            Output:
                cam_poses [DotMap]: Poses of cameras corresponding to the image paths
        '''
        cam_poses = DotMap()
        camera_poses_dir = os.path.join(self.dataset_path,'cam_poses')
        for cam in cameras_present:
            camera_poses_groundtruth_file = os.path.join(camera_poses_dir,'vCamPose_'+str(cam).zfill(4) + '.txt')
            RT, _, RT_index = self._read_extrinsics(camera_poses_groundtruth_file)
            RT_index = [int(elem) for elem in RT_index]
            seq_poses = DotMap()
            image_inds_for_cam = image_ids[cam]
            for img_ind in image_inds_for_cam:
                try:
                    cam_found_index = RT_index.index(int(img_ind))
                    T = RT[cam_found_index]
                    seq_poses[int(img_ind)] = T
                except ValueError as e:
                    continue
                    # print(f"SE(3) | Rt for {img_ind} does not exist. {e}")
            print(f"cam_id: {cam} num_poses_found: {len(seq_poses)}")
            cam_poses[cam] = seq_poses
        return cam_poses        

    def _sanity_check(self):
        '''
            The dataset has afair amount of missing data, ex. cam_poses for certain poses, images and intrinsic matrices.
            If after reading the dicts for poses, images and intrinsics do not exist for certain cameras, we mark them as ignored.
        '''
        print(f"Performing sanity check for the dataset")
        print(f"Initial ignored cameras: {self.ignored_cameras}.")

        dataset_range = self.stop_index - self.start_index
        minimum_images_in_dataset = int(np.round(0.5 * dataset_range))
        for cam_id in self.pose_dict.keys():
            if (len(self.pose_dict[cam_id]) < minimum_images_in_dataset):
                self.ignored_cameras.append(cam_id)
        
        for cam_id in self.K_dict.keys():
            if (len(self.K_dict[cam_id]) < minimum_images_in_dataset):
                self.ignored_cameras.append(cam_id)

        print(f"Updated ignored cameras: {self.ignored_cameras}.")

 
    def _read_extrinsics(self, filename: str, rolling_shutter_flag: bool = False) -> tuple:
        '''
            Read the intrinsics from a file.

            Input: 
                filename [str]: Filename of the intrinsics to read

            Output:
                Rt [List[np.array]]: List of SE(3) transformations to read from the file
        '''
        with open(filename) as f:
            lines = f.readlines()
        Rt, rolling_shutter_parameters, index = [], [], []      #   What is Rs? 
        for line in lines:
            index.append(line.split(' ')[0])
            T = np.array(line.split(' ')[4:7]).astype(float).reshape(3,1)
            R, jacobian = cv2.Rodrigues(np.array(line.split(' ')[1:4]).astype(float))
            #R = np.transpose(R)
            #T = -(np.dot(np.transpose(R),T))
            if (rolling_shutter_flag == True):
                rolling_shutter_parameters.append(np.array(line.split(' ')[7:13]).astype(np.float).reshape(6,1))
            Rt.append(np.concatenate((R, T), axis=1))
        return Rt, rolling_shutter_parameters, index

if __name__ == "__main__":

    DATASET_PATH = "/home/sid/Desktop/carfusion/MorewoodSmall3"
    ignored_cameras = []
    dataset_range = DatasetRange(11950, 12000, 1)
    dataset = MiniDataset(DATASET_PATH, dataset_range, ignored_cameras)
    dataset.sanity_check()
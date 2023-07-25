import cv2
import glob
import os
from typing import List
from dotmap import DotMap
import torch
import numpy as np
from utils import cameraSync_framerate, get_immediate_subdirectories, read_image_PIL, list_txt_files_in_url
import shutil
from PIL import Image
import matplotlib.pyplot as plt

from mini_dataset import MiniDataset
from download_utils import download_file, download_files, list_files_in_url_with_extension,  download_images
from download_utils import list_jpg_files,  get_pose_files_from_carfusion_url,  get_intrisics_files_from_carfusion_url

class MiniDatasetCreator:
    '''
        As the datasets we have are really huge, we extract a subset of the frames 
        in the dataset and create a smaller dataset, for ease of testing.
    '''

    def __init__(self, dataset_path, new_dataset_path, start_frame: int, end_frame: int, camera_id: int, cameras_to_skip: List):

        self.dataset_path = dataset_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.camera_id = camera_id
        self.new_dataset_path = new_dataset_path
        self.missing_files = []
        self.cameras_to_skip = cameras_to_skip
        
    def read_camerasync_information(self):
        
        initial_synchronization_file = os.path.join(self.dataset_path, 'InitSync.txt')
        print(f"initial_synchronization_file: {initial_synchronization_file}")

        if os.path.exists(initial_synchronization_file):
            self.diff, self.fps = cameraSync_framerate(initial_synchronization_file)
        else:
            raise IOError("The InitSync.txt file does not exist.")
        print(f'Done')
    
    def _dataset_path_checks(self):
        if not os.path.exists(self.dataset_path):
            raise IOError("Dataset Path : {self.dataset_path} does not exist.")
        
        images_folder = os.path.join(self.dataset_path, 'images')
        if not os.path.exists(images_folder):
            raise IOError("Dataset Path : {self.dataset_path} does not exist.")
        self.images_folder = images_folder

        intrinsics_folder = os.path.join(self.dataset_path, 'intrinsics')
        if not os.path.exists(intrinsics_folder):
            raise IOError("Intrinsics Folder : {intrinsics_folder} does not exist.")
        self.intrinsics_folder = intrinsics_folder

        cam_poses_folder = os.path.join(self.dataset_path, 'cam_poses')
        if not os.path.exists(cam_poses_folder):
            raise IOError("Cam Poses Path : {cam_poses_folder} does not exist.")
        self.cam_poses_folder = cam_poses_folder

    def get_num_cameras_in_dataset(self):
        self.cameras_present = get_immediate_subdirectories(self.images_folder)
        
        if (len(self.cameras_present)==0):
            raise IOError("No camera sequences detected in {self.images_folder}. Please check if this path makes sense.")
        
        self.num_cameras = len(self.cameras_present)

    def read_images_for_current_timestep(self, timestep: int) -> DotMap:

        access_index = timestep-self.start_frame
        images_for_current_timestep = DotMap()
        if str(timestep) not in self.image_paths.keys():
            print(f"Unable to find key {timestep} in image_paths")
        image_paths_for_current_timestep = self.image_paths[str(timestep)]
        for cam_id in image_paths_for_current_timestep.keys():
            img_path = image_paths_for_current_timestep[cam_id]
    
            if (len(img_path) == 0):
                continue
            print(f"{cam_id} {img_path}")
            image = read_image_PIL(img_path)
            images_for_current_timestep[cam_id] = image
        return images_for_current_timestep

    def compute_frames_for_current_timestep(self, timestep: int, fps: np.array, diff: np.array):
        '''

        '''
        image_paths_for_current_timestep = DotMap()
        for cam in range(self.num_cameras):
            if (str(cam) in self.cameras_to_skip):
                continue
            delta, fps_multiplier = self.diff[cam], self.fps[cam] 
            frame_in_cam = int(timestep * fps_multiplier - delta)
            
            #   frame_in_cam < 0 indicates the frame is not present in the sequence and 
            #   was captured before the current camera started recording. These we skip.
            if frame_in_cam < 0:
                image_path = ''
            else:
                image_name = str(frame_in_cam).zfill(6) + ".jpg"
                image_path = os.path.join(self.images_folder,str(cam).zfill(2),image_name)
                image_path_backup = os.path.join(self.images_folder,str(cam).zfill(2),image_name)
                
                #   In case the image does not exist for whatever reason.
                if not os.path.exists(image_path):  
                    image_path = ''
            
            if image_path == '':
                # print(f"{image_path_backup} does not exist.")
                self.missing_files.append(image_path_backup)
            image_paths_for_current_timestep[str(cam).zfill(2)] = image_path
        
        return image_paths_for_current_timestep

    def get_image_arrangement(self, num_images: int):

        if (num_images == 17 or  num_images == 18 or num_images ==19 or num_images == 20):
            num_rows, num_cols = 4, 5
        elif (num_images ==15 or num_images == 16 or num_images ==13 or num_images == 14):
            num_rows, num_cols = 4, 4
        return num_rows, num_cols

    def visualize_images_from_same_timestep(self, images: dict):
        # images = self.get_images_for_current_timestep(self.current_timestep)
        num_rows, num_cols = self.get_image_arrangement(len(images.keys()))
        fig = plt.figure()
        for cam_id in images.keys():
            # keypoints_in_cam = np.round(keypoints[cam_id]).astype(int)
            try:
                fig.add_subplot(num_rows, num_cols, int(cam_id))
                plt.imshow(images[cam_id])
                # plt.scatter(keypoints_in_cam[:,0].tolist(), keypoints_in_cam[:,1].tolist(), s=3)
                plt.axis('off')
                plt.title(cam_id)
            except:
                print(f"Error in plotting: {images[cam_id]}.")
        
        plt.subplots_adjust(left=0.01, bottom=0.01,right=0.99,top=0.95,wspace=0.1,hspace=0.1)
        plt.show()

    def compute_frames_to_copy(self):
        '''
            
        '''
        image_paths = DotMap()
        for frame in range(self.start_frame, self.end_frame):
            image_paths_for_current_timestep = self.compute_frames_for_current_timestep(frame, self.fps, self.diff)
            # print(f"Frame: {frame}")
            # print(image_paths_for_current_timestep)
            # input()
            image_paths[str(frame).zfill(2)] = image_paths_for_current_timestep
        return image_paths


    def create_folder_structure(self):
        
        print(f"creating folder structure")
        if os.path.exists(self.new_dataset_path):
            print(f"Dataset path already exists")
            # raise Exception("Dataset path already exists please change the location.")
            # print("YAY")
        else:
            os.mkdir(self.new_dataset_path)

        new_dataset_images_folder = os.path.join(self.new_dataset_path,'images')
        new_dataset_intrinsics_folder = os.path.join(self.new_dataset_path,'intrinsics')
        new_dataset_cam_poses_folder = os.path.join(self.new_dataset_path,'cam_poses')

        # print(f"images folder: {new_dataset_images_folder}")
        # print(f"intrinsics folder: {new_dataset_intrinsics_folder}")
        # print(f"cam_poses folder: {new_dataset_cam_poses_folder}")

        if os.path.exists(new_dataset_images_folder):
            print(f"images folder: {new_dataset_images_folder} exists")
            # raise Exception(f"new_dataset_images_folder {new_dataset_images_folder} path already exists please change the location.")
        else:
            os.mkdir(new_dataset_images_folder)

        if os.path.exists(new_dataset_intrinsics_folder):
            print(f"intrinsics folder: {new_dataset_intrinsics_folder} exists")
            # raise Exception(f"new_dataset_images_folder {new_dataset_intrinsics_folder} path already exists please change the location.")
        else:
            os.mkdir(new_dataset_intrinsics_folder)

        if os.path.exists(new_dataset_cam_poses_folder):
            print(f"cam_poses folder: {new_dataset_cam_poses_folder} exists")
            # raise Exception(f"new_dataset_images_folder {new_dataset_cam_poses_folder} path already exists please change the location.")
        else:
            os.mkdir(new_dataset_cam_poses_folder)


        # #   Create folders for camera sequences
        # for cam in self.cameras_present:
        #     camera_sequence_folder = os.path.join(new_dataset_images_folder, str(cam).zfill(2))
        #     if not os.path.exists(camera_sequence_folder):
        #         os.mkdir(camera_sequence_folder)



    def download_mini_dataset(self):

        self.create_folder_structure()

    def create_mini_dataset(self):

        #   Checking if the dataset paths exist or not.
        self._dataset_path_checks()
        
        #   Reading the camera synchronization information
        self.read_camerasync_information()

        #   Get the cameras present in the dataset.
        self.get_num_cameras_in_dataset()
        print(f"self.num_cameras: {self.num_cameras}")

        #   Get the frames that are going to be copied to the 
        self.image_paths = self.compute_frames_to_copy()

        self.camera_sequence_paths = [os.path.join(self.images_folder,cam) for cam in self.cameras_present]
        

        #   First create the folder structure for the dataset
        if os.path.exists(self.new_dataset_path):
            raise Exception("Dataset path already exists please change the location.")
            # print("YAY")
        else:
            os.mkdir(self.new_dataset_path)

        # self.download_images()

        new_dataset_images_folder = os.path.join(self.new_dataset_path,'images')
        new_dataset_intrinsics_folder = os.path.join(self.new_dataset_path,'intrinsics')
        new_dataset_cam_poses_folder = os.path.join(self.new_dataset_path,'cam_poses')

        if os.path.exists(new_dataset_images_folder):
            raise Exception(f"new_dataset_images_folder {new_dataset_images_folder} path already exists please change the location.")
        else:
            os.mkdir(new_dataset_images_folder)

        if os.path.exists(new_dataset_intrinsics_folder):
            raise Exception(f"new_dataset_images_folder {new_dataset_intrinsics_folder} path already exists please change the location.")
        else:
            os.mkdir(new_dataset_intrinsics_folder)

        if os.path.exists(new_dataset_cam_poses_folder):
            raise Exception(f"new_dataset_images_folder {new_dataset_cam_poses_folder} path already exists please change the location.")
        else:
            os.mkdir(new_dataset_cam_poses_folder)

        #   Create folders for camera sequences
        for cam in self.cameras_present:
            camera_sequence_folder = os.path.join(new_dataset_images_folder, str(cam).zfill(2))
            if not os.path.exists(camera_sequence_folder):
                os.mkdir(camera_sequence_folder)



        # images_for_current_timestep = self.read_images_for_current_timestep(self.start_frame)


        
        # print(f"Hi")
        # image_paths_for_current_timestep = self.compute_frames_for_current_timestep(self.start_frame, self.fps, self.diff)
        
        # print(image_paths_for_current_timestep)

        # self.visualize_images_from_same_timestep(images_for_current_timestep)

        
        # new_dataset_images_folder = os.path.join(self.new_dataset_path,'images')
        # new_dataset_intrinsics_folder = os.path.join(self.new_dataset_path,'intrinsics')
        # new_dataset_cam_poses_folder = os.path.join(self.new_dataset_path,'cam_poses')

        # if os.path.exists(new_dataset_images_folder):
        #     raise Exception(f"new_dataset_images_folder {new_dataset_images_folder} path already exists please change the location.")
        # else:
        #     os.mkdir(new_dataset_images_folder)

        # #   Create folders for camera sequences
        # for cam in self.cameras_present:
        #     camera_sequence_folder = os.path.join(new_dataset_images_folder, str(cam))
        #     if not os.path.exists(camera_sequence_folder):
        #         os.mkdir(camera_sequence_folder)

        # #   Copy intrinsics and camera poses
        # shutil.copytree(self.intrinsics_folder, new_dataset_intrinsics_folder)
        # shutil.copytree(self.cam_poses_folder, new_dataset_cam_poses_folder)

        # for frame in self.image_paths.keys():
        #     image_paths_for_current_timestep = self.image_paths[frame]
        #     for cam_id in image_paths_for_current_timestep.keys():
        #         img_path = image_paths_for_current_timestep[cam_id]
        #         if (img_path == ''):
        #             continue
        #         image_name = os.path.split(img_path)[1]
        #         new_image_path = os.path.join(new_dataset_images_folder, cam_id, image_name)
        #         shutil.copy(img_path, new_image_path)

        # self._download_missing_files()
        # self._check_damaged_images()

if __name__ == "__main__":
    
    DATASET_PATH = "/home/sid/Desktop/carfusion/Morewood"
    small_dataset_save_path = "/home/sid/Desktop/carfusion/MorewoodSmall2"

    start_frame, end_frame, camera_id = 25000, 25200, 1

    # cameras_to_skip = ['13']
    cameras_to_skip = []
    dataset_creator = MiniDatasetCreator(DATASET_PATH, small_dataset_save_path, start_frame, end_frame, camera_id, cameras_to_skip)

    dataset_creator.download_mini_dataset()

    # dataset_creator.create_mini_dataset()
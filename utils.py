from typing import List
import numpy as np
import torch
import cv2
import os
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from bs4 import BeautifulSoup


def read_image_superpoint(impath, img_size=[480, 640]):
    """ Read image as grayscale and resize to img_size.
    Inputs
        impath: Path to input image.
        img_size: (W, H) tuple specifying resize size.
    Returns
        grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
        raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)   #   
    grayim = (grayim.astype('float32') / 255.)
    return grayim


def convert_to_greyscale(image: np.array) -> np.array:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    assert(image.ndim == 2)
    return image

def resize_image_scaling_factor(image: np.array, scaling_factor_height, scaling_factor_width) -> np.array:
    # assert(image.ndim == 2)
    interp = cv2.INTER_AREA
    gray_image = cv2.resize(image, (int(image.shape[1]*scaling_factor_width),int(image.shape[0]*scaling_factor_height)), interpolation=interp)   #   
    return gray_image


def compute_scaling_factor(original_size, resized_size = [480, 640]):
    """
    Compute the scaling factor between an original image size and a resized image size.

    Args:
        original_size (tuple or list): Original image size in the format (height, width).
        resized_size (tuple or list): Resized image size in the format (height, width).

    Returns:
        float: Scaling factor.
    """
    original_height, original_width = original_size
    resized_height, resized_width = resized_size

    scaling_factor_height, scaling_factor_width = resized_height / original_height, resized_width / original_width

    return scaling_factor_height, scaling_factor_width

def resize_image(image: np.array, img_size = [480, 640]) -> np.array:
    # assert(image.ndim == 2)
    interp = cv2.INTER_AREA
    gray_image = cv2.resize(image, (int(img_size[1]),int(img_size[0])), interpolation=interp)   #   
    return gray_image

def read_image_PIL(path: str):

    if not os.path.exists(path):
        raise IOError(f"Unable to read image. {path} does not exist.")
    try:
        image = np.array(Image.open(path))
    except:
        print(f"Unable to read image: {path}")

    return image

def to_torch(img: np.array):
    return torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0) / 255.0

def create_lexical_ordering(list: List[str]) -> List[str]:        
    return [(list[i] + "_" + list[j]) for i in range(len(list)) for j in range(i,len(list)) ]

def read_coloured_image(image_path: str) -> np.array:
    return cv2.imread(image_path)

def bgr_to_rgb(img: np.array):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img: np.array):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def rgb_to_grey(img: np.array):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_numpy(arr: torch.Tensor) -> np.array:
    assert(len(arr.shape) == 4 or len(arr.shape) == 3)
    if(len(arr.shape) == 3):
        arr = arr.unsqueeze(0)
    return [img.permute(1,2,0).detach().cpu().numpy() for img in arr]


def get_immediate_subdirectories(directory_path: str):
    return sorted([name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))])


# def check_if_file_exists(filename: str):
#     return os.path.exists(filename)

# def to_torch(arr: np.array) -> torch.Tensor:
#     assert(len(arr.shape) == 4 or len(arr.shape) == 3)
#     if(len(arr.shape) == 3):
#         arr = np.expand_dims(arr, axis=0)
#     arr = torch.from_numpy(arr)
#     return [img.permute(0,1,2) for img in arr]

#######################################################################
#
#   NOT SURE WHAT THIS IS FOR
#
#######################################################################

def sample_keypoints(keypoints_0: np.array, keypoints_1: np.array, ratio_to_keep: float = 0.3):

    assert(keypoints_0.shape[0] == keypoints_1.shape[0]) 
    assert(keypoints_0.shape[1] == 2)
    assert(keypoints_1.shape[1] == 2)

    num_keypoints = keypoints_0.shape[0]

    permutation = np.random.permutation(num_keypoints)
    num_keypoints_to_keep = int(np.round(ratio_to_keep* num_keypoints))
    indices_to_keep = permutation[:num_keypoints_to_keep]
    keypoints_0 = keypoints_0[indices_to_keep,:]
    keypoints_1 = keypoints_1[indices_to_keep,:]
    return keypoints_0, keypoints_1



###################################################################
#   
#   DETECTRON BASED INSTANCE TRACKING
#   
#   SEMANTIC MASKS
#   
###################################################################

PERSON_CLASS_ID = 0
CAR_CLASS_ID = 2
VAN_CLASS_ID = 7

def create_combined_binary_mask_image(masks: torch.Tensor):
    '''
        Create combined binary mask image.
        
    '''
    mask_combined=torch.ones(masks[0].shape)
    for img in masks:
        mask_combined[img]=0

    return mask_combined.numpy()

def extract_masks_by_class_id(masks, class_ids, selected_class_ids):
    '''

    '''
    selected_instances = [i for i, class_id in enumerate(class_ids) if (class_id in selected_class_ids)]

    # masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
    selected_masks = masks[selected_instances,:,:]
    return selected_masks

#############################################################################
#
#   DATASET UTILS 
#
#############################################################################

def cameraSync_framerate(filepath: str):
    '''
        The different cameras start taking sequences at different time instances. To find the frames taken
        from the different cameras for a particular time instance, we need to find the correct synchronization
        between the frames. This is done by reading the `InitSync.txt' file per dataset. The file consists of N
        lines where N is the number of cameras. Each line has 3 numbers in the following format:
        camera_id, frame_rate_multiplier, start frame rates difference. 
        If a line is 1 0.2 0.

        Input:
            filename: [str] File name for the InitSync.txt file.
            return diff, fps.
    '''

    with open(filepath) as f:
        lines = f.readlines()
    index, diff, fps = [], [], []

    for line in lines:
        index.append(line.split(' ')[0])
        diff.append(line.split(' ')[2].split('\n')[0])
        fps.append(line.split(' ')[1].split(' ')[0])
    
    diff = np.array(diff).astype(int)
    fps = np.array(fps).astype(float)
    return diff, fps

def download_file(url, file_name):
    '''
      (  Used for dow innloading images
    '''
    response = requests.get(url)
    with open(file_name, "wb") as f:
        f.write(response.content)

def download_files(urls, image_paths):


    for (url,file_name) in zip(urls,image_paths):
        response = requests.get(url)
        with open(file_name, "wb") as f:
            f.write(response.content)



# def find_image_path_url(image_path: str):
#     assert(len(image_path)>0)
#     dir_name, image_name = os.path.split(image_path)
    
def list_txt_files_in_url(url,ext = 'txt'):
    page = requests.get(url).text
    # print(page)
    soup = BeautifulSoup(page, 'html.parser')
    return sorted([url +  node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)])

def download_images(urls_to_download, image_paths):
    '''
        Images to be downloaded.
    '''
    with ThreadPoolExecutor(max_workers=6) as exe:

        # dispatch all download tasks to worker threads
        futures = [exe.submit(download_file, url, out_file) for (url,out_file) in zip(urls_to_download, image_paths)]

        # report results as they become available
        for future in as_completed(futures):
            # retrieve result
            link, outpath = future.result()
            # check for a link that was skipped
            if outpath is None:
                print(f'>skipped {link}')
            else:
                print(f'Downloaded {link} to {outpath}')
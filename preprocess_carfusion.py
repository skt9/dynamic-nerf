import os
import shutil
import cv2
import numpy as np
import torch
import configargparse
from typing import List, OrderedDict, DefaultDict, Tuple
from glob import glob
from PIL import Image
import h5py
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
# import torchvision.transforms.functional as F
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cpu"

def tuple_type(strings: str):
    strings = strings.replace("(", "").replace(")", "").split(",")
    return tuple(strings)

def get_immediate_subdirectories(a_dir: str):
    return sorted([name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))])

def parse_args():
    parser = configargparse.ArgParser(description = "Carfusion Preprocessing Configuration.")
    parser.add('--dataset_dir', 
               type=str, required=True, help="Directory path for the dataset.")
    parser.add('--cams_to_ignore', type=tuple_type, 
               required=False, help="Cameras to ignore procesing.")
    parser.add('--img_size', type=tuple_type, 
               required=False, help="Size of the image to resize to for processing by dynibar.")
    parser.add('--midas_model_type', type=str, required=False, help='Midas Model to use for disparity estimation.')
    return parser.parse_args()

def get_midas_model(model_type: str) -> torch.nn.Module:
    """Return the midas model to use for disparity estimation
        Args:
            model_type, str: It can be one of three types "DPT_Large", "DPT_Hybrid", "MiDaS_small".

        Return:
            model, torch.nn.Module: Module of the torch Tensor. 
    """
    if (model_type == "DPT_Large"):
        return torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    elif (model_type == "DPT_Hybrid"):
        return torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    elif (model_type == "MiDaS_small"):
        return torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

def get_image_names(images_dir: str, ignore_cameras: tuple) -> DefaultDict:
    """Get the names of the images in each subfolder as a dict.
    
    """

    image_folders = get_immediate_subdirectories(images_dir)
    image_ids=DefaultDict()
    for img_folder in image_folders:
        if img_folder in ignore_cameras:
            continue
        img_folder_fullpath=os.path.join(images_dir, img_folder)
        image_names=sorted(glob(os.path.join(img_folder_fullpath,"*.jpg")))
        image_ids[img_folder]=[os.path.split(img_name)[1].split('.')[0] for img_name in image_names]
    return image_ids

def read_image_cv2(filename: str, transform, device: str) -> torch.Tensor:
    """Read image convert to torch tensor
    
    """
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_size=(img.shape[0],img.shape[1])
    input_batch = transform(img).to(device)
    return input_batch,img_size

def load_midas_transforms(model_type: str):
    """Load MiDaS transforms
    
    """
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return transform


def compute_and_save_disparity(images_dir: str, 
                               image_ids: DefaultDict, 
                               model_type: str, 
                               disparity_dir: str,
                               img_size: Tuple[int,int]) -> None:
    """Compute and save disparity for the models.
    
        Args:
            images_dir, str: Directory of the images
            image_ids, str: Image ids in the images dir
            model_type, str: 
            disparity_dir, str:

    """
    model=get_midas_model(model_type).to(device)
    transform=load_midas_transforms(model_type)
    model.eval()
    for cam_seq in tqdm(image_ids.keys()):
        image_seq_ids=image_ids[cam_seq]
        print(f"Processing sequence: {cam_seq}")
        for img_id in image_seq_ids:
            image_name=os.path.join(images_dir,cam_seq,img_id+".jpg")
            img,orig_size=read_image_cv2(image_name,transform, device)
            # print(f"orig_size: {orig_size}")
            with torch.no_grad():
                # print(f"img.shape: {img.shape}")
                prediction = model(img)
                # print(f"prediction.shape: {prediction.shape}")
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_size,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                # print(f"prediction.shape: {prediction.shape}")
                # input()
            output = prediction.cpu().numpy()
            disparity_file_path=os.path.join(disparity_dir,cam_seq,img_id+".h5")
            with h5py.File(disparity_file_path,"w") as fp:
                fp["data"]=output

def create_folder_structure(directory_path: str, 
                            subfolders: List[str], 
                            cameras_to_ignore: List[str]) -> None:
    """Create the camera folder structure.
       
        Args:
            directory_path, str: Directory within which to create the folder structure
            subfolders, List[str]: List of subfolders to create within the `directory_path`
            cameras_to_ignore, List[str]: List of cameras to ignore
    """
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    
    for folder in subfolders:
        subdirectory_path=os.path.join(directory_path, folder)
        if folder in cameras_to_ignore:
            continue
        if not os.path.exists(subdirectory_path):
            os.mkdir(subdirectory_path)

def read_image_PIL(img1_filename: str):
    img=np.array(Image.open(img1_filename))
    return torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()

def preprocess_for_optical_flow(img1_filename, img2_filename, img_size):

    img1_batch=read_image_PIL(img1_filename)
    img2_batch=read_image_PIL(img2_filename)
    orig_shape = img1_batch.shape[2:]
    img1_batch = F.interpolate(img1_batch, size=img_size, mode='bilinear', align_corners=True)
    img2_batch = F.interpolate(img2_batch, size=img_size, mode='bilinear', align_corners=True)
    return img1_batch, img2_batch, orig_shape

def to_numpy(img: torch.Tensor) ->np.ndarray:
    return img.squeeze(0).permute(1,2,0).detach().cpu().numpy()

def warp_image(image: torch.Tensor, flow: torch.Tensor, threshold: float=0.999):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = image.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)                            #   
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)                            #   
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)                                 #   
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)                                 #   
    grid = torch.cat((xx, yy), 1).float()                                       #   

    if image.is_cuda:
        grid = grid.cuda()                                                      #   
    vgrid = grid + flow                                                         #   Flow warping happens here
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0   #   Scale to between -1 and 1
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0   #   

    vgrid = vgrid.permute(0, 2, 3, 1)               #   Scale to 
    output = F.grid_sample(image, vgrid)            #   Compute the output on the extracted flow points
    mask = torch.ones(1,1,H,W).to(image.device)     #   Compute the mask of the image
    mask = F.grid_sample(mask, vgrid)               #   

    mask[mask < threshold] = 0
    mask[mask > 0] = 1

    return output, mask

    
def compute_optical_flow(images_dir: str,
                         image_ids: DefaultDict,
                         flow_dir: str,
                         spacing: int=1,
                         compute_forward: bool=True,
                         compute_backward: bool=True):
    """ Compute the optical flow for the images in the directory.

    
    """
    
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)

    transforms = Raft_Large_Weights.DEFAULT.transforms()
    for cam_seq in tqdm(image_ids.keys()):
        image_seq_ids=image_ids[cam_seq]
        first_frames = image_seq_ids[:-spacing]
        second_frames = image_seq_ids[spacing:]
        if (compute_forward):
            for (fr1,fr2) in zip(first_frames,second_frames):
                flow_fwd_file=os.path.join(flow_dir, cam_seq, fr1 + "_fwd.npz")
                if os.path.exists(flow_fwd_file):
                    continue
                img1_path=os.path.join(images_dir,cam_seq,fr1+".jpg")
                img2_path=os.path.join(images_dir,cam_seq,fr2+".jpg")
                img1, img2, orig_size=preprocess_for_optical_flow(img1_path, img2_path, img_size)
                img1_batch, img2_batch=transforms(img1, img2)
                forward_flows=model(img1_batch.to(device), img2_batch.to(device))
                predicted_flow=forward_flows[-1].cpu().detach()
                warped_image, mask=warp_image(img1,predicted_flow,threshold=0.9)
                flow=predicted_flow.squeeze(0).permute(1,2,0).numpy()
                mask=mask.squeeze().squeeze().cpu().numpy()
                img1=to_numpy(img1)
                np.savez_compressed(flow_fwd_file,flow=flow,mask=mask)

        if (compute_backward):
            for (fr1,fr2) in zip(first_frames,second_frames):
                flow_bwd_file=os.path.join(flow_dir, cam_seq, fr2 + "_bwd.npz")
                if os.path.exists(flow_bwd_file):
                    continue
                img1_path=os.path.join(images_dir,cam_seq,fr1+".jpg")
                img2_path=os.path.join(images_dir,cam_seq,fr2+".jpg")
                img1, img2, orig_size=preprocess_for_optical_flow(img1_path, img2_path, img_size)
                img1_batch, img2_batch=transforms(img1, img2)
                forward_flows=model(img2_batch.to(device), img1_batch.to(device))
                predicted_flow=forward_flows[-1].cpu().detach()
                warped_image, mask=warp_image(img2,predicted_flow,threshold=0.9)
                flow=predicted_flow.squeeze(0).permute(1,2,0).numpy()
                mask=mask.squeeze().squeeze().cpu().numpy()
                img2=to_numpy(img2)
                np.savez_compressed(flow_bwd_file,flow=flow,mask=mask)

def bgr2rgb(img:np.ndarray)->np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2bgr(img:np.ndarray)->np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def resize_and_save_images(images_dir: str,
                           image_ids: DefaultDict,
                           resize_size: tuple):
    """Resize and save images 
    
    """
    dataset_dir,images_folder =os.path.split(images_dir)
    img_folder_path=f"images_{resize_size[0]}x{resize_size[1]}"
    resize_image_folder=os.path.join(dataset_dir,img_folder_path)

    if not os.path.exists(resize_image_folder):
        os.mkdir(resize_image_folder)

    for cam_seq in image_ids.keys():
        cam_seq_dir=os.path.join(resize_image_folder,cam_seq)
        if not os.path.exists(cam_seq_dir):
            os.mkdir(cam_seq_dir)

    transform=T.Resize(resize_size,antialias=True)

    for cam_seq in image_ids.keys():
        image_seq_ids=image_ids[cam_seq]
        for img_id in image_seq_ids:
            img_path=os.path.join(images_dir,cam_seq,img_id+".jpg")
            image=read_image_PIL(img_path)
            resized_img=transform(image)        
            resize_image_np=to_numpy(resized_img)
            resized_image_path=os.path.join(resize_image_folder,cam_seq,img_id+".jpg")
            resize_image_np=bgr2rgb(resize_image_np)
            cv2.imwrite(resized_image_path, resize_image_np)
            

def write_pose_bounds_file(poses_dict):
    pass


if __name__ == "__main__":
    
    args=parse_args()
    dataset_dir = args.dataset_dir
    image_dir = os.path.join(dataset_dir,'images')
    disparity_dir = os.path.join(dataset_dir,'disp')
    
    subfolders = get_immediate_subdirectories(image_dir)
    cameras_to_ignore=args.cams_to_ignore
    model_type=args.midas_model_type

    create_folder_structure(disparity_dir, subfolders, cameras_to_ignore)
    
    flow_1_dir = os.path.join(dataset_dir,'flow_i1')
    create_folder_structure(flow_1_dir, subfolders, cameras_to_ignore)

    flow_2_dir = os.path.join(dataset_dir,'flow_i2')
    create_folder_structure(flow_2_dir, subfolders, cameras_to_ignore)

    flow_3_dir = os.path.join(dataset_dir,'flow_i3')
    create_folder_structure(flow_3_dir, subfolders, cameras_to_ignore)

    image_ids=get_image_names(image_dir, cameras_to_ignore)
    print(image_ids.keys())
    if len(args.img_size)>0:
        img_size=(int(args.img_size[0]), int(args.img_size[1]))

    
    # start=time.perf_counter()
    # compute_and_save_disparity(image_dir, image_ids, model_type, disparity_dir, img_size)
    # stop=time.perf_counter()
    # print(f"Time taken for computing depth: {stop-start} seconds")

    # start=time.perf_counter()
    # spacing=1
    # compute_optical_flow(image_dir, image_ids, flow_1_dir, spacing,compute_forward=True, compute_backward=True)
    # stop=time.perf_counter()
    # print(f"Time taken for computing flow_i1: {stop-start} seconds")

    # start=time.perf_counter()
    # spacing=2
    # compute_optical_flow(image_dir, image_ids, flow_2_dir, spacing,compute_forward=True, compute_backward=True)
    # stop=time.perf_counter()
    # print(f"Time taken for computing flow_i2: {stop-start} seconds")

    # start=time.perf_counter()
    # spacing=3
    # compute_optical_flow(image_dir, image_ids, flow_3_dir, spacing,compute_forward=True, compute_backward=True)
    # stop=time.perf_counter()
    # print(f"Time taken for computing flow_i3: {stop-start} seconds")

    # start=time.perf_counter()
    # resize_and_save_images(image_dir, image_ids, (544,960))
    # stop=time.perf_counter()
    # print(f"Time taken for computing resizing: {stop-start} seconds")
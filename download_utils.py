import requests, os
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

DATASET_URL = "http://platformpgh.cs.cmu.edu/live_stream/carfusion/Morewood"

SEQUENCES  = ['Morewood']

def download_file(url, file_name):
    '''
      (  Used for dow innloading images
    '''
    response = requests.get(url)
    with open(file_name, "wb") as f:
        f.write(response.content)

def download_files(urls, image_paths):
    
    [download_file(url, file_name) for (url,file_name) in zip(urls,image_paths)]

def download_files_multithreaded(urls, image_paths, num_workers = 6):

    with ThreadPoolExecutor(max_workers=num_workers) as exe:

        # dispatch all download tasks to worker threads
        for (url, out_file) in zip(urls, image_paths):
            exe.submit(download_file, url, out_file) 

        # report results as they become available
        # for future in as_completed(futures):
        #     # retrieve result
        #     link, outpath = future.result()
        #     # check for a link that was skipped
        #     if outpath is None:
        #         print(f'>skipped {link}')
        #     else:
        #         print(f'Downloaded {link} to {outpath}')

def list_files_in_url_with_extension(url,ext = 'txt'):
    page = requests.get(url).text
    # print(page)
    soup = BeautifulSoup(page, 'html.parser')
    return sorted([os.path.join(url, node.get('href')) for node in soup.find_all('a') if node.get('href').endswith(ext)])

def download_images(urls_to_download, image_paths):

    pass

def list_jpg_files(url):
    return list_files_in_url_with_extension(url,'jpg')

def get_pose_files_from_carfusion_url(carfusion_url):
    txt_files = list_files_in_url_with_extension(carfusion_url,'txt')
    pose_files = [file for file in txt_files if 'Pose' in file]
    return pose_files

def get_intrisics_files_from_carfusion_url(carfusion_url):
    txt_files = list_files_in_url_with_extension(carfusion_url,'txt')
    pose_files = [file for file in txt_files if 'Intrinsic' in file]
    return pose_files

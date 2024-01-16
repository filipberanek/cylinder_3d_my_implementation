import numpy as np 
import numpy.lib.recfunctions as rfn
from pypcd import pypcd
import pathlib
import math
from tqdm import tqdm

INPUT_PCD_PATH= pathlib.Path("/home/fberanek/Desktop/datasets/segmentation/semantic/valeo_to_be_published/preprocessed/2clss_55rect_55kpts_5rect/pcd_adjusted")
INPUT_NPZ_PATH = pathlib.Path("/home/fberanek/Desktop/datasets/segmentation/semantic/valeo_to_be_published/preprocessed/2clss_55rect_55kpts_5rect/overfited_labels_prediction")
OUTPUT_PATH = pathlib.Path("/home/fberanek/Desktop/datasets/segmentation/semantic/valeo_to_be_published/preprocessed/2clss_55rect_55kpts_5rect/pcd_merged_with_overfit")
RECT = 25
N_PTS = 65000

for file_path in tqdm(list(INPUT_PCD_PATH.rglob("*.pcd"))):
    try:
        pc = pypcd.PointCloud.from_path(str(file_path))
        pc_data = pc.pc_data
        pc_arr = np.array(pc_data)
        npz = np.load(INPUT_NPZ_PATH/(file_path.stem + ".npz"), allow_pickle=True)["data"]
        npz = rfn.append_fields(pc_arr, "predictions", npz,usemask=False)
        new_pcd = pypcd.PointCloud.from_array(npz)
        pypcd.save_point_cloud(new_pcd,str(OUTPUT_PATH/file_path.name))
    except FileNotFoundError:
        print(f"File {file_path.stem} not found among npzs.")
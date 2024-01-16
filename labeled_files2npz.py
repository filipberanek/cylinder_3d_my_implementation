import numpy as np 
from pypcd import pypcd
import pathlib
import math
from tqdm import tqdm

INPUT_PATH = pathlib.Path("/home/fberanek/Desktop/datasets/segmentation/semantic/valeo_to_be_published/pcd")
OUTPUT_PATH = pathlib.Path("/home/fberanek/Desktop/datasets/segmentation/semantic/valeo_to_be_published/labeled_npz")
COLUMNS = ['Classification', 'intensity', 'x', 'y', 'z']
RECT = 25
N_PTS = 65000

def create_dummy_records(orig_array, number_of_records_to_add):
    record = []
    for dtype_name in orig_array.dtype.names:
        record.append(np.min(orig_array[dtype_name]))
    row_record = np.array([tuple(ValueError)], dtype=orig_array.dtype)
    rows_to_add = np.tile(row_record, number_of_records_to_add)
    return rows_to_add

def harmonize_voxels(current_voxel, number_of_pts_per_voxel):
    complete_arr = np.array([], dtype=current_voxel.dtype)
    if len(current_voxel)<number_of_pts_per_voxel:
        rows_to_add = create_dummy_records(current_voxel, len(current_voxel)-number_of_pts_per_voxel)
        complete_arr = np.append(current_voxel, rows_to_add)
    elif len(current_voxel)==number_of_pts_per_voxel:
        complete_arr = current_voxel
    elif len(current_voxel)>number_of_pts_per_voxel:
        complete_arr = np.random.choice(current_voxel, number_of_pts_per_voxel)
    return complete_arr


def create_voxels(np_array:np.array, rect_size, min_number_of_point, number_of_pts_per_voxel, file_name_stem, output_path):
    # Get number of x voxels to loop
    x_number_of_voxels = math.ceil((np.max(np_array["x"])-np.min(np_array["x"]))/rect_size)
    x_min = np.min(np_array["x"])
    for x_voxel_id in tqdm(range(x_number_of_voxels)):
        # Get number of y voxels to loop
        y_number_of_voxels = math.ceil((np.max(np_array["y"])-np.min(np_array["y"]))/rect_size)
        y_min = np.min(np_array["y"])
        for y_voxel_id in tqdm(range(y_number_of_voxels)):
            # X mask for voxel
            x_mask = np.logical_and(np_array["x"] >= x_min, np_array["x"] < (x_min + rect_size))
            # Y mask for voxel
            y_mask = np.logical_and(np_array["y"] >= y_min, np_array["y"] < (y_min + rect_size))
            # Get current voxel 
            current_voxel = np_array[np.logical_and(x_mask, y_mask)]
            if len(current_voxel) >= min_number_of_point: 
                current_voxel = harmonize_voxels(current_voxel, number_of_pts_per_voxel)
                np.savez("output_path"/(file_path.stemp + ".npz"), data = current_voxel)
            else: 
                print(f"Skipping x_voxel_id {x_voxel_id}, y_voxel_id {y_voxel_id}")
            y_min += rect_size
        x_min += rect_size

for file_path in list(INPUT_PATH.rglob("*.pcd")):
    pc = pypcd.PointCloud.from_path(str(file_path))
    pc_data = pc.pc_data
    pc_arr = np.array(pc_data)
    pc_arr = np.array(pc_arr[COLUMNS])
    voxels = create_voxels(pc_arr, RECT,1000,N_PTS, file_path.stem, OUTPUT_PATH)
    
    print()
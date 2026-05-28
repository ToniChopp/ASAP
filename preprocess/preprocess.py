import os
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from tqdm import tqdm
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation
import ipdb

# TODO: Modify these paths as needed
img_root = "./train_images/"
save_folder = "./train_preprocessed/"


def read_nii_files(directory):
    """
    Retrieve paths of all NIfTI files in the given directory.

    Args:
    directory (str): Path to the directory containing NIfTI files.

    Returns:
    list: List of paths to NIfTI files.
    """
    nii_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
    return nii_files
    

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


def process_file(file_path):
    """
    Process a single NIfTI file.

    Args:
    file_path (str): Path to the NIfTI file.

    Returns:
    None
    """
    nii_img = nib.load(file_path)
    img_data = np.asanyarray(nii_img.dataobj)
    affine_current = nii_img.affine

    if img_data is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return

    current_ornt = io_orientation(affine_current)
    ras_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(current_ornt, ras_ornt)
    img_data = nib.orientations.apply_orientation(img_data, transform)
    affine_current = nii_img.as_reoriented(transform).affine

    current_x_spacing, current_y_spacing, current_z_spacing = nib.affines.voxel_sizes(affine_current)

    # Define the target spacing values
    ### Modify: We keep the spacing to [1.5, 1.5, 3.0]
    target_x_spacing = 1.5
    target_y_spacing = 1.5
    target_z_spacing = 3.0

    current = (current_x_spacing, current_y_spacing, current_z_spacing)
    target = (target_x_spacing, target_y_spacing, target_z_spacing)
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = img_data.astype(np.float32)

    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    try:
        resized_array = resize_array(tensor, current, target)
    except Exception as e:
        print(f"Error resizing file {file_path}: {e}")
        return
    resized_array = resized_array[0][0]
    resized_array = resized_array.astype(np.int16)

    target_affine = np.diag([target_x_spacing, target_y_spacing, target_z_spacing, 1])


    os.makedirs(save_folder, exist_ok=True)
    file_name = file_path.replace(img_root, "")
    file_name_dirs = file_name.split("/")
    for dir in file_name_dirs[:-1]:
        folder = os.path.join(save_folder, dir)
        os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(save_folder, file_name)
    # np.savez(save_path, resized_array)
    nii_img = nib.Nifti1Image(resized_array, target_affine)
    nib.save(nii_img, save_path)



if __name__ == "__main__":
    nii_files = read_nii_files(img_root)
    nii_files = sorted(nii_files) #sort the files

    num_workers = 32  # Number of worker processes
    # Process files using multiprocessing with tqdm progress bar
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_file, nii_files), total=len(nii_files)))

    
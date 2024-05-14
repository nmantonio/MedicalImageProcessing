import os
import pydicom
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt, animation
from skimage.transform import resize
from scipy.ndimage import rotate, shift
from scipy.optimize import least_squares

# Resize usando skimage.transform
# Transform usando scipy.ndimage rotate, shift
# Utiilizar MSE

# def normalize_img(img):
#     return (img - (-1000)) / (1000 - (-1000))

def normalize_img(img): 
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def resize_img(img, output_dim=(64, 64)):
    return resize(img, (img.shape[0],) + output_dim)

def resize_as(img, reference):
    return resize(img, reference.shape)

def apply_threshold(img, th):
    return img

atlas_path = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing\AAL3_1mm.dcm"
input_path = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing\RM_Brain_3D-SPGR"
referenced_path = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing\icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm"

# Input image
unsorted_input_info = []
for idx, slice_path in enumerate(os.listdir(input_path)):
    dcm_img = pydicom.dcmread(os.path.join(input_path, slice_path))
    unsorted_input_info.append({"img": dcm_img.pixel_array, "pos": dcm_img["ImagePositionPatient"].value[2]})

sorted_list = sorted(unsorted_input_info, key=lambda x: x["pos"], reverse=False)
input_img = np.array([el["img"] for el in sorted_list])
input_img = input_img[50:(input_img.shape[0]), 0:(input_img.shape[1] - 40), 70:(input_img.shape[2] - 40)]
input_img = normalize_img(input_img)

# Referenced atlas image
referenced_dcm = pydicom.dcmread(referenced_path)
referenced_img = referenced_dcm.pixel_array
referenced_img = normalize_img(referenced_img)
# referenced_img = np.flip(referenced_img, axis=0)

import cv2

# referenced_img = resize_img(referenced_img)
# input_img = resize_as(input_img, reference=referenced_img)

def mse_3d_array(arr1, arr2):
    diff = arr1 - arr2
    squared_diff = np.square(diff)
    mse = np.mean(squared_diff)
    return mse

import numpy as np

def translate(image, translation):
    t1, t2, t3 = translation
    return shift(image, shift=(t1, t2, t3))

def rotate_img(image, angle, axis):
    return rotate(image, angle, axes=axis, reshape=False)

def three_axis_rotation(image, angles):
    angle_0, angle_1, angle_2 = angles
    image = rotate_img(image, angle_0, axis=(1, 2))
    image = rotate_img(image, angle_1, axis=(0, 2))
    image = rotate_img(image, angle_2, axis=(0, 1))
    return image
    
def translation_then_axialrotation(image, parameters: tuple[float, ...]):
    """ Apply to `point` a translation followed by an axial rotation, both defined by `parameters`. """
    t1, t2, t3, angle_0, angle_1, angle_2 = parameters
    translated = translate(image, translation=(t1, t2, t3))
    axial_rotated = three_axis_rotation(translated, angles=(angle_0, angle_1, angle_2))
    return axial_rotated

def vector_of_residuals(ref_image, input_image):
    return (ref_image - input_image).flatten()


def coregister_images(ref_image, input_image):
    """ Coregister two sets of landmarks using a rigid transformation. """
    initial_parameters = (
        0, 0, 0, # t1, t2, t3 (translation)
        175, 20, 0 # angle_0, angle_1, angle_2 (rotations)
    )

    def function_to_minimize(parameters):
        """ Transform input landmarks, then compare with reference landmarks."""        
        moved_image = translation_then_axialrotation(input_image, parameters=parameters)
        
        return vector_of_residuals(ref_image, moved_image)

    # Apply least squares optimization
    result = least_squares(
        function_to_minimize,
        x0=initial_parameters,
        # max_nfev=200,
        verbose=2)
    return result


# results = coregister_images(referenced_img, input_img)
# print(results)
# print(dir(results))

rotated_input_img = translation_then_axialrotation(input_img, parameters=(2.784, -1.497, 1.721, 175.7, 0.2669, 0.7668))

import time
# for slice_ref, slice_input in zip(referenced_img, rotated_input_img):
for idx in range(50):
    slice_ref = referenced_img[:, idx, :]
    slice_input = rotated_input_img[:, idx, :]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(slice_ref, cmap="bone")
    ax1.set_title('Referenced Image')
    ax1.axis('off')
    ax2.imshow(slice_input, cmap="bone")
    ax2.set_title('Input Image')
    ax2.axis('off')
    plt.show()
    time.sleep(0.5)
    plt.close(fig)
    


# Extract




# Read atlas
# atlas_dcm = pydicom.dcmread(atlas_path)
# print(atlas_dcm)

# atlas = atlas_dcm.pixel_array

# thalamus_atlas = atlas[atlas >= 121] & atlas[atlas <= 150]

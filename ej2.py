import os
import pydicom
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt, animation
from skimage.transform import resize
from scipy.ndimage import rotate, shift
from scipy.optimize import least_squares

from utils import create_rotation
root = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing"
os.makedirs(os.path.join(root, "results", "2"), exist_ok=True)


def expand_images(img1, img2):
    # Get the shapes of both images
    shape1 = img1.shape
    shape2 = img2.shape
    
    # Determine the maximum dimensions
    max_shape = [max(shape1[i], shape2[i]) for i in range(3)]
    
    # Calculate the number of additional pixels on each side
    add_pixels1 = [(max_shape[i] - shape1[i]) // 2 for i in range(3)]
    add_pixels2 = [(max_shape[i] - shape2[i]) // 2 for i in range(3)]
    
    # Create new arrays to hold the expanded images
    expanded_img1 = np.zeros(max_shape, dtype=img1.dtype)
    expanded_img2 = np.zeros(max_shape, dtype=img2.dtype)
    
    # Calculate the starting and ending indices for pasting images
    start1 = tuple(add_pixels1[i] for i in range(3))
    end1 = tuple(add_pixels1[i] + shape1[i] for i in range(3))
    start2 = tuple(add_pixels2[i] for i in range(3))
    end2 = tuple(add_pixels2[i] + shape2[i] for i in range(3))
    
    # Paste images into the expanded arrays
    expanded_img1[start1[0]:end1[0], start1[1]:end1[1], start1[2]:end1[2]] = img1
    expanded_img2[start2[0]:end2[0], start2[1]:end2[1], start2[2]:end2[2]] = img2
    
    return expanded_img1, expanded_img2

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

atlas_path = os.path.join(root, "AAL3_1mm.dcm")
input_path = os.path.join(root, "RM_Brain_3D-SPGR")
referenced_path = os.path.join(root, "icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm")

# Referenced atlas image
referenced_dcm = pydicom.dcmread(referenced_path)
referenced_img = referenced_dcm.pixel_array
referenced_img = normalize_img(referenced_img)
referenced_img = np.flip(referenced_img, axis=0)

pixel_len_mm = [2.0, 0.5078, 0.5078]
aspect_ratio = pixel_len_mm[0]/pixel_len_mm[1]
# Input image
unsorted_input_info = []
for idx, slice_path in enumerate(os.listdir(input_path)):
    dcm_img = pydicom.dcmread(os.path.join(input_path, slice_path))
    unsorted_input_info.append({"img": dcm_img.pixel_array, "pos": dcm_img["ImagePositionPatient"].value[2]})

sorted_list = sorted(unsorted_input_info, key=lambda x: x["pos"], reverse=False)
input_img = np.array([el["img"] for el in sorted_list])
# create_rotation(normalize_img(np.flip(input_img, axis=0)), n=128, name="pre_input", root=root, aspect_ratio=aspect_ratio)
input_img = input_img[80:(input_img.shape[0]), 0:(input_img.shape[1] - 40), 70:(input_img.shape[2] - 40)]
input_img = np.flip(input_img, axis=0)
input_img = normalize_img(input_img)
# input_img = np.concatenate((np.zeros((referenced_img.shape[0] - input_img.shape[0], input_img.shape[1], input_img.shape[2])), input_img), axis=0)

referenced_img = resize_img(referenced_img)
input_img = resize_as(input_img, reference=referenced_img)

for img, name in zip((input_img, referenced_img), ("post_input", "reference", "substraction")):
    if name =="reference":
        anim_aspect_ratio=1
    else: 
        anim_aspect_ratio=aspect_ratio
    create_rotation(img, n=128, name=name, root=root, aspect_ratio=1, show=False)

def mse_3d_array(arr1, arr2):
    diff = arr1 - arr2
    squared_diff = np.square(diff)
    mse = np.mean(squared_diff)
    return mse

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
        method="lm", 
        max_nfev=250,
        verbose=2)
    return result


results = coregister_images(referenced_img, input_img)
print(results)
print(results.x)
print(dir(results))

rotated_input_img = translation_then_axialrotation(input_img, parameters=results.x)

for ref, inp in zip(referenced_img[40:], rotated_input_img[40:]):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first image
    axes[0].imshow(ref, cmap="bone")
    axes[0].set_title('Reference')

    # Plot the second image
    axes[1].imshow(inp, cmap="bone")
    axes[1].set_title('Input')

    # Hide the axes
    for ax in axes:
        ax.axis('off')

    # Show the plot
    plt.show()
    


# Extract
# Read atlas
# atlas_dcm = pydicom.dcmread(atlas_path)
# print(atlas_dcm)

# atlas = atlas_dcm.pixel_array

# thalamus_atlas = atlas[atlas >= 121] & atlas[atlas <= 150]

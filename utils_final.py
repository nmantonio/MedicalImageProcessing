import numpy as np

from scipy.ndimage import rotate, shift
from scipy.optimize import minimize
import matplotlib
from matplotlib import pyplot as plt, animation
import os

def apply_colormap(img, colormap):
    return matplotlib.colormaps[colormap](img)

def alpha_fusion(image, seg, non_colormapped_seg=False, ALPHA=0.25):
    fused_image = image*ALPHA + seg*(1 - ALPHA)
    # if non_colormapped_seg:
    fused_image[non_colormapped_seg == 0] = image[non_colormapped_seg == 0]
    return fused_image

def MIP_sagittal_plane(img: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img, axis=2)

def MIP_coronal_plane(img: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img, axis=1)

def rotate_on_axial_plane(img: np.ndarray, angle_in_degrees: float, mode: str) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    if mode == "img":
        return rotate(img, angle=angle_in_degrees, axes=[1, 2], reshape=False, mode="constant", order=3)

    elif (mode == "realistic") or (mode == "nonrealistic"):
        return rotate(img, angle=angle_in_degrees, axes=[1, 2], reshape=False, mode="constant", order=0)

def get_projection(img: np.ndarray) -> np.ndarray:
    """ Create the point-of-view-dependent representation of the segmentation projection """
    non_zero_indices = np.nonzero(img)
    projection = np.zeros((img.shape[0], img.shape[2]), dtype=np.uint8)
    projection[non_zero_indices[0], non_zero_indices[1]] = img[non_zero_indices]
    return projection

def create_projections(img, n, mode):
    """
    Given an image and a number of desired projections, 
    compute the rotation and needed projections to obtain the animated image
    """
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
        print(f"{mode}: {idx}")
        rotated_img = rotate_on_axial_plane(img, alpha, mode=mode)
        if mode == "img" or mode == "nonrealistic":
            projection = MIP_sagittal_plane(rotated_img)
            # projection = MIP_coronal_plane(rotated_img)
        elif mode == "realistic":
            projection = get_projection(rotated_img)

        projections.append(projection)  # Save for later animation
    return projections

def create_rotation(img, n, name, root, results_path, aspect_ratio=1, show=False, cmap=False):
    projections = create_projections(img, n=n, mode="img")

    img_min = np.amin(img)
    img_max = np.amax(img)

    fig, ax = plt.subplots()
    plt.axis('off')

    if cmap: 
        animation_data = [
            [plt.imshow(proj, animated=True, vmin=img_min, vmax=img_max, cmap=cmap, aspect=aspect_ratio)]
            for proj in projections
        ]
    else: 
        animation_data = [
            [plt.imshow(proj, animated=True, vmin=img_min, vmax=img_max, aspect=aspect_ratio)]
            for proj in projections
        ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                interval=0.390625*n, blit=True)
    anim.save(os.path.join(results_path, f"{name}_rotation.gif"))
    if show:
        plt.show() 

def save_slice(image, axis, filename, pixel_len_mm, cmap='bone', legend_labels=False):
    """
    Guarda una slice de una imagen 3D como un archivo PNG.
    
    :param image: numpy.ndarray, la imagen 3D.
    :param axis: str, el eje sobre el cual tomar la slice.
    :param filename: str, el nombre del archivo para guardar la slice.
    """
    if axis == 'axial':
        slice_data = image[image.shape[0] // 2, :, :]
        slice_data = image[300, :, :]
        aspect_ratio = pixel_len_mm[1] / pixel_len_mm[2]
    elif axis == 'coronal':
        slice_data = image[:, image.shape[1] // 2, :]
        aspect_ratio = pixel_len_mm[0] / pixel_len_mm[2]
    elif axis == 'sagittal':
        slice_data = image[:, :, image.shape[2] // 2]
        aspect_ratio = pixel_len_mm[0] / pixel_len_mm[1]
    else:
        raise ValueError("El eje debe ser 'axial', 'coronal' o 'sagittal'")
    fig, ax = plt.subplots()
    if legend_labels: 
        # Plotting legend
        legend_colors = [matplotlib.colormaps['tab10'](seg_idx) for seg_idx in range(1, len(legend_labels)+1)]
        legend_handles = [matplotlib.patches.Patch(color=color, label=label) 
                        for label, color in zip(legend_labels, legend_colors)]
        ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize='x-small')
        
    plt.imshow(slice_data, cmap=cmap, aspect=aspect_ratio)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close() 

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

def expand_images_unilateral(img1, img2, axis):
    # Get the shapes of both images
    shape1 = img1.shape
    shape2 = img2.shape
    
    # Determine the maximum dimensions
    max_shape1 = list(shape1)
    max_shape2 = list(shape2)
    max_shape1[axis] = max(shape1[axis], shape2[axis])
    max_shape2[axis] = max(shape1[axis], shape2[axis])
    
    # Create new arrays to hold the expanded images
    expanded_img1 = np.zeros(max_shape1, dtype=img1.dtype)
    expanded_img2 = np.zeros(max_shape2, dtype=img2.dtype)
    
    # Calculate the number of additional pixels on each side
    add_pixels1 = max_shape1[axis] - shape1[axis]
    add_pixels2 = max_shape2[axis] - shape2[axis]
    
    # Calculate the starting and ending indices for pasting images
    start1 = max_shape1[axis] - shape1[axis]
    end1 = max_shape1[axis]
    start2 = max_shape2[axis] - shape2[axis]
    end2 = max_shape2[axis]
    
    # Paste images into the expanded arrays along the specified axis
    if axis == 0:
        expanded_img1[start1:end1, :, :] = img1
        expanded_img2[start2:end2, :, :] = img2
    elif axis == 1:
        expanded_img1[:, start1:end1, :] = img1
        expanded_img2[:, start2:end2, :] = img2
    elif axis == 2:
        expanded_img1[:, :, start1:end1] = img1
        expanded_img2[:, :, start2:end2] = img2
    else:
        raise ValueError("Invalid axis, must be 0, 1, or 2")
    
    return expanded_img1, expanded_img2

def minmax_normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def mse_3d_array(arr1, arr2):
    diff = arr1 - arr2
    squared_diff = np.square(diff)
    mse = np.mean(squared_diff)
    return mse

def translate(image, translation, **kwargs):
    t1, t2, t3 = translation
    return shift(image, shift=(t1, t2, t3), **kwargs)

def rotate_img(image, angle, axis ,**kwargs):
    return rotate(image, angle, axes=axis, reshape=False, **kwargs)

def three_axis_rotation(image, angles, **kwargs):
    angle_0, angle_1, angle_2 = angles
    image = rotate_img(image, angle_0, axis=(1, 2), **kwargs)
    image = rotate_img(image, angle_1, axis=(0, 2), **kwargs)
    image = rotate_img(image, angle_2, axis=(0, 1), **kwargs)
    return image
    
def translation_then_axialrotation(image, parameters: tuple[float, ...]):
    """ Apply to `point` a translation followed by an axial rotation, both defined by `parameters`. """
    t1, t2, t3, angle_0, angle_1, angle_2 = parameters
    translated = translate(image, translation=(t1, t2, t3))
    axial_rotated = three_axis_rotation(translated, angles=(angle_0, angle_1, angle_2))
    return axial_rotated

def inverse_transformation(img, params, **kwargs):
    t1, t2, t3, angle0, angle1, angle2 = params
    rotated = three_axis_rotation(img, angles=(-angle0, -angle1, -angle2), **kwargs)
    translated = translate(rotated, translation=(-t1, -t2, -t3), **kwargs)
    return translated

def coregister_images(ref_image, input_image, initial_parameters=False):
    """ Coregister two sets of landmarks using a rigid transformation. """
    if not initial_parameters:
        initial_parameters = (
            20, 0, 0, # t1, t2, t3 (translation)
            0, 0, 0 # angle_0, angle_1, angle_2 (rotations)
        )

    def function_to_minimize(parameters):
        """ Transform input landmarks, then compare with reference landmarks."""
        print(parameters)        
        moved_image = translation_then_axialrotation(input_image, parameters=parameters)
        
        return mse_3d_array(ref_image, moved_image)

    result = minimize(
        function_to_minimize,
        x0=initial_parameters,
        method="Powell"
        )
    return result
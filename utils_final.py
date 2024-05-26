import numpy as np

from scipy.ndimage import rotate, shift
from scipy.optimize import minimize


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

def inverse_transformation(img, params):
    t1, t2, t3, angle0, angle1, angle2 = params
    rotated = three_axis_rotation(img, angles=(-angle0, -angle1, -angle2))
    translated = translate(rotated, translation=(-t1, -t2, -t3))
    return translated

def coregister_images(ref_image, input_image, initial_parameters=False):
    """ Coregister two sets of landmarks using a rigid transformation. """
    if not initial_parameters:
        initial_parameters = (
            -20, 0, 0, # t1, t2, t3 (translation)
            160, 0, 0 # angle_0, angle_1, angle_2 (rotations)
        )

    def function_to_minimize(parameters):
        """ Transform input landmarks, then compare with reference landmarks."""        
        moved_image = translation_then_axialrotation(input_image, parameters=parameters)
        
        return mse_3d_array(ref_image, moved_image)

    result = minimize(
        function_to_minimize,
        x0=initial_parameters,
        # method="Powell"
        method='BFGS',
        options={'maxiter': 1000, 'disp': True}
        )
    return result
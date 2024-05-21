import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt, animation


from scipy.ndimage import rotate

def MIP_sagittal_plane(img: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img, axis=2)

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """    
    return rotate(img_dcm, angle=angle_in_degrees, axes=[1, 2], reshape=False, mode="constant", order=3)

# def get_projection(img: np.ndarray) -> np.ndarray:
#     projection = np.zeros((img.shape[0], img.shape[2]), dtype=np.uint8)
#     for j in range(img.shape[1]): 
#         for i in range(img.shape[0]):
#             for k in range(img.shape[2]):
#                 if img[i, j, k] != 0:
#                     projection[i, j] = img[i, j, k]
#                     break
#     return projection 

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
        rotated_img = rotate_on_axial_plane(img, alpha)
        if mode == "img":
            projection = MIP_sagittal_plane(rotated_img)
        elif mode == "seg":
            projection = get_projection(rotated_img)
        else: 
            raise ValueError("mode must be one of 'img' or 'seg'!")
        projections.append(projection)  # Save for later animation
    return projections

def alpha_fusion(image, seg, ALPHA=0.7):
    fused_image = image*ALPHA + seg*(1 - ALPHA)
    fused_image[seg == 0] = image[seg == 0]

    return fused_image

def apply_colormap(img, colormap):
    return matplotlib.colormaps[colormap](img)

def create_rotation(img, n, aspect_ratio, name, root, show=True):
    projections = create_projections(img, n=n, mode="img")

    img_min = np.amin(img)
    img_max = np.amax(img)

    fig, ax = plt.subplots()
    plt.axis('off')

    animation_data = [
        [plt.imshow(proj, animated=True, cmap="bone", vmin=img_min, vmax=img_max, aspect=aspect_ratio)]
        for proj in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                interval=0.390625*n, blit=True)
    anim.save(os.path.join(root, "results", "2", f"{name}_rotation.gif"))
    if show:
        plt.show() 
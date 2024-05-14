import os
import pydicom
import numpy as np
import scipy
import matplotlib

from matplotlib import pyplot as plt, animation
from scipy.ndimage import rotate
import cv2

root = r"/home/slimbook/Escritorio/Antonio/MedicalImageProcessing"
med_folder = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/103.000000-LIVER 3 PHASE CAP-83135")
segmentation_path = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/300.000000-Segmentation-91221/1-1.dcm")

pixel_len_mm = [2.5, 0.742188, 0.742188] # Slice thikness
aspect_ratio = pixel_len_mm[0] / pixel_len_mm[1]

# Images -------------------------------------------------------------------
unsorted_image_info = []
acquisition_numers = set()
for idx, slice_path in enumerate(os.listdir(med_folder)):
    dcm_img = pydicom.dcmread(os.path.join(med_folder, slice_path))
    acquisition_numers.add(dcm_img["AcquisitionNumber"].value)
    unsorted_image_info.append({"img": dcm_img.pixel_array, "pos": dcm_img["ImagePositionPatient"].value[2]})
print("Only one acquisition value!" if len(acquisition_numers) == 1 else "More than one acquisition value found !")
    
sorted_list = sorted(unsorted_image_info, key=lambda x: x["pos"], reverse=True)
img = np.array([el["img"] for el in sorted_list])
# Windowing
img = np.where(np.logical_and(-120<=img, img<=1200), img, 0)
# Normalization (minmax)
img = (img - (np.min(img))) / (np.max(img) - (np.min(img)))

# Segmentations -------------------------------------------------------------------
seg_dcm = pydicom.dcmread(segmentation_path) 

segmentations = {}
segmentations_labels = {}

for segseq in seg_dcm["SegmentSequence"].value:
    seg_idx = segseq["SegmentNumber"].value
    segmentations[seg_idx] = {
        "SegmentDescription": segseq["SegmentDescription"].value, 
        "SegmentationData": [],
    }
    segmentations_labels[seg_idx] = segseq["SegmentLabel"].value

    
seg_array = seg_dcm.pixel_array
for slice_idx, element in enumerate(seg_dcm["PerFrameFunctionalGroupsSequence"]):
    seg_idx = element["SegmentIdentificationSequence"][0]["ReferencedSegmentNumber"].value
    segmentations[seg_idx]["SegmentationData"].append({
        "ImagePositionPatient": float(element["PlanePositionSequence"][0]["ImagePositionPatient"].value[2]),
        "SliceArray": seg_array[slice_idx]
    })
    
for seg_idx in segmentations.keys():
    segmentations[seg_idx]["SegmentationSortedData"] = sorted(segmentations[seg_idx]["SegmentationData"], key=lambda x: x["ImagePositionPatient"], reverse=True)
    segmentations[seg_idx]["SegmentationSortedArray"] = np.array([el["SliceArray"] for el in segmentations[seg_idx]["SegmentationSortedData"]])

segmentations_img = np.zeros_like(img, dtype=np.uint8)
for seg_idx in segmentations.keys():
    segmentations_img[segmentations[seg_idx]["SegmentationSortedArray"]!= 0] = int(seg_idx)

def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    projection = np.max(img_dcm, axis=2)
    return projection

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """    
    return rotate(img_dcm, angle=angle_in_degrees, axes=[1, 2], reshape=False, mode="nearest", order=0)

# def get_projection(array_3d: np.ndarray) -> np.ndarray:
#     projection = np.zeros((array_3d.shape[0], array_3d.shape[2]), dtype=np.uint8)
#     for j in range(array_3d.shape[1]): 
#         for i in range(array_3d.shape[0]):
#             for k in range(array_3d.shape[2]):
#                 if array_3d[i, j, k] != 0:
#                     projection[i, j] = array_3d[i, j, k]
#                     break
#     return projection 

def get_projection(array_3d: np.ndarray) -> np.ndarray:
    non_zero_indices = np.nonzero(array_3d)
    projection = np.zeros((array_3d.shape[0], array_3d.shape[2]), dtype=np.uint8)
    projection[non_zero_indices[0], non_zero_indices[1]] = array_3d[non_zero_indices]
    return projection



def create_projections(img, n, mode):
    """
    #TODO
    """
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
        print(f"{mode}: {idx}")
        rotated_img = rotate_on_axial_plane(img, alpha)
        # if mode == "img":
        if True:
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

n = 256
projections = create_projections(img, n=n, mode="img")
seg_projections = create_projections(segmentations_img, n=n, mode="seg")

colormapped_projections = [apply_colormap(projection, "bone") for projection in projections]
colormapped_seg_projections = [apply_colormap(seg_projection, "tab10") for seg_projection in seg_projections]

fused_projections = [alpha_fusion(proj, seg_proj) for proj, seg_proj in zip(colormapped_projections, colormapped_seg_projections)]

# Save and visualize animation
img_min = np.amin(img)
img_max = np.amax(img)

fig, ax = plt.subplots()
plt.axis('off')
legend_labels = [segmentations_labels[seg_idx] for seg_idx in segmentations.keys()]

# Plotting legend
legend_colors = [matplotlib.colormaps['tab10'](seg_idx) for seg_idx in range(1, len(legend_labels)+1)]
legend_handles = [matplotlib.patches.Patch(color=color, label=label) 
                for label, color in zip(legend_labels, legend_colors)]
ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize='x-small')


animation_data = [
    [plt.imshow(img, animated=True, vmin=img_min, vmax=img_max, aspect=aspect_ratio)]
    for img in fused_projections
]
anim = animation.ArtistAnimation(fig, animation_data,
                            interval=25, blit=True)
anim.save(os.path.join(root, "results", "1", "simple_rotation.gif"))
# plt.show() 

# # Define the number of subplots
# num_subplots = len(fused_projections)

# # Create a figure and axes
# fig, axes = plt.subplots(1, num_subplots, figsize=(15, 5))

# # Plot each fused projection
# for i, fused_projection in enumerate(fused_projections):
#     axes[i].imshow(fused_projection, vmin=np.amin(img), vmax=np.amax(img), aspect=pixel_len_mm[0] / pixel_len_mm[1])  # You can change the colormap if needed
#     axes[i].set_title(f'Projection {i+1}')

# # Adjust layout
# plt.tight_layout()
# plt.show()


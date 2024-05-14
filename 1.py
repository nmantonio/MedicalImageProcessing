import os
import pydicom
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt, animation
from matplotlib.colors import ListedColormap
from scipy.ndimage import rotate
import cv2

tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
tab10_5_colors = tab10_colors[:5]  # Take the first 5 colors from the tab10 colormap
custom_cmap = ListedColormap(tab10_5_colors)

root = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing"
med_folder = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/103.000000-LIVER 3 PHASE CAP-83135")
segmentation_path = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/300.000000-Segmentation-91221/1-1.dcm")

pixel_len_mm = [2.5, 0.742188, 0.742188] # Slice thikness

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
img = (img - (np.min(img))) / (np.max(img) - (np.min(img)))

# Segmentations -------------------------------------------------------------------
seg_dcm = pydicom.dcmread(segmentation_path) 

segmentations = {}
for segseq in seg_dcm["SegmentSequence"].value:
    seg_idx = segseq["SegmentNumber"].value
    segmentations[seg_idx] = {
        "SegmentDescription": segseq["SegmentDescription"].value, 
        "SegmentationData": [],
    }
    
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

segmentations_img = np.zeros_like(img)
for seg_idx in segmentations.keys():
    segmentations_img[segmentations[seg_idx]["SegmentationSortedArray"]!= 0] = int(seg_idx)


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img_dcm, axis=2)

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """    
    return rotate(img_dcm, angle=angle_in_degrees, axes=[1, 2], reshape=False, mode="nearest", order=0)

def get_projection(array_3d: np.ndarray) -> np.ndarray:
    projection = np.zeros((array_3d.shape[0], array_3d.shape[2]))
    for k in range(array_3d.shape[2]):
        for i in range(array_3d.shape[0]):
            for j in range(array_3d.shape[1]): 
                if array_3d[i, j, k] != 0:
                    projection[i, k] = array_3d[i, j, k]
                    break
    return projection 
                

def create_projections(img, n, mode):
    """
    #TODO
    """
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
        rotated_img = rotate_on_axial_plane(img, alpha)
        if mode == "img":
            projection = MIP_sagittal_plane(rotated_img)
        elif mode == "seg":
            projection = get_projection(rotated_img)
        else: 
            raise ValueError("mode must be one of 'img' or 'seg'!")
        projections.append(projection)  # Save for later animation
    return projections

n = 4
img_min = np.min(img)
img_max = np.max(img)
cm = plt.cm.bone
cm_tab = plt.cm.tab10
projections = create_projections(img, n=n, mode="img")
seg_projections = create_projections(segmentations_img, n=n, mode="seg")

ALPHA = 0.3
combined_projections = []
for proj, seg_proj in zip(projections, seg_projections):
    print(np.unique(proj))

    proj = cm(proj)
    seg_proj = cm_tab(seg_proj)
    print(np.unique(proj))
    combined_proj = ALPHA*seg_proj + (1 - ALPHA)*proj
    print(combined_proj.shape)
    combined_projections.append(combined_proj)
    
fig, ax = plt.subplots()
plt.axis('off')
# Save and visualize animation
animation_data = [
    [plt.imshow(proj, animated=True, vmin=0, vmax=1, cmap=matplotlib.colormaps["bone"], aspect=pixel_len_mm[0] / pixel_len_mm[1])]
    for proj in combined_projections
]
anim = animation.ArtistAnimation(fig, animation_data,
                            interval=250, blit=True)
anim.save('results/MIP/Animation.gif')  # Save animation
plt.show()                              # Show animation


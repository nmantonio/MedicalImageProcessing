import os
import pydicom
import numpy as np
import scipy

root = r"/home/slimbook/Escritorio/Antonio/MedicalImageProcessing"
med_folder = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/103.000000-LIVER 3 PHASE CAP-83135")
segmentation_path = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/300.000000-Segmentation-91221/1-1.dcm")

# for idx, slice_path in enumerate(os.listdir(med_folder)[::-1]):
    # dcm_img = pydicom.dcmread(os.path.join(med_folder, slice_path))
    # print(dcm_img)
    # print(dcm_img["AcquisitionNumber"])
    # break
    

seg_dcm = pydicom.dcmread(segmentation_path) 
# seg_groups = np.unique([elem["SegmentIdentificationSequence"][0]["ReferencedSegmentNumber"].value for elem in seg_dcm["PerFrameFunctionalGroupsSequence"]])
# print(dir(seg_dcm))
segmentations = {}
for segseq in seg_dcm["SegmentSequence"].value:
    seg_idx = segseq["SegmentNumber"].value
    segmentations[seg_idx] = {
        "SegmentDescription": segseq["SegmentDescription"].value, 
        "SegmentationData": [],
    }
    

# print(segmentations)
# print(dir(seg_dcm))
# print(seg_dcm["PerFrameFunctionalGroupsSequence"])
seg_array = seg_dcm.pixel_array
for slice_idx, element in enumerate(seg_dcm["PerFrameFunctionalGroupsSequence"]):
    seg_idx = element["SegmentIdentificationSequence"][0]["ReferencedSegmentNumber"].value
    segmentations[seg_idx]["SegmentationData"].append({
        "ImagePositionPatient": float(element["PlanePositionSequence"][0]["ImagePositionPatient"].value[2]),
        "SliceArray": seg_array[slice_idx]
    })
    
for seg_idx in segmentations.keys():
    segmentations[seg_idx]["SegmentationSortedData"] = np.array(sorted(segmentations[seg_idx]["SegmentationData"], key=lambda x: x["ImagePositionPatient"], reverse=True))
    segmentations[seg_idx]["SegmentationSortedArray"] = np.array([el["SliceArray"] for el in segmentations[seg_idx]["SegmentationSortedData"]])
    print(segmentations[seg_idx]["SegmentationSortedArray"].shape)
    # print(float(element["PlanePositionSequence"][0]["ImagePositionPatient"].value[2]))
    
img = segmentations[3]["SegmentationSortedArray"]

def median_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """
    return img_dcm[:, :, img_dcm.shape[1]//2]    # Why //2?


def median_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """
    return img_dcm[:, img_dcm.shape[2]//2, :]


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    # Your code here:
    #   See `np.max(...)`
    # ...
    
    return np.max(img_dcm, axis=2)


def AIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the average intensity projection on the sagittal orientation. """
    # Your code here:
    #   See `np.mean(...)`
    # ...
    
    return np.mean(img_dcm, axis=2)


def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """
    # Your code here:
    # ...
    return np.max(img_dcm, axis=1)


def AIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the average intensity projection on the coronal orientation. """
    # Your code here:
    # ...
    return np.mean(img_dcm, axis=1)

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    # Your code here:
    #   See `scipy.ndimage.rotate(...)`
    # ...
    
    return scipy.ndimage.rotate(img_dcm, angle=angle_in_degrees, axes=[1, 2], reshape=False, mode="nearest")

from matplotlib import pyplot as plt
import matplotlib
from matplotlib import pyplot as plt, animation

pixel_len_mm = [2.5, 0.742188, 0.742188] # Slice thikness 
# pixel_len_mm = [1, 1, 1]
# Show median planes
fig, ax = plt.subplots(1, 2)
ax[0].imshow(median_sagittal_plane(img), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
ax[0].set_title('Sagittal')
ax[1].imshow(median_coronal_plane(img), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[2])
ax[1].set_title('Coronal')
fig.suptitle('Median planes')
plt.show()

# Show MIP/AIP/Median planes
fig, ax = plt.subplots(1, 3)
ax[0].imshow(median_sagittal_plane(img), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
ax[0].set_title('Median')
ax[1].imshow(MIP_sagittal_plane(img), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
ax[1].set_title('MIP')
ax[2].imshow(AIP_sagittal_plane(img), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
ax[2].set_title('AIP')
fig.suptitle('Sagittal')
plt.show()
fig, ax = plt.subplots(1, 3)
ax[0].imshow(median_coronal_plane(img), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
ax[0].set_title('Median')
ax[1].imshow(MIP_coronal_plane(img), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
ax[1].set_title('MIP')
ax[2].imshow(AIP_coronal_plane(img), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
ax[2].set_title('AIP')
fig.suptitle('Coronal')
plt.show()

# Create projections varying the angle of rotation
#   Configure visualization colormap
img_min = np.amin(img)
img_max = np.amax(img)
cm = matplotlib.colormaps['bone']
fig, ax = plt.subplots()
ax.axis('off')
#   Configure directory to save results
os.makedirs('results/MIP/', exist_ok=True)
#   Create projections
n = 18
projections = []
for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
    rotated_img = rotate_on_axial_plane(img, alpha)
    projection = MIP_sagittal_plane(rotated_img)
    plt.imshow(projection, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
    plt.savefig(f'results/MIP/Projection_{idx}.png')      # Save animation
    projections.append(projection)  # Save for later animation
# Save and visualize animation
animation_data = [
    [plt.imshow(img, animated=True, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])]
    for img in projections
]
anim = animation.ArtistAnimation(fig, animation_data,
                            interval=250, blit=True)
anim.save('results/MIP/Animation.gif')  # Save animation
plt.show()                              # Show animation


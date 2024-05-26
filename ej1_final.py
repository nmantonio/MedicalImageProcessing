import os
import pydicom
import numpy as np
import matplotlib

from matplotlib import pyplot as plt, animation
from scipy.ndimage import rotate

from utils_final import minmax_normalize, save_slice, alpha_fusion, apply_colormap, create_projections

root = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing"
med_folder = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/103.000000-LIVER 3 PHASE CAP-83135")
segmentation_path = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/300.000000-Segmentation-91221/1-1.dcm")
results_path = os.path.join(root, "results", "1")

# ###########################################################
# ----------------------- Read Images -----------------------
# ###########################################################

# Read all slices and sort them using header ImagePositionPatient
## Also check that only one adquisition is present, using header AcquisitionNumber
unsorted_image_info = []
acquisition_numers = set()
for idx, slice_path in enumerate(os.listdir(med_folder)):
    dcm_img = pydicom.dcmread(os.path.join(med_folder, slice_path))
    if idx == 0: 
        pixel_len_mm = [float(dcm_img["SliceThickness"].value), float(dcm_img["PixelSpacing"][0]), float(dcm_img["PixelSpacing"][1])] # to be used in aspect_ratio calculations
        print(pixel_len_mm)
    acquisition_numers.add(dcm_img["AcquisitionNumber"].value)
    unsorted_image_info.append({"img": dcm_img.pixel_array, "pos": dcm_img["ImagePositionPatient"].value[2]})
print("Only one acquisition value!" if len(acquisition_numers) == 1 else "More than one acquisition value found !")

sorted_list = sorted(unsorted_image_info, key=lambda x: x["pos"], reverse=True)
img_raw = np.array([el["img"] for el in sorted_list])

# Image Windowing and normalization
img = img_raw.copy()
img[img < -100] = -100
img[img > 800] = 800
img = minmax_normalize(img) # normalize image between 0 and 1

for idx, axis in enumerate(('axial', 'coronal', 'sagittal')):
    save_slice(img, axis=axis, filename=os.path.join(results_path, f'{axis}_slice.png'), pixel_len_mm=pixel_len_mm)

# ###########################################################
# -------------- Read and order Segmentations --------------
# ###########################################################

seg_dcm = pydicom.dcmread(segmentation_path) # read seg file

segmentations = {}
segmentations_labels = {}

# Create segmentation dict container for each index in SegmentSequence
for segseq in seg_dcm["SegmentSequence"].value:
    seg_idx = segseq["SegmentNumber"].value
    segmentations[seg_idx] = {
        "SegmentDescription": segseq["SegmentDescription"].value, 
        "SegmentationData": [],
    }
    segmentations_labels[seg_idx] = segseq["SegmentLabel"].value

# Get seg pixel array and extract ImagePositionPatient
seg_array = seg_dcm.pixel_array
for slice_idx, element in enumerate(seg_dcm["PerFrameFunctionalGroupsSequence"]):
    seg_idx = element["SegmentIdentificationSequence"][0]["ReferencedSegmentNumber"].value
    segmentations[seg_idx]["SegmentationData"].append({
        "ImagePositionPatient": float(element["PlanePositionSequence"][0]["ImagePositionPatient"].value[2]),
        "SliceArray": seg_array[slice_idx]
    })

# Sort segmentation arrays
for seg_idx in segmentations.keys():
    segmentations[seg_idx]["SegmentationSortedData"] = sorted(segmentations[seg_idx]["SegmentationData"], key=lambda x: x["ImagePositionPatient"], reverse=True)
    segmentations[seg_idx]["SegmentationSortedArray"] = np.array([el["SliceArray"] for el in segmentations[seg_idx]["SegmentationSortedData"]])

# Combine ordered seg arrays in a multi-level seg array
segmentations_img = np.zeros_like(img, dtype=np.uint8)
for seg_idx in segmentations.keys():
    segmentations_img[segmentations[seg_idx]["SegmentationSortedArray"]!= 0] = int(seg_idx)

# Get mid slices of each plane
for idx, axis in enumerate(('axial', 'coronal', 'sagittal')):
    save_slice(segmentations_img, axis=axis, filename=os.path.join(results_path, f'seg_{axis}_slice.png'), pixel_len_mm=pixel_len_mm)

# ###########################################################
# ------------- Overlay image and segmentation -------------
# ###########################################################
# Apply colormaps to image and segmentations
cm_img = apply_colormap(img, colormap='bone')
cm_seg = apply_colormap(segmentations_img, colormap='tab10')

# Overlay seg to img using alpha fusion and store mid slices on each plane
overlayed_img = alpha_fusion(cm_img, cm_seg, non_colormapped_seg=segmentations_img)
for idx, axis in enumerate(('axial', 'coronal', 'sagittal')):
    save_slice(overlayed_img, axis=axis, filename=os.path.join(results_path, f'overlayed_{axis}_slice.png'), pixel_len_mm=pixel_len_mm, legend_labels=[segmentations_labels[seg_idx] for seg_idx in segmentations.keys()])

# ###########################################################
# ---------------------- Rotating MIP ----------------------
# ###########################################################
n = 256
img = img_raw.copy()
img = np.where(np.logical_and(-120<=img, img<=1200), img, 0)
img = minmax_normalize(img)

projections = create_projections(img, n=n, mode="img")
seg_nonrealistic_projections = create_projections(segmentations_img, n=n, mode="nonrealistic")
seg_realistic_projections = create_projections(segmentations_img, n=n, mode="realistic")
for name, seg_projections in zip(("realistic", "basic"), (seg_realistic_projections, seg_nonrealistic_projections)):
    print(name)
    colormapped_projections = [apply_colormap(projection, "bone") for projection in projections]
    colormapped_seg_projections = [apply_colormap(seg_projection, "tab10") for seg_projection in seg_projections]

    fused_projections = [alpha_fusion(proj, seg_proj, noncm_seg_proj) for proj, seg_proj, noncm_seg_proj in zip(colormapped_projections, colormapped_seg_projections, seg_projections)]

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
        [plt.imshow(img, animated=True, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0]/pixel_len_mm[1])]
        for img in fused_projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                interval=0.390625*n, blit=True)
    anim.save(os.path.join(results_path, f"{name}_rotation.gif"))
    # plt.show() 
    plt.close()
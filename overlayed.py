import os
import pydicom
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt, animation

root = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing"
med_folder = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/103.000000-LIVER 3 PHASE CAP-83135")
segmentation_path = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/300.000000-Segmentation-91221/1-1.dcm")

pixel_len_mm = [2.5, 0.742188, 0.742188] # Slice thikness

# ------------------------ CT Scan ------------------------
img = {}
for idx, slice_path in enumerate(os.listdir(med_folder)):
    dcm_img = pydicom.dcmread(os.path.join(med_folder, slice_path))

    img[float(dcm_img["ImagePositionPatient"].value[2])] = dcm_img.pixel_array
    
whole_image = np.array([img[position] for position in sorted(img.keys(), reverse=True)])

# ------------------------ Segmentation ------------------------

seg_dcm = pydicom.dcmread(segmentation_path) 
ROWS, COLS = seg_dcm["Rows"].value, seg_dcm["Columns"].value

segmentations = {}
segmentations_labels = {}
for segseq in seg_dcm["SegmentSequence"].value:
    seg_idx = segseq["SegmentNumber"].value
    segmentations[seg_idx] = {}
    segmentations_labels[seg_idx] = segseq["SegmentLabel"].value
    
seg_array = seg_dcm.pixel_array
for slice_idx, element in enumerate(seg_dcm["PerFrameFunctionalGroupsSequence"]):
    seg_idx = element["SegmentIdentificationSequence"][0]["ReferencedSegmentNumber"].value
    segmentations[seg_idx][float(element["PlanePositionSequence"][0]["ImagePositionPatient"].value[2])] = seg_array[slice_idx]

# ------------------------ Overlay and save ------------------------
img_min = np.amin(whole_image)
img_max = np.amax(whole_image)
cm = matplotlib.colormaps['bone']

for seg_idx in segmentations.keys():
    mask = np.array([segmentations[seg_idx].get(position, np.zeros((ROWS, COLS))) for position in sorted(img.keys(), reverse=True)])

    fig, ax = plt.subplots()
    plt.axis('off')
    ALPHA = 0.6
    overlayed_data = []
    for slice_idx in range(whole_image.shape[0]):
        slice_img = matplotlib.colormaps['bone'](whole_image[slice_idx, :, :])
        slice_mask = mask[slice_idx, :, :]
        mask_plot = matplotlib.colormaps['Set1'](slice_mask)
        
        overlayed = (1-ALPHA)*slice_img + ALPHA*mask_plot
        overlayed[slice_mask == 0] = slice_img[slice_mask == 0]
        overlayed_data.append(overlayed)
        
        # plt.imshow(overlayed)
        # plt.show()
    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True, vmin=img_min, vmax=img_max)]
        for img in overlayed_data
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                              interval=250, blit=True)
    anim.save(f'results/1/Animation_seg{seg_idx}.gif')  # Save animation
    # plt.show()        


combined_masks = np.zeros_like(whole_image)
for seg_idx in segmentations.keys():
    mask = np.array([segmentations[seg_idx].get(position, np.zeros((ROWS, COLS))) for position in sorted(img.keys(), reverse=True)])
    combined_masks[mask!=0] = seg_idx

fig, ax = plt.subplots()
plt.axis('off')
ALPHA = 0.6
overlayed_data = []
for slice_idx in range(whole_image.shape[0]):
    slice_img = matplotlib.colormaps['bone'](whole_image[slice_idx, :, :])
    slice_mask = combined_masks[slice_idx, :, :]
    mask_plot = matplotlib.colormaps['tab10'](slice_mask)
    
    overlayed = (1-ALPHA)*slice_img + ALPHA*mask_plot
    overlayed[slice_mask == 0] = slice_img[slice_mask == 0]
    overlayed_data.append(overlayed)
    
    # plt.imshow(overlayed)
    # plt.show()
    
# Plotting legend
legend_labels = [segmentations_labels[seg_idx] for seg_idx in segmentations.keys()]
legend_colors = [matplotlib.colormaps['tab10'](seg_idx) for seg_idx in segmentations.keys()]
legend_handles = [matplotlib.patches.Patch(color=color, label=label) 
                  for label, color in zip(segmentations_labels.values(), legend_colors)]
ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize='x-small')

# Save and visualize animation
animation_data = [
    [plt.imshow(img, animated=True, vmin=img_min, vmax=img_max)]
    for img in overlayed_data
]
anim = animation.ArtistAnimation(fig, animation_data,
                            interval=250, blit=True)
anim.save('results/1/Animation_combined.gif')  # Save animation
# plt.show()        
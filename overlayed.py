import os
import pydicom
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt, animation


def get_overlayed_slice(slice_idx, image, mask, axis, alpha):
    """
    Return overlayed slice given 3D image, 3D mask, slice index and interested plane
    """
    slice_img = matplotlib.colormaps['bone'](np.take(image, slice_idx, axis=axis))
    slice_mask = np.take(mask, slice_idx, axis=axis)
    mask_plot = matplotlib.colormaps['tab10'](slice_mask)
    
    overlayed = (1-alpha)*slice_img + alpha*mask_plot
    overlayed[slice_mask == 0] = slice_img[slice_mask == 0]
    return overlayed
    

def get_overlay_animation(image, mask, plane, show=False, save_path=False, alpha=0.75, legend_labels=False, aspect_ratio=1):
    """
    Compute the overlay of a 3D mask over a 3D image, slice by slice, and create an animation.
    """
    plane = plane.lower()
    axis = {"sagittal": 2, "coronal": 1, "axial": 0}.get(plane, None)
    if axis is None:
        raise ValueError(f"{plane} plane not supported. Try 'sagittal', 'coronal' or 'axial'")
    
    overlayed_data = []
    for slice_idx in range(image.shape[axis]):
        overlayed = get_overlayed_slice(slice_idx, image, mask, axis, alpha=alpha)
        overlayed_data.append(overlayed)
    
    # Save and visualize animation
    img_min = np.amin(image)
    img_max = np.amax(image)
    
    fig, ax = plt.subplots()
    plt.axis('off')
    
    if legend_labels: 
        # Plotting legend
        legend_colors = [matplotlib.colormaps['tab10'](seg_idx) for seg_idx in range(1, len(legend_labels)+1)]
        legend_handles = [matplotlib.patches.Patch(color=color, label=label) 
                        for label, color in zip(legend_labels, legend_colors)]
        ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize='x-small')

    animation_data = [
        [plt.imshow(img, animated=True, vmin=img_min, vmax=img_max, aspect=aspect_ratio)]
        for img in overlayed_data
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                              interval=250, blit=True)
    if save_path:
        anim.save(save_path)
    if show:
        plt.show()        
    

root = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing"
med_folder = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/103.000000-LIVER 3 PHASE CAP-83135")
# med_folder = os.path.join(root, r"HCC-TACE-Seg\HCC_009\02-15-1998-NA-PP CAPLIVER PROTO.-10975\103.000000-LIVER 3 PHASE CAP-83135")
segmentation_path = os.path.join(root, "HCC-TACE-Seg/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/300.000000-Segmentation-91221/1-1.dcm")

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
# print(seg_dcm["SharedFunctionalGroupsSequence"][0]["PixelMeasuresSequence"][0]["PixelSpacing"].value[0])
pixel_len_mm = [float(seg_dcm["SharedFunctionalGroupsSequence"][0]["PixelMeasuresSequence"][0]["SliceThickness"].value), float(seg_dcm["SharedFunctionalGroupsSequence"][0]["PixelMeasuresSequence"][0]["PixelSpacing"].value[0]), float(seg_dcm["SharedFunctionalGroupsSequence"][0]["PixelMeasuresSequence"][0]["PixelSpacing"].value[1])] # Slice thikness
for plane in ('axial', 'coronal', 'sagittal'):
    if plane == 'axial':
        aspect_ratio = 1
    else:
        aspect_ratio = pixel_len_mm[0]/pixel_len_mm[1]

    # for seg_idx in segmentations.keys():
    #     mask = np.array([segmentations[seg_idx].get(position, np.zeros((ROWS, COLS))) for position in sorted(img.keys(), reverse=True)])

    #     get_overlay_animation(
    #         image=whole_image,
    #         mask=mask,
    #         plane=plane,
    #         show=False,
    #         save_path=f"results/1/{plane}_{seg_idx}.gif", 
    #         aspect_ratio=aspect_ratio
    #     )

    combined_masks = np.zeros_like(whole_image)
    for seg_idx in segmentations.keys():
        mask = np.array([segmentations[seg_idx].get(position, np.zeros((ROWS, COLS))) for position in sorted(img.keys(), reverse=True)])
        combined_masks[mask!=0] = seg_idx
        
    # whole_image = np.fliplr(whole_image)
    # combined_masks = np.fliplr(combined_masks)

    get_overlay_animation(
        image=whole_image, 
        mask=combined_masks,
        plane=plane,
        show=False,
        save_path=f"results/1/{plane}_combined.gif",
        legend_labels=[segmentations_labels[seg_idx] for seg_idx in segmentations.keys()], 
        aspect_ratio=aspect_ratio
    )
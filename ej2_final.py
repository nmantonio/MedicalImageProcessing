from utils_final import *
from utils_final import minmax_normalize, inverse_transformation, create_rotation, alpha_fusion, create_projections, apply_colormap

import os
import pydicom
from scipy.ndimage import zoom
from skimage.transform import resize
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, animation

root = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing"
results_path = os.path.join(root, "results", "2")
os.makedirs(os.path.join(results_path), exist_ok=True)

N = 16
CREATE_IMAGES = False
DO_COREGISTER = False

# ###########################################################
# ----------------- Reference image reading -----------------
# ###########################################################
print("----- Reference Image Treatment -----")
reference_path = os.path.join(root, "icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm")
reference_dcm = pydicom.dcmread(reference_path)
reference_img = reference_dcm.pixel_array
reference_img = minmax_normalize(reference_img) # Normalize reference image
print(f'Reference Image Orientation Patient: {reference_dcm["SharedFunctionalGroupsSequence"][0]["PlaneOrientationSequence"][0]["ImageOrientationPatient"].value}')
reference_img = np.rot90(reference_img, k=2, axes=(1, 2)) # Rotate reference image to match [1 0 0 0 1 0] convention
print(f'Reference Image fliped! [1 0 0 0 1 0]')
print(f'Reference Image Shape: {reference_img.shape}')

if CREATE_IMAGES:
    create_rotation(reference_img, n=N, name="reference", root=root, show=False, results_path=results_path, cmap='bone') # Create visualization

# ###########################################################
# ------------------- Input image reading -------------------
# ###########################################################
print("----- Input Image Treatment -----")
input_path = os.path.join(root, "RM_Brain_3D-SPGR") 
unsorted_input_info = []
for idx, slice_path in enumerate(os.listdir(input_path)): # Read input image slice by slice and store
    dcm_img = pydicom.dcmread(os.path.join(input_path, slice_path))
    if idx == 0:  # Get info about the slices: Image Orientation and Pixel Dimensions
        print(f'Input Image Orientation Patient: {dcm_img["ImageOrientationPatient"].value}')
        pixel_len_mm = [float(dcm_img["SliceThickness"].value), float(dcm_img["PixelSpacing"][0]), float(dcm_img["PixelSpacing"][1])]
        print(f'Input Pixel Length (mm): {pixel_len_mm}')

    unsorted_input_info.append({"img": dcm_img.pixel_array, "pos": dcm_img["ImagePositionPatient"].value[2]})

sorted_list = sorted(unsorted_input_info, key=lambda x: x["pos"], reverse=False) # Order input image slices by ImagePositionPatient
input_img = np.array([el["img"] for el in sorted_list])
print(f'Input Image Initial Shape: {input_img.shape}')
if CREATE_IMAGES:
    create_rotation(minmax_normalize(input_img), n=N, name="pre_zoom", root=root, show=False, results_path=results_path, cmap='bone')
input_img = zoom(input_img, pixel_len_mm, order=3)
if CREATE_IMAGES:
    create_rotation(minmax_normalize(input_img), n=N, name="post_zoom", root=root, show=False, results_path=results_path, cmap='bone')
print(f'Input Image Zoomed Shape: {input_img.shape}')

# Normalize input image to have the same range of values as reference image
input_img = minmax_normalize(input_img)

# ###########################################################
# ------------------- Atlas image reading -------------------
# ###########################################################
print('----- Atlas Image Treatment -----')
atlas_path = os.path.join(root, "AAL3_1mm.dcm")
atlas_dcm = pydicom.dcmread(atlas_path)
atlas_img = atlas_dcm.pixel_array
print(f'Atlas Image Orientation Patient: {atlas_dcm["SharedFunctionalGroupsSequence"][0]["PlaneOrientationSequence"][0]["ImageOrientationPatient"].value}')
atlas_img = np.rot90(atlas_img, k=2, axes=(1, 2))
print(f'Atlas Image fliped! [1 0 0 0 1 0]')
print(f'Atlas Shape: {atlas_img.shape}')
# Atlas thalamus extraction
atlas_img = (atlas_img >= 121) & (atlas_img <= 150)

print('Padd Atlas to match Reference Image')
reference_img, atlas_img = expand_images(reference_img, atlas_img)
print(f'    Reference Image Shape: {reference_img.shape},\n    Atlas Image Shape: {atlas_img.shape}')

if CREATE_IMAGES:
    projections = create_projections(reference_img, n=N, mode="img")
    seg_projections = create_projections(atlas_img, n=N, mode="nonrealistic")
    
    colormapped_projections = [apply_colormap(projection, "bone") for projection in projections]
    colormapped_seg_projections = [apply_colormap(seg_projection, "tab10") for seg_projection in seg_projections]

    fused_projections = [alpha_fusion(proj, seg_proj, noncm_seg_proj) for proj, seg_proj, noncm_seg_proj in zip(colormapped_projections, colormapped_seg_projections, seg_projections)]

    # Save and visualize animation
    img_min = np.amin(reference_img)
    img_max = np.amax(reference_img)

    fig, ax = plt.subplots()
    plt.axis('off')

    animation_data = [
        [plt.imshow(img, animated=True, vmin=img_min, vmax=img_max)]
        for img in fused_projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                interval=0.390625*N, blit=True)
                                
    anim.save(os.path.join(results_path, f"thalamus_reference_rotation.gif"))
    # plt.show() 
    plt.close()

# ###########################################################
# ---------------- Image Dimension Matching ----------------
# ###########################################################
print('----- Image Dimension Matching -----')
print('Padd Reference Image and Atlas Image to match Input Image shape')
reference_img, input_img = expand_images_unilateral(reference_img, input_img, axis=0)
atlas_img, input_img = expand_images_unilateral(atlas_img, input_img, axis=0)

reference_img, input_img = expand_images(reference_img, input_img)
atlas_img, input_img = expand_images(atlas_img, input_img)
print(f'    Reference Image Shape: {reference_img.shape},\n    Atlas Image Shape: {atlas_img.shape},\n    Input Image Shape: {input_img.shape}')

if CREATE_IMAGES:
    create_rotation(reference_img, n=N, name="matching_reference", root=root, show=False, results_path=results_path, cmap='bone')
    create_rotation(input_img, n=N, name="matching_input", root=root, show=False, results_path=results_path, cmap='bone')
# create_rotation(atlas_img, n=N, name="matching_atlas", root=root, show=False, cmap='tab10', results_path=results_path)

# ###########################################################
# ----------------- Coregister Minimization -----------------
# ###########################################################
# Coregister minimization
## Discard the inferior part of the head, since we are going to coregister the upper one, and resize the image to accelerate the minimization
coregister_shape = (64, 64, 64)
reference_img_tomin = resize(reference_img[150:, :, :], output_shape=coregister_shape)
input_img_tomin = resize(input_img[150:, :, :], output_shape=coregister_shape)
if CREATE_IMAGES:
    create_rotation(reference_img_tomin, n=N, name="minimize_reference", root=root, show=False, results_path=results_path, cmap='bone')
    create_rotation(input_img_tomin, n=N, name="minimize_input", root=root, show=False, results_path=results_path, cmap='bone')

translation_scale = [reference_img[150:, :, :].shape[idx]/coregister_shape[idx] for idx in range(3)]
# atlas_img = resize(atlas_img, output_shape=coregister_shape)

print("Coregistering...")
if DO_COREGISTER:    
    results = coregister_images(reference_img_tomin, input_img_tomin, initial_parameters=(5, 0, 0, 10, 0, 0))
    results = results.x
    print("Coregister ended!")
    for idx in range(3):
        results[idx] *= translation_scale[idx]
    print(results)
else:
    results=np.array([81.34374028, -2.56417755, -2.00138013, -3.45796167,  0.94065069,  1.02547316])


# ###########################################################
# ------------------- Animation Creation -------------------
# ###########################################################
transformed_input_img = translation_then_axialrotation(input_img, parameters=results)
if CREATE_IMAGES:
    create_rotation(transformed_input_img, n=N, name="rotated_input", root=root, show=False, cmap="bone", results_path=results_path)
    create_rotation(reference_img, n=N, name="rotated_reference", root=root, show=False, cmap="bone", results_path=results_path)
    
    # Apply colormaps to input_image and segmentations
    cm_input = apply_colormap(transformed_input_img, colormap='bone')
    cm_ref = apply_colormap(reference_img, colormap='Reds')

    # Overlay seg to img using alpha fusion and store mid slices on each plane
    overlayed_img = alpha_fusion(cm_input, cm_ref, np.where(reference_img > 0.05, 1, 0), ALPHA=0.8)
    for idx, axis in enumerate(('axial', 'coronal', 'sagittal')):
        save_slice(overlayed_img, axis=axis, filename=os.path.join(results_path, f'coregistered_{axis}_slice.png'), pixel_len_mm=[1, 1, 1])

    # ---- Transformed atlas in the input space (axial animation) ----
    transformed_atlas = inverse_transformation(atlas_img, params=results, order=0)
    cm_transformed_input = apply_colormap(input_img, colormap="bone")
    cm_transformed_atlas = apply_colormap(transformed_atlas, colormap="tab10")

    overlayed = [alpha_fusion(cm_inp, cm_atlas, non_colormapped_seg=atlas) for cm_inp, cm_atlas, atlas in zip(cm_transformed_input, cm_transformed_atlas, transformed_atlas)]
    overlayed = overlayed[150:]
    # Save and visualize animation
    img_min = np.amin(cm_transformed_input)
    img_max = np.amax(cm_transformed_input)

    fig, ax = plt.subplots()
    plt.axis('off')

    animation_data = [
        [plt.imshow(img, animated=True, vmin=img_min, vmax=img_max)]
        for img in overlayed
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                interval=250, blit=True)

    anim.save(os.path.join(results_path, "atlas_in_input_space.gif"))
    plt.close()

    # ---- Transformed atlas in the input space (rotation animation) ----
    projections = create_projections(input_img, n=N, mode="img")
    seg_projections = create_projections(transformed_atlas, n=N, mode="nonrealistic")

    colormapped_projections = [apply_colormap(projection, "bone") for projection in projections]
    colormapped_seg_projections = [apply_colormap(seg_projection, "tab10") for seg_projection in seg_projections]

    fused_projections = [alpha_fusion(proj, seg_proj, noncm_seg_proj) for proj, seg_proj, noncm_seg_proj in zip(colormapped_projections, colormapped_seg_projections, seg_projections)]

    # Save and visualize animation
    img_min = np.amin(reference_img)
    img_max = np.amax(reference_img)

    fig, ax = plt.subplots()
    plt.axis('off')

    animation_data = [
        [plt.imshow(img, animated=True, vmin=img_min, vmax=img_max)]
        for img in fused_projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                interval=0.390625*N, blit=True)
                                
    anim.save(os.path.join(results_path, f"thalamus_input_rotation.gif"))
    # plt.show() 
    plt.close()
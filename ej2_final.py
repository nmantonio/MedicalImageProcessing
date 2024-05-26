from utils_final import *
from utils_final import minmax_normalize, inverse_transformation
from utils import create_rotation, alpha_fusion, create_projections, apply_colormap


import os
import pydicom
from scipy.ndimage import zoom
from skimage.transform import resize
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, animation



root = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing"
os.makedirs(os.path.join(root, "results", "2"), exist_ok=True)
N = 7


# Reference image reading
print("----- Reference Image Treatment -----")
reference_path = os.path.join(root, "icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm")
reference_dcm = pydicom.dcmread(reference_path)
reference_img = reference_dcm.pixel_array
reference_img = minmax_normalize(reference_img)
print(f'Reference Image Orientation Patient: {reference_dcm["SharedFunctionalGroupsSequence"][0]["PlaneOrientationSequence"][0]["ImageOrientationPatient"].value}')
# reference_img = np.flipud(np.fliplr(reference_img))
# reference_img = np.flip(np.flip(reference_img, axis=1), axis=2)
reference_img = np.rot90(reference_img, k=2, axes=(1, 2))
print(f'Reference Image fliped!')
print(f'Reference Image Shape: {reference_img.shape}')
# create_rotation(reference_img, n=N, name="reference", root=root, show=False)


# Input image reading
print("----- Input Image Treatment -----")
input_path = os.path.join(root, "RM_Brain_3D-SPGR")
unsorted_input_info = []
for idx, slice_path in enumerate(os.listdir(input_path)):
    dcm_img = pydicom.dcmread(os.path.join(input_path, slice_path))
    if idx == 0: 
        print(f'Input Image Orientation Patient: {dcm_img["ImageOrientationPatient"].value}')
        pixel_len_mm = [float(dcm_img["SliceThickness"].value), float(dcm_img["PixelSpacing"][0]), float(dcm_img["PixelSpacing"][1])]
        print(f'Input Pixel Length (mm): {pixel_len_mm}')

    unsorted_input_info.append({"img": dcm_img.pixel_array, "pos": dcm_img["ImagePositionPatient"].value[2]})

sorted_list = sorted(unsorted_input_info, key=lambda x: x["pos"], reverse=False)
input_img = np.array([el["img"] for el in sorted_list])
print(f'Input Image Initial Shape: {input_img.shape}')
# create_rotation(minmax_normalize(input_img), n=N, name="pre_zoom", root=root, show=False)
input_img = zoom(input_img, pixel_len_mm, order=3)
# create_rotation(minmax_normalize(input_img), n=N, name="post_zoom", root=root, show=False)
print(f'Input Image Zoomed Shape: {input_img.shape}')

# Normalize input image to have the same range as reference image
input_img = minmax_normalize(input_img)

# Atlas image reading
print('----- Atlas Image Treatment -----')
atlas_path = os.path.join(root, "AAL3_1mm.dcm")
atlas_dcm = pydicom.dcmread(atlas_path)
atlas_img = atlas_dcm.pixel_array
print(f'Atlas Image Orientation Patient: {atlas_dcm["SharedFunctionalGroupsSequence"][0]["PlaneOrientationSequence"][0]["ImageOrientationPatient"].value}')
atlas_img = np.rot90(atlas_img, k=2, axes=(1, 2))
print(f'Atlas Image fliped!')
print(f'Atlas Shape: {atlas_img.shape}')
# Atlas thalamus straction
atlas_img = (atlas_img >= 121) & (atlas_img <= 150)

# Matching shapes treatment
print('----- Shapes Preprocessing -----')
print('Padd Atlas to match Reference Image')
reference_img, atlas_img = expand_images(reference_img, atlas_img)
print(f'    Reference Image Shape: {reference_img.shape},\n    Atlas Image Shape: {atlas_img.shape}')

print('Padd Reference Image and Atlas Image to match Input Image shape')
reference_img, input_img = expand_images_unilateral(reference_img, input_img, axis=0)
atlas_img, input_img = expand_images_unilateral(atlas_img, input_img, axis=0)

reference_img, input_img = expand_images(reference_img, input_img)
atlas_img, input_img = expand_images(atlas_img, input_img)

# create_rotation(reference_img, n=N, name="matching_reference", root=root, show=False)
# create_rotation(input_img, n=N, name="matching_input", root=root, show=False)
# create_rotation(atlas_img, n=N, name="matching_atlas", root=root, show=False, cmap='tab10')


# Coregister minimization
## Discard the inferior part of the head, since we are going to coregister the upper one, and resize the image to accelerate the minimization
coregister_shape = (input_img.shape[0], 50, 50)
reference_img = resize(reference_img, output_shape=coregister_shape)[200:]
input_img = resize(input_img, output_shape=coregister_shape)[200:]
atlas_img = resize(atlas_img, output_shape=coregister_shape)[200:]
print("Coregistering...")


results = coregister_images(reference_img, input_img, initial_parameters=(0, 0, 0, 0, 0, 0))
# print("Coregister ended!")
# print(results)
# print(results.x)
# results = results.x
# results=np.array([33.9999983,  -3.48060517, -2.51845272,  8.96974994, -0.69243038,  1.11763455])

# transformed_input_img = translation_then_axialrotation(input_img, parameters=results)
# # create_rotation(rotated_input_img, n=N, name="rotated_input", root=root, show=False, cmap="bone")
# # create_rotation(reference_img, n=N, name="rotated_reference", root=root, show=False, cmap="bone")

# inp_proj = [apply_colormap(proj, colormap="bone") for proj in create_projections(rotated_input_img, n=N, mode="img")]
# ref_proj = [apply_colormap(proj, colormap="Reds") for proj in create_projections(reference_img, n=N, mode="img")]
# seg = np.where(np.array(ref_proj) > 0.5, 1, 0)

# fused_images = [alpha_fusion(rot_inp, ref, seg_slice) for ref, rot_inp, seg_slice in zip(inp_proj, ref_proj, seg)]
# fig, ax = plt.subplots()
# plt.axis('off')
# plt.imshow(ref_proj[0])
# plt.imshow(seg[0])
# plt.show()
# plt.imshow(inp_proj[0])
# plt.imshow(fused_images[0])
# plt.show()

# Save and visualize animation
# from matplotlib import pyplot as plt, animation

# img_min = np.amin(input_img)
# img_max = np.amax(input_img)

# fig, ax = plt.subplots()
# plt.axis('off')

# animation_data = [
#     [plt.imshow(img, animated=True, vmin=0, vmax=1)]
#     for img in fused_images
# ]
# anim = animation.ArtistAnimation(fig, animation_data,
#                             interval=0.390625*N, blit=True)
# anim.save(os.path.join(root, "results", "2", "overlayed.gif"))

# transformed_atlas = inverse_transformation(atlas_img, params=results)
cm_transformed_input = apply_colormap(reference_img, colormap="bone")
cm_transformed_atlas = apply_colormap(atlas_img, colormap="tab10")

overlayed = [alpha_fusion(cm_inp, cm_atlas, non_colormapped_seg=atlas) for cm_inp, cm_atlas, atlas in zip(cm_transformed_input, cm_transformed_atlas, atlas_img)]

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

plt.show()        




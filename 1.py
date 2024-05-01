import os
import pydicom
import numpy as np

segmentation_path = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing\manifest-1714499547568\HCC-TACE-Seg\HCC_009\02-15-1998-NA-PP CAPLIVER PROTO.-10975\300.000000-Segmentation-91221\1-1.dcm"
med_folder = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing\manifest-1714499547568\HCC-TACE-Seg\HCC_009\02-15-1998-NA-PP CAPLIVER PROTO.-10975\103.000000-LIVER 3 PHASE CAP-83135"

seg_dcm = pydicom.dcmread(segmentation_path) 
seg_imgpos = np.array([])
for elem in seg_dcm["PerFrameFunctionalGroupsSequence"]:
    print(elem["SegmentIdentificationSequence"].value)
    # seg_imgpos = np.append(seg_imgpos, elem["PlanePositionSequence"][0]["ImagePositionPatient"].value[2])
    # print(elem["PlanePositionSequence"][0]["ImagePositionPatient"].value)
# print(seg_imgpos)
# seg_imgpos = np.reshape(seg_imgpos, (4, 69))   
# seg = seg_dcm.pixel_array
# seg = np.reshape(seg, (4, 69, 512, 512))

# unsorted_image_info = []
# for idx, slice_path in enumerate(os.listdir(med_folder)):
#     dcm_img = pydicom.dcmread(os.path.join(med_folder, slice_path))
#     unsorted_image_info.append({"img": dcm_img.pixel_array, "pos": dcm_img["ImagePositionPatient"].value[2]})
    
# sorted_list = sorted(unsorted_image_info, key=lambda x: x["pos"])
# img = np.array([el["img"] for el in sorted_list])
# print(img.shape)

# for img_info, seg_pos in zip(unsorted_image_info, seg_imgpos):
#     if not (all(img_info["pos"] == seg_pos_value for seg_pos_value in seg_pos)):
#         print(img_info["pos"], seg_pos)

    # for elem in dcm_img: 
    #     print(elem)
    # for elem in dcm_img["PerFrameFunctionalGroupsSequence"]:
    #     print(elem["PlanePositionSequence"][0]["ImagePositionPatient"].value)
#     img[idx] = dcm_img.pixel_array
    
# print(img.shape)

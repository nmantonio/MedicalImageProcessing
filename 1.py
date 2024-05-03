import os
import pydicom
import numpy as np
import scipy

med_folder = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing\manifest-1714499547568\HCC-TACE-Seg\HCC_009\02-15-1998-NA-PP CAPLIVER PROTO.-10975\103.000000-LIVER 3 PHASE CAP-83135"
segmentation_path = r"C:\Users\tonin\Desktop\Master\PIM\MedicalImageProcessing\manifest-1714499547568\HCC-TACE-Seg\HCC_009\02-15-1998-NA-PP CAPLIVER PROTO.-10975\300.000000-Segmentation-91221\1-1.dcm"

# for idx, slice_path in enumerate(os.listdir(med_folder)[::-1]):
    # dcm_img = pydicom.dcmread(os.path.join(med_folder, slice_path))
    # print(dcm_img)
    # print(dcm_img["AcquisitionNumber"])
    # break
    

seg_dcm = pydicom.dcmread(segmentation_path) 
# seg_groups = np.unique([elem["SegmentIdentificationSequence"][0]["ReferencedSegmentNumber"].value for elem in seg_dcm["PerFrameFunctionalGroupsSequence"]])
# print(dir(seg_dcm))
print(dir(seg_dcm["SegmentSequence"].value[0]))
for segseq in seg_dcm["SegmentSequence"].value:
    print(segseq["SegmentDescription"])
# import time
# for key in dir(seg_dcm): 
#     try: 
#         print(key, ":", seg_dcm[key])   
#         time.sleep(2)
#     except: 
#         pass
# print([el["ReferencedSegmentNumber"].value for el in seg_dcm["PerFrameFunctionalGroupsSequence"]])
# for elem in seg_dcm["PerFrameFunctionalGroupsSequence"]:
#     print()
# print(n_segmented_objects)
# print(seg_dcm[""])
# print(seg_dcm)
# print("SEG: ", seg_dcm["AcquisitionNumber"])
seg = {}

#     break
#     print(elem["SliceIndex"])
    # print(elem["SegmentIdentificationSequence"][0]["ReferencedSegmentNumber"].value)
    # seg_imgpos = np.append(seg_imgpos, elem["PlanePositionSequence"][0]["ImagePositionPatient"].value[2])
    # print(elem["PlanePositionSequence"][0]["ImagePositionPatient"].value)
# print(seg_imgpos)
# seg_imgpos = np.reshape(seg_imgpos, (4, 69))   
# seg = seg_dcm.pixel_array
# seg = np.reshape(seg, (4, 69, 512, 512))


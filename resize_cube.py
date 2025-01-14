import spectral.io.envi as envi
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat

datapath = Path(r'HS_Data/Scream/SWIR/')

header_file = str(datapath / 'MM.M00514_cube_full.hdr')
spectral_file = str(datapath / 'MM.M00514_cube_full')


# header_file = str("C:\\Users\peter\Desktop\Work\PERCEIVE\Python\AVS\HS_Data\Mock-ups HSI\Mock-ups HSI\Varnish_Squares_VNIR_1800_SN00841_HSNR3_2150us_2023-10-10T130253_raw_rad_float32.hdr")
# spectral_file = str("C:\\Users\peter\Desktop\Work\PERCEIVE\Python\AVS\HS_Data\Mock-ups HSI\Mock-ups HSI\Varnish_Squares_VNIR_1800_SN00841_HSNR3_2150us_2023-10-10T130253_raw_rad_float32.img")
#
#
# header_file = str("C:\\Users\peter\Desktop\Work\PERCEIVE\Python\AVS\HS_Data\Mock-ups HSI\Mock-ups HSI\Varnish_stripes_VNIR_1800_SN00841_HSNR3_2150us_2023-10-10T125741_raw_rad_float32.hdr")
# spectral_file = str("C:\\Users\peter\Desktop\Work\PERCEIVE\Python\AVS\HS_Data\Mock-ups HSI\Mock-ups HSI\Varnish_stripes_VNIR_1800_SN00841_HSNR3_2150us_2023-10-10T125741_raw_rad_float32.img")
#

#header_file = str("C:\\Users\peter\Downloads\G193-03-scream-1_refl.hdr")
#spectral_file = str("C:\\Users\peter\Downloads\G193-03-scream-1_refl.dat")

#header_file = str("C:\\Users\peter\Downloads\M00514-SWIR mosaic-resize.hdr")
#spectral_file = str("C:\\Users\peter\Downloads\M00514-mosaic-resize")

numpy_ndarr = envi.open(header_file, spectral_file)

hypersp_data = np.zeros((625, 500, numpy_ndarr.nbands))

#RESIZE CUBE
for i in range(0,numpy_ndarr.nbands):
    img_slice = (numpy_ndarr.read_bands([i]))

    im_slice_resized = img_slice
    #cv2.imwrite('export_slice.png',np.uint8(im_slice_resized*255))
    hypersp_data[:,:,i] = cv2.resize(np.squeeze(im_slice_resized), (500,625))

    #cv2.imwrite('export_slice.png', np.uint8(im_slice_resized * 255))

savemat('hyperspectral_image_reduced.mat', {'hyperspectral_image_reduced': hypersp_data})

#envi.save_image('hyperspectral_image_reduced.hdr', hypersp_data, interleave='bil', force=True)

#CHECK REDUCED SLICE

slice_check=100
img_slice = (numpy_ndarr.read_bands([slice_check]))
cv2.imwrite('export_slice_orig.png',np.uint8(img_slice*255))

header_file = str('hyperspectral_image_reduced_625_500.hdr')
spectral_file = str('hyperspectral_image_reduced_625_500.img')

numpy_ndarr = envi.open(header_file, spectral_file)
img_slice = (numpy_ndarr.read_bands([slice_check]))
cv2.imwrite('export_slice_reduced.png',np.uint8(img_slice*255))




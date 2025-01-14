import spectral.io.envi as envi
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat

datapath = Path(r'HS_Data/')

header_file = str(datapath / 'MM.M00514_cube_full.hdr')
spectral_file = str(datapath / 'MM.M00514_cube_full')

header_file = str(datapath / 'M00514-SWIR mosaic-resize.hdr')
spectral_file = str(datapath / 'M00514-mosaic-resize')

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

#OPTION 1: EXTRACT ALL SLICES AS BW IMAGES FROM SINGLE WAVELENGTHS

for i in range(len(numpy_ndarr.bands.centers)):
    img = numpy_ndarr.read_bands([i, i, i])  # select the bands
    print(img.shape)
    cv2.imwrite('./OUTPUT/%.03f.png'%(numpy_ndarr.bands.centers[i]),np.uint8(img*255))


#OPTION 2:  EXTRACT FAKE RGB IMAGE WITH SINGLE WAVELENGTHS FOR R,G,B, CHANNELS

#img = numpy_ndarr.read_bands([10, 60, 110])  # select the bands
#print(img.shape)
#cv2.imwrite('export.png',np.uint8(img*255))

#OPTION 3: EXTRACT FAKE RGB IMAGE WITH INTEGRATED WAVELENGTH RANGES FOR R,G,B, CHANNELS

img_combR = np.zeros_like((numpy_ndarr.read_bands([0])))
for i in range(0,10):
    img_new = (numpy_ndarr.read_bands([i]))
    img_diff = img_new - img_combR
    condition = img_diff > 0
    img_combR[condition] = img_new[condition]

img_combG = np.zeros_like((numpy_ndarr.read_bands([0])))
for i in range(60,70):
    img_new = (numpy_ndarr.read_bands([i]))
    img_diff = img_new - img_combG
    condition = img_diff > 0
    img_combG[condition] = img_new[condition]

img_combB = np.zeros_like((numpy_ndarr.read_bands([0])))
for i in range(100,115):
    img_new = (numpy_ndarr.read_bands([i]))
    img_diff = img_new - img_combB
    condition = img_diff > 0
    img_combB[condition] = img_new[condition]

comb_array = np.concatenate((img_combR,img_combG,img_combB),2)

#cv2.imshow('rgb',cv2.resize(img,(0, 0), fx = 0.45, fy = 0.45))
#k = cv2.waitKey(0);

cv2.imwrite('export.png',np.uint8(comb_array*255))

#EXTRACT SPECTRA

#CREATE TABLE FROM AREAS

no_of_areas=1

regiontable = np.zeros([no_of_areas,4])


# #REGION 4
# regiontable[0,:] = [533,494,23,15]
# regiontable[1,:] = [1605,474,53,53]


# #REGION 5
# regiontable[0,:] = [823, 621, 11, 11]
# regiontable[1,:] = [1110, 570, 4, 5]
# regiontable[2,:] = [527, 596, 27, 6]
# regiontable[3,:] = [469, 593, 40, 9]


# #REGION 6
# regiontable[0,:] = [1147, 369 , 24 ,16]
# regiontable[1,:] = [1198,358,65,13]
# regiontable[2,:] = [237,834,27,27]


# #REGION 7
# regiontable[0,:] = [224,729,123,14]

#REGION 8
regiontable[0,:] =[1529,615,54,67]


no_of_pixels = 0
for f in range(no_of_areas):
    no_of_pixels = no_of_pixels + regiontable[f,2]*regiontable[f,3]

trace_image = np.zeros([int(no_of_pixels), numpy_ndarr.nbands])

k=0
for f in range(no_of_areas):

    x1 = int(regiontable[f,0])
    y1 = int(regiontable[f,1])

    x2 = int(x1 + regiontable[f,2])
    y2 = int(y1 + regiontable[f,3])

     #= (x2 - x1) * (y2 - y1)

    for i in range(x1,x2):
        for j in range(y1,y2):
            trace_image[k,:] = np.squeeze(numpy_ndarr[j,i,:]) # Y , X OF IMAGE
            k=k+1



mdic = {"spectra8": trace_image}
savemat("spectra_img8.mat",mdic)

# #CREATE TABLE FROM POITNS
# X = [206,54, 1490, 1618, 443, 623, 1047, 1143]
# Y = [1068,170, 1466, 493, 1668, 503, 594, 372]
#
# no_of_pixels = len(X)
# trace_image = np.zeros([no_of_pixels,numpy_ndarr.nbands])
#
# k=0
# for i in range(len(X)):
#     trace_image[k,:] = np.squeeze(numpy_ndarr[Y[i],X[i],:]) # Y , X OF IMAGE
#     k=k+1
# mdic = {"trace_image": trace_image}
# savemat("spectra_img.mat",mdic)

#np.savetxt("spectra_img.txt",trace_image)


density = 1

c = numpy_ndarr.ncols
r = numpy_ndarr.nrows

cpointno = 500
rpointno = 625


trace = np.zeros([cpointno*rpointno,numpy_ndarr.nbands])
#
j=1
i=1
for p in range(cpointno*rpointno-1):

      if i>cpointno:
        j=j+1
        i = 1







      rowpixel = (1 + j) * r/(rpointno+1) - r/(rpointno+1)
      colpixel = (1 + i) * c/(cpointno+1) - c/(cpointno+1)


      #print(p)
      #
      # print(j)
      # print(i)
      #
      #print(int(np.floor(rowpixel)))
      #print(int(np.floor(colpixel)))


      trace[p,:] = np.squeeze(numpy_ndarr[int(np.floor(rowpixel)),int(np.floor(colpixel)),:]) # Y , X OF IMAGE
#     #plt.plot(trace[i,:])
#     #plt.show()
#     #plt.xlabel('Wavelengths')
#     #plt.ylabel('Intensity')
#     #plt.savefig('spectrum.png')
      i = i+1

print("total rows %d" , r)
print("total columns %d", c)

mdic = {"painting": trace}
savemat("./OUTPUT/spectra_painting.mat",mdic)

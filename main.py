#%% 
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as im
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#import cv2 as cv
from os.path import join
import reconlib as rl
import csv

# ---------- NALOZI SLIKE IZ MAPE ----------
# pth = '/home/martin/Desktop/RV_Seminar_v2/rekonstrukcija_minimal'
# pth = 'C://Users//lapaj//oneDrive//RV_Seminar_v2//rekonstrukcija_minimal'
pth = 'C:/Users/Martin/Desktop/RV_Seminar_v2/rekonstrukcija_minimal'

calibration_image_fname = join(pth, 'calibration', 'kalibr.jpg')
calibration_data_fname = join(pth, 'calibration', 'tocke_kalibra.npy')

#source
# acquisition_data_pth_ref = join(pth, 'acquisitions', '0stopinj')
acquisition_data_pth = join(pth, 'acquisitions', '135stopinj')
# acquisition_data_pth = join(pth, 'acquisitions', '45stopinj')
# acquisition_data_pth = join(pth, 'acquisitions', '90stopinj')
# acquisition_data_pth = join(pth, 'acquisitions', '135stopinj')
# acquisition_data_pth = join(pth, 'acquisitions', '180stopinj')
# acquisition_data_pth = join(pth, 'acquisitions', '225stopinj')
# acquisition_data_pth = join(pth, 'acquisitions', '315stopinj')
acquisition_data_pth_ref = join(pth, 'acquisitions', '0stopinj_ref90')
# acquisition_data_pth = join(pth, 'acquisitions', 'slusalke90')

#results
# out_volume_fname_ref = join(pth, 'reconstructions', '0stopinj.nrrd')
out_volume_fname = join(pth, 'reconstructions', '135stopinj.nrrd')
# out_volume_fname = join(pth, 'reconstructions', '45stopinj.nrrd')
# out_volume_fname = join(pth, 'reconstructions', '90stopinj.nrrd')
# out_volume_fname = join(pth, 'reconstructions', '135stopinj.nrrd')
# out_volume_fname = join(pth, 'reconstructions', '180stopinj.nrrd')
# out_volume_fname = join(pth, 'reconstructions', '225stopinj.nrrd')
# out_volume_fname = join(pth, 'reconstructions', '315stopinj.nrrd')
out_volume_fname_ref = join(pth, 'reconstructions', '0stopinj_ref90.nrrd')
# out_volume_fname = join(pth, 'reconstructions', 'slusalke90.nrrd')

slike_ref, koti_ref = rl.load_images(acquisition_data_pth_ref, proc=rl.rgb2gray)

# # obrezovanje
# ce potrebujes -> (ni se preverjena)funkcija crop_image

# rl.showImage(slike[10])
# plt.show()

# ---------- DOLOCI 3D KOORDINATE TOCK NA KALIBRU ----------
pts3d = rl.IRCT_CALIBRATION_OBJECT()
# plt.close('all')
# r3d.show_points_in_3d(pts3d)


# ---------- OZNACI 8 TOCK NA KALIBRU, KI NAJ OZNACUJEJO SREDISCE KROGEL ----------
if not os.path.exists(calibration_data_fname):
    #order of centers selecting is important!
    calibration_image = np.array(im.open(calibration_image_fname))
    pts2d =  rl.annotate_caliber_image(calibration_image, calibration_data_fname, n=8)

    plt.close('all')
    pts2d = np.load(calibration_data_fname)[0]
    rl.showImage(slike_ref[0], iTitle='Oznacena sredisca krogel na kalibru.')
    plt.plot(pts2d[:,0], pts2d[:,1],'mx',markersize=15)

pts2d = np.load(calibration_data_fname)[0]

# ---------- KALIBRIRAJ SISTEM ZA ZAJEM SLIK ----------
Tproj, pts3dproj = rl.calibrate_irct(pts2d, pts3d) #Tproj- incldes: Tproj and Ttable!

# plt.close('all')
# imlib.showImage(slike[0], iTitle='Oznacena sredisca krogel na kalibru.')
# plt.plot(pts2d[:,0], pts2d[:,1],'rx', markersize=15)
# plt.plot(pts3dproj[:,0], pts3dproj[:,1],'gx', markersize=15)


# ---------- FILTRIRANJE 2D SLIK PRED POVRATNO PROJEKCIJO ----------
slika = np.squeeze(slike_ref[0])
#tip_filtra = 'hann'  # none, ram-lak, cosine, hann, hamming
tip_filtra = 'hann'
slika_f = rl.filter_projection(slika, tip_filtra, cut_off=0.9)
# rl.showImage(slika_f, iCmap=cm.jet)
# plt.show()

# ---------- REKONSTRUKCIJA 3D SLIKE (REFERENČNA)----------
# FBP = Filtered BackProjection
vol_ref = rl.fbp(slike_ref[::1], koti_ref[::1], Tproj,
              filter_type='hann', sampling_mm=3,
              out_fname=out_volume_fname_ref, cut_off=0.75)


pointCoorX_ref, pointCoorY_ref, pointCoorZ_ref = rl.get_point_cloud(vol_ref, 0.5, 1, 0.1, 0.9, 50)

# ---------- REKONSTRUKCIJA 3D SLIKE (IZBRANA)----------
slike, koti = rl.load_images(acquisition_data_pth, proc=rl.rgb2gray)

# FBP = Filtered BackProjection
vol = rl.fbp(slike[::1], koti[::1], Tproj,
              filter_type='ram-lak', sampling_mm=3,
              out_fname=out_volume_fname, cut_off=0.75)

pointCoorX, pointCoorY, pointCoorZ = rl.get_point_cloud(vol, 0.5, 1, 0.1, 0.9, 50)


# ---------- IZRIS POINT CLOUD ----------
rl.plot_point_cloud(pointCoorX_ref, pointCoorY_ref, pointCoorZ_ref)
rl.plot_point_cloud(pointCoorX, pointCoorY, pointCoorZ)
plt.show()


#%% 
#------------- PORAVNAVA TEST------------------
# modelCoor = [pointCoorX_ref,pointCoorY_ref,pointCoorZ_ref]
# dataCoor = [pointCoorX,pointCoorY,pointCoorZ]
# model_in, data_in = rl.prepare_sets(modelCoor, dataCoor)

# #transform both point cloud to center of coordinate system
# Mat_trans = rl.transAffine3D(iTrans = (-59,-59,0))
# model_in = np.dot(model_in, Mat_trans.transpose())
# data_in = np.dot(data_in, Mat_trans.transpose())
# rl.visualize(data_in, model_in)

# #align both data sets
# register_points_icp_best, angleZ_aprox = rl.transform_data(model_in, data_in, 10)
# rl.visualize(register_points_icp_best, model_in)

# print(angleZ_aprox)
#%% 
#----------------- IZRIS GRAF NAPAKE PORAVNAVE ---------------------------

files_endings = ['45stopinj', '90stopinj', '135stopinj', '180stopinj', '225stopinj', '315stopinj']
angle_appx_list = []
for ends in range(len(files_endings)):
    acquisition_data_pth = join(pth, 'acquisitions', files_endings[ends])
    slike, koti = rl.load_images(acquisition_data_pth, proc=rl.rgb2gray)
    vol = rl.fbp(slike[::1], koti[::1], Tproj,
                filter_type='hann', sampling_mm=3,
                out_fname=out_volume_fname, cut_off=0.75)

    pointCoorX, pointCoorY, pointCoorZ = rl.get_point_cloud(vol, 0.5, 1, 0.1, 0.9, 50)

    modelCoor = [pointCoorX_ref,pointCoorY_ref,pointCoorZ_ref]
    dataCoor = [pointCoorX,pointCoorY,pointCoorZ]
    model_in, data_in = rl.prepare_sets(modelCoor, dataCoor)

    #transform both point cloud to center of coordinate system
    Mat_trans = rl.transAffine3D(iTrans = (-59,-59,0))
    model_in = np.dot(model_in, Mat_trans.transpose())
    data_in = np.dot(data_in, Mat_trans.transpose())

    #align both data sets
    register_points_icp_best, angleZ_aprox = rl.transform_data(model_in, data_in, 10)
    angle_appx_list.append(angleZ_aprox)

x = np.arange(1,len(angle_appx_list)+1)
angle_ref = [45, 90, 135, 180, 225, 315]
plt.plot( x, angle_ref,'.--',  color='red', linewidth=1, label="Referenčne")
plt.plot( x, angle_appx_list, '.--', color='blue', linewidth=1, label="Dejanske")
plt.legend()
plt.xlabel('Številka oblaka točk')
plt.ylabel('kot [°]')
plt.show()

#%% 
#--------------------IZLOČITEV PORŠINSKIH TOČK MERJENCA TER CSV IZVOZ-----------------
surfPointCoorX, surfPointCoorY, surfPointCoorZ = rl.get_point_cloud_surface(vol_ref, 0.5, 1, 0.25, 0.9, 45)

rl.plot_point_cloud(surfPointCoorX, surfPointCoorY, surfPointCoorZ)
surfPoints = [surfPointCoorX, surfPointCoorY, surfPointCoorZ]

with open('ptCloud.csv','wb') as f:
    out = csv.writer(f, delimiter=',')
    out.writerows(zip(*surfPoints))
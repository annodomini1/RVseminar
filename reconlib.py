import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
import SimpleITK as itk
import nrrd
import matplotlib.cm as cm
import PIL.Image as im
from os.path import join
import scipy.ndimage as ni
#import cv2 as cv
from sklearn.neighbors import NearestNeighbors
import sys

def rgb2gray(img):
    '''
    Convert RGB image to grayscale.

    :param img: Input RGB image. 
    :return: Grayscale image.
    '''
    if img.ndim == 3:
        return np.mean(img.astype('float'), axis=-1)
    return img

def load_images(pth, proc=rgb2gray, ext='.jpg'):
    imgs, angles = [], []
    for file in os.listdir(pth):
        if file.endswith(ext):
            try:
                print('Loading file "{}"...'.format(file))
                angles.append(float(file.replace(ext, '').split('_')[-1]))
                imgs.append(proc(np.array(im.open(join(pth, file)))))
            except ValueError:
                print('Error while reading file "{}", skipping.'.format(file))
    idx = np.argsort(angles)
    imgs = [imgs[i] for i in idx]
    angles = [angles[i] for i in idx]
    return imgs, angles

#kalibracija
# IRCT CALIBRATION DATA
# Use a metering device to measure the geometric setup of the infrared CT system, namely the distances from the camera
# to the rotating table base.
TABLE_HEIGHT_MM = 79 # rotation table height in mm
CAMERA_HEIGHT_MM = 310 # distance in mm, in direction perpedicular to base, i.e. vertical axis
#CAMERA_TO_TABLE_DX_MM = 530 # distance in mm, in direction from camera towards the rotating table
CAMERA_TO_TABLE_DX_MM = 440
CAMERA_TO_TABLE_DY_MM = 10 # distance in mm, direction based on rhs coordinate frame (dx x dz), i.e. to the left w.r.t. dx

ROTATION_DIRECTION = -1# +1 corresponds to rotation direction about camera z axis, -1 is the opposite direction

# The volume is rectangular and camera-axis aligned. It is defined w.r.t. reference point on top of the caliber,
# which is determined by imaging the calibration object.
VOLUME_LENGTH = 350# x size
VOLUME_WIDTH = 350# y size
VOLUME_HEIGHT = 300# z size


def IRCT_CALIBRATION_OBJECT():
    """
    Marker locations on the IRCT calibration object based on manual measurements. The reference point should be
    at the center of rotating table base, i.e. all marker coordinate are to be defined w.r.t. to the center.

    :return: numpy array of 3D point coordinates corresponding to marker location on the calibration object
    """
    #h = 8
    h = 90
    #x kaze v smeri od kalibracijskega objekta do kamere
    #z kaze navzgor
    #y kaze tako, da imamo desnosucni KS
    pts = [
        [0,0,134+h], #[x,y,z]
        [33,0,108+h],
        [-16.5,-28,92+h],
        [-16.5,28,75+h],
        [33,0,60.5+h],
        [-16.5,-28,43+h],
        [-16.5,28,28+h],
        [33,0,12+h]
    ]
    return np.array(pts)

# TRACKER CALIBRATION DATA
CHECKERBOARD_SQUARE_SIZE_MM = 25.4

def showImage(iImage, iTitle='', iCmap=cm.Greys_r):
    '''
    Prikaze sliko iImage in jo naslovi z iTitle

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika 
    iTitle : str 
        Naslov za sliko
    iCmap : colormap 
        Barvna lestica za prikaz sivinske slike        

    Returns
    ---------
    None

    '''
    plt.figure()  # odpri novo prikazno okno

    # if iImage.ndim == 3:
    #     iImage = np.transpose(iImage, [1, 2, 0])

    plt.imshow(iImage, cmap=iCmap)  # prikazi sliko v novem oknu
    plt.suptitle(iTitle)  # nastavi naslov slike
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axes().set_aspect('equal', 'datalim')  # konstantno razmerje pri povecevanju, zmanjsevanju slike
    #plt.show()

def annotate_caliber_image(img, filename, n=8):
    pts_ann = []
    plt.close('all')
    showImage(img, iTitle='Oznaci sredisca krogel na sliki!')
    pts_ann.append(plt.ginput(n, timeout=-1))
    np.save(filename, pts_ann)

def transRigid3D(trans=(0, 0, 0), rot=(0, 0, 0)):
    '''
    Rigid body transformation

    :param trans: Translation 3-vector (tx,ty,tz).
    :param rot: Rotation vector (rx,ry,rz).
    :return: Rigid-body 4x4 transformation matrix.
    '''
    Trotx = np.array(((1, 0, 0, 0),
                      (0, np.cos(rot[0]), -np.sin(rot[0]), 0),
                      (0, np.sin(rot[0]), np.cos(rot[0]), 0),
                      (0, 0, 0, 1)))
    Troty = np.array(((np.cos(rot[1]), 0, np.sin(rot[1]), 0),
                      (0, 1, 0, 0),
                      (-np.sin(rot[1]), 0, np.cos(rot[1]), 0),
                      (0, 0, 0, 1)))
    Trotz = np.array(((np.cos(rot[2]), -np.sin(rot[2]), 0, 0),
                      (np.sin(rot[2]), np.cos(rot[2]), 0, 0),
                      (0, 0, 1, 0),
                      (0, 0, 0, 1)))
    Ttrans = np.array(((1, 0, 0, trans[0]),
                       (0, 1, 0, trans[1]),
                       (0, 0, 1, trans[2]),
                       (0, 0, 0, 1)))
    return np.dot(np.dot(np.dot(Trotx, Troty), Trotz), Ttrans)

def dlt_calibration(pts2d, pts3d):
    '''
    Perform DLT camera calibration based on input points

    :param pts2d: Nx2 array of (x,y) coordinates.
    :param pts3d: Nx3 array of (x,y,z) coordinates.
    :return: Transformation matrix
    '''

    def get_mat_row(pt2d, pt3d): #vhod v funkcijo je posamezna tocka v koordinatah (x,y,z)
        row1 = np.array((pt3d[0], pt3d[1], pt3d[2], 1, 0, 0, 0, 0, -pt3d[0]*pt2d[0], -pt3d[1]*pt2d[0], -pt3d[2]*pt2d[0]))
        row2 = np.array((0, 0, 0, 0, pt3d[0], pt3d[1], pt3d[2], 1, -pt3d[0]*pt2d[1], -pt3d[1]*pt2d[1], -pt3d[2]*pt2d[1]))
        return np.vstack((row1, row2)), np.vstack((pt2d[0], pt2d[1]))

    dmat = np.zeros((0, 11))
    dvec = np.zeros((0, 1))
    for i in range(pts2d.shape[0]):
        # print(dmat.shape)
        # print(get_mat_row(ptsW[i,:], ptsU[i,:]).shape)
        dmatp, dvecp = get_mat_row(pts2d[i, :], pts3d[i, :]) #notri da korespondencni tocki
        dmat = np.vstack((dmat, dmatp)) #ubistu appendamo v matriko matrike dmatp
        dvec = np.vstack((dvec, dvecp))
    return dmat, dvec

def calibrate_irct(pts2d, pts3d):
    '''
    Geometrically calibrate the IRCT system

    :param pts2d: Nx2 array of (x,y) coordinates.
    :param pts3d: Nx3 array of (x,y,z) coordinates.
    :return: 
    '''
    # setup transformations
    cam2table = [
        CAMERA_TO_TABLE_DX_MM,
        CAMERA_TO_TABLE_DY_MM,
        -CAMERA_HEIGHT_MM+TABLE_HEIGHT_MM
    ]
    Ttable = transRigid3D(trans=cam2table)

    # position points in space
    # pts3dcam = np.array((0,0,0,1))
    pts3dh = np.hstack((pts3d, np.ones((np.size(pts3d, 0), 1))))
    pts3dht = np.dot(np.dot(Ttable, transRigid3D(rot=(0, 0, 0))), np.transpose(pts3dh))
    pts3dht = np.transpose(pts3dht)

    # sensor_size = 25.4/4 # 1/4" is the sensor size
    # pixel_size = (25.4/4) / np.sqrt(float(imgs[0].shape[0]**2 + imgs[0].shape[1]**2))

    # perform dlt calibration
    A, b = dlt_calibration(pts2d, pts3dht)
    D = np.linalg.lstsq(A, b)

    Tproj = np.reshape(np.vstack((D[0], 1.0)), (3, 4))

    ptsXYZproj = np.dot(pts3dht, Tproj.transpose())
    # to homogeneous coordinates
    ptsXYZproj[:, 0] = ptsXYZproj[:, 0] / ptsXYZproj[:, 2]
    ptsXYZproj[:, 1] = ptsXYZproj[:, 1] / ptsXYZproj[:, 2]

    return ((Tproj, Ttable), ptsXYZproj)

def ramp_flat(n):
    '''
    Create 1D ramp filter in the spatial domain.

    :param n: Size of the filter, should be of power of 2. 
    :return: Ramp filter response vector.
    '''
    nn = np.arange(-(n // 2), (n // 2)) 
    h = np.zeros((nn.size,), dtype='float')
    h[n // 2] = 1 / 4 #in the middle sets 1/4
    odd = np.mod(nn, 2) == 1 #make array : [False,  True, False, ...,  True, False,  True]
    h[odd] = -1.0 / (np.pi * nn[odd]) ** 2.0
    return h, nn

def nextpow2(i):
    '''
    Find 2^n that is equal to or greater than

    :param i: arbitrary non-negative integer
    :return: integer with power of two
    '''
    n = 1
    while n < i: n *= 2
    return n

def create_filter(filter_type, ramp_kernel, order, d):
    '''
    Create 1D filter of selected type.

    :param filter_type: Select filter by name, e.g. "ram-lak", "shepp-logan", "cosine", "hamming", "hann". 
    :param ramp_kernel: Input ramp filter kernel, size defined by input image size.
    :param order: Filter order, should be of power of 2.
    :param d: Cut-off frequency (0-1).
    :return: Desired filter response vector.
    '''
    f_kernel = np.abs(np.fft.fft(ramp_kernel)) * 2  # transform ramp filter to freqency domain
    filt = f_kernel[0:order // 2 + 1].transpose()
    w = 2.0 * np.pi * np.arange(0, filt.size) / order  # frequency axis up to Nyquist

    filter_type = filter_type.lower()
    if filter_type == 'ram-lak':
        # do nothing
        None
    elif filter_type == 'shepp-logan':
        # be careful not to divide by 0:
        filt[1:] = filt[1:] * (np.sin(w[1:] / (2 * d)) / (w[1:] / (2 * d)))
    elif filter_type == 'cosine':
        filt[1:] = filt[1:] * np.cos(w[1:] / (2 * d))
    elif filter_type == 'hamming':
        filt[1:] = filt[1:] * (.54 + .46 * np.cos(w[1:] / d))
    elif filter_type == 'hann':
        filt[1:] = filt[1:] * (1 + np.cos(w[1:] / d)) / 2
    else:
        raise ValueError('filter_type: invalid filter selected "{}"'.format(filter_type))

    filt[w > np.pi * d] = 0  # crop the frequency response
    filt = np.hstack((filt, filt[-2:0:-1]))  # enforce symmetry of the filter

    return filt

def get_volume(voldims=(100, 100, 100), sampling_mm=1):
    '''
    Define volume size and sampling points.
    
    :param voldims: Define volume size in mm (for X x Y x Z axes).
    :param sampling_mm: Volume sampling step in mm. For anisotropic sampling define a tuple or list.
    :return: Grid of points in the volume (in homogeneous coordinates) in "grid" and grid sizes (X x Y x Z) in "volsiz". 
    '''
    if not isinstance(sampling_mm, (tuple, list)):
        sampling_mm = [sampling_mm] * 3
    # get sampling points; the grid is axis aligned, and defined wrt to reference
    # point on top of the caliber, which is determined by imaging the caliber
    xr = np.arange(-voldims[0] / 2, voldims[0] / 2, sampling_mm[0]) #prepare sampling points regarding volume taken
    yr = np.arange(-voldims[1] / 2, voldims[1] / 2, sampling_mm[1])
    zr = np.arange(0, voldims[2], sampling_mm[2])
    xg, yg, zg = np.meshgrid(xr, yr, zr, indexing='xy')
    # store volume shape
    grid_size = xg.shape
    # define matrix of homogeneous point coordinates
    grid = np.vstack((xg.flatten(), yg.flatten(), zg.flatten(), np.ones_like(xg.flatten()))).transpose()
    return grid, grid_size

def filter_projection(proj, filter_type='hann', cut_off=1, axis=0):
    '''
    Filter projection image using ramp-like filter. Be careful to select the filter axis, which is (corresponding to 
    the rotation axis.

    :param proj: Projection image (u x v)
    :param filter_type: Select filter by name, e.g. "ram-lak", "shepp-logan", "cosine", "hamming", "hann".
    :param cut_off: Cut-off frequency (0-1).
    :param axis: Select axis 0/1 to apply the filter.
    :return: Filtered projection image. 
    '''
    if filter_type == 'none':
        return proj

    if axis == 1:
        proj = proj.transpose()

    nu, nv = proj.shape
    filt_len = np.max([64, nextpow2(2 * nu)])
    ramp_kernel, nn = ramp_flat(filt_len)

    filt = create_filter(filter_type, ramp_kernel, filt_len, cut_off)
    filt = np.tile(filt[:, np.newaxis], nv) #Construct an array by repeating A the number of times given by reps (nv)

    # append zeros
    fproj = np.zeros((filt_len, nv), dtype='float')
    fproj[filt_len // 2 - nu // 2:filt_len // 2 + nu // 2, :] = proj

    # filter using fourier theorem
    fproj = np.fft.fft(fproj, axis=0)
    fproj = fproj * filt
    fproj = np.real(np.fft.ifft(fproj, axis=0))
    fproj = fproj[filt_len // 2 - nu // 2:filt_len // 2 + nu // 2, :]

    if axis == 1:
        fproj = fproj.transpose()

    return fproj

def deg2rad(ang):
    '''
    Convert angle in degrees to radians.

    :param ang: Angle in degrees. 
    :return: Angle in radians.
    '''
    return ang * np.pi / 180.0

def fbp(imgs, angles, Tproj, out_fname='volume', sampling_mm=2, filter_type='hann', cut_off=0.75):
    '''
    Filtered backprojection 3D reconstruction.
    
    :param imgs: List of projection images for reconstruction.
    :param angles: List of relative rotation angles corresponding to each image. 
    :param Tproj: Transformation matrix obtained from imaging the calibration object.
    :param out_fname: Filename for output reconstructed 3D volume. 
    :param sampling_mm: Volume sampling step in mm. For anisotropic sampling define a tuple or list.
    :param filter_type: Select filter by name, e.g. "ram-lak", "shepp-logan", "cosine", "hamming", "hann".
    :return: Volume of reconstructed 3D grayscale image.
    '''

    # debug: show result
    # i = 0; rvlib.showImage(imgs[i], iTitle='image #%d' % i)
    # plt.plot(grid[:,0], grid[:,1],'rx')
    if not isinstance(sampling_mm, (tuple, list)):
        sampling_mm = [sampling_mm] * 3

    # get sampling points in homogeneous coordinates
    grid_raw, grid_size = get_volume(
        voldims=(VOLUME_LENGTH, VOLUME_WIDTH, VOLUME_HEIGHT),
        sampling_mm=sampling_mm
    )

    # initialize volume
    vol = np.zeros(grid_size)
    xs, ys, zs = grid_size

    for i in range(len(angles) - 1):
        # display current status
        print("processing image #%d/%d" % (i + 1, len(angles)))

        # normalize image
        img_t = rgb2gray(imgs[i]).astype('float')
        #    img_t = (img_t - np.min(img_t)) / (np.max(img_t) - np.min(img_t))
        img_t = (img_t - np.mean(img_t)) / (np.std(img_t))

        # filter projection image
        img_f = filter_projection(img_t, filter_type, cut_off=cut_off, axis=1)
        img_f = (img_f - np.mean(img_f)) / (np.std(img_f))

        # define function to put points in reference space
        get_grid_at_angle = lambda ang: \
            np.dot(np.dot(Tproj[1], transRigid3D(trans=(0, 0, 0), rot=(0, 0, ang))), #Tproj[1] == Ttable
                   np.transpose(grid_raw)).transpose()  

        # project points to imaging plane
        grid = np.dot(get_grid_at_angle(deg2rad(ROTATION_DIRECTION*angles[i])), Tproj[0].transpose())
        grid[:, 0] = grid[:, 0] / grid[:, 2]
        grid[:, 1] = grid[:, 1] / grid[:, 2]
        grid[:, 2] = 1

        # correct in-plane errors due to incorrect geometry
        #    grid = np.dot(grid, Tcorr[i].transpose())

        #    plt.close('all')
        #    rvlib.showImage(img_t, iTitle='original grid')
        #    plt.plot(grid[:,0], grid[:,1],'rx')
        #
        # rvlib.showImage(img_t, iTitle='corrected grid')
        # plt.plot(grid[:,0], grid2[:,1],'rx')

        # rvlib.showImage(img_f)

        # interpolate points to obtain backprojected volume
        us, vs = img_f.shape
        img_backprojected = interpn((np.arange(vs), np.arange(us)), img_f.transpose(),
                                    grid[:, :2], method='linear', bounds_error=False)
        img_backprojected = img_backprojected.reshape((xs, ys, zs))
        vol = vol + img_backprojected
        vol[np.isnan(vol)] = 0

    ##if we want to create nrrd file
    # print('Writing volume to file "{}.nrrd"...'.format(out_fname))
    # if os.path.splitext(out_fname)[-1] != '.nrrd':
    #     out_fname = '{}.nrrd'.format(out_fname)

    # img = itk.GetImageFromArray(np.transpose(vol, [2,1,0]))
    # img.SetSpacing(sampling_mm)
    # itk.WriteImage(img, out_fname, True)

    return vol

def thresholdImage(iImage, iThreshold):
    oImage = 255 * np.array(iImage > iThreshold, dtype='uint8')
    return oImage

def calculate_threshold_percentage(image, percentage):
    flat_image = image.flatten()
    sort_flat_image = np.sort(flat_image)
    len_flat_image = len(flat_image)
    index_50 = int(len_flat_image * percentage)
    
    #plt.figure()
    #plt.hist(sort_flat_image)
    
    t_1 = sort_flat_image[index_50]
    t_2 = sort_flat_image[index_50 - 1]
    
    t = (t_1 + t_2)/2
    #print(t_1, t_2)
    #print(t)

    return t

def get_point_cloud(vol, ThresImageMaxShare=0.3, Deci=5, startHeightShare=0.1, endHeightShare=0.9, circleRadiusLimit=20):

    def distanceFromCenter(centerX, centerY, pointX, pointY):
        distance = np.sqrt((centerX - pointX)**2 + (centerY - pointY)**2)
        return distance

    pointCoorX = []
    pointCoorY = []
    pointCoorZ = []

    initZ = len(vol[0,0,:])

    startZ = int(np.round(startHeightShare*initZ))
    endZ = int(np.round(endHeightShare*initZ))
    vol = vol[:,:,startZ:endZ]

    [dx, dy, dz] = vol.shape
    centerX = dx/2
    centerY = dy/2
    for dZ in range(dz):
        dImage = vol[:,:,dZ]
        dImage = dImage + abs(np.min(dImage))

        dImage = thresholdImage(dImage, np.median(dImage)*ThresImageMaxShare)
        #dImage = thresholdImage(dImage, np.mean(dImage)*ThresImageMaxShare)
        #dImage = thresholdImage(dImage, np.max(dImage)*ThresImageMaxShare)

        for dX in range(dx):
            for dY in range(dy):
                if (distanceFromCenter(centerX, centerY, dX, dY) < circleRadiusLimit):
                    if (dImage[dX, dY] == 0):
                        pointCoorX.append(dX)
                        pointCoorY.append(dY)
                        pointCoorZ.append(dZ)

    pointCoorX = pointCoorX[::Deci]
    pointCoorY = pointCoorY[::Deci]
    pointCoorZ = pointCoorZ[::Deci]

    return pointCoorX, pointCoorY, pointCoorZ

def sharpenImage(iImage, c):
    laplace_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    oImage = np.zeros_like(iImage)
    copy_iImage = np.copy(iImage)
    delta = ni.convolve(iImage, laplace_kernel, mode='nearest')
    oImage = copy_iImage - c*delta
    oImage = limitRange(oImage, iImage.dtype)
    
    return oImage

def get_point_cloud_surface(vol, ThresImageMaxShare=0.3, Deci=5, startHeightShare=0.1, endHeightShare=0.9, circleRadiusLimit=20):

    def distanceFromCenter(centerX, centerY, pointX, pointY):
        distance = np.sqrt((centerX - pointX)**2 + (centerY - pointY)**2)
        return distance

    pointCoorX = []
    pointCoorY = []
    pointCoorZ = []

    initZ = len(vol[0,0,:])

    startZ = int(np.round(startHeightShare*initZ))
    endZ = int(np.round(endHeightShare*initZ))
    vol = vol[:,:,startZ:endZ]

    [dx, dy, dz] = vol.shape
    centerX = dx/2
    centerY = dy/2
    for dZ in range(dz):
        dImage = vol[:,:,dZ]
        dImage = dImage + abs(np.min(dImage))

        dImage = thresholdImage(dImage, np.median(dImage)*ThresImageMaxShare)
        shrpImage = sharpenImage(dImage, 0.3)
        diffImage = dImage - shrpImage

        for dX in range(dx):
            for dY in range(dy):
                if (distanceFromCenter(centerX, centerY, dX, dY) < circleRadiusLimit):
                    if (diffImage[dX, dY] >= 1):
                        pointCoorX.append(dX)
                        pointCoorY.append(dY)
                        pointCoorZ.append(dZ)

    pointCoorX = pointCoorX[::Deci]
    pointCoorY = pointCoorY[::Deci]
    pointCoorZ = pointCoorZ[::Deci]

    return pointCoorX, pointCoorY, pointCoorZ

def plot_point_cloud(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_aspect('equal')

    scat = ax.scatter(X, Y, Z)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.max(np.array([np.max(X) - np.min(X), np.max(Y) - np.min(Y), np.max(Z) - np.min(Z)]))
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(np.max(X) - np.min(X))
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(np.max(Y) - np.min(Y))
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(np.max(Z) - np.min(Z))
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.grid()
    # plt.show()

def crop_image(iImageArray, pxFromRight, pxFromLeft, pxFromUp, pxFromDown):
    #original iz maina
    # # obrezovanje
    # slika_x, slika_y = slike[0].shape
    # nslike = []
    # for sl in range(len(slike)):
    #     dslika =  slike[sl]
    #     dslika = dslika[200:slika_x-100, 300:slika_y-300]
    #     nslike.append(dslika)

    # slike = nslike
    slika_x, slika_y = iImageArray[0].shape
    oImageArray = []
    for sl in range(len(iImageArray)):
        dImage =  iImageArray[sl]
        dImage = dImage[pxFromUp:slika_x-pxFromDown, pxFromLeft:slika_y-pxFromRight]
        oImageArray.append(dImage)

    return oImageArray

def limitRange(iImage, outputType):
    if outputType.kind in ('u', 'i'):
        max_val = np.iinfo(outputType).max
        min_val = np.iinfo(outputType).min
        iImage[iImage < min_val] = min_val
        iImage[iImage > max_val] = max_val
    return iImage.astype(outputType)

def sharpenImage(iImage, c):
    laplace_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    oImage = np.zeros_like(iImage)
    copy_iImage = np.copy(iImage)
    delta = ni.convolve(iImage, laplace_kernel, mode='nearest')
    oImage = copy_iImage - c*delta
    oImage = limitRange(oImage, iImage.dtype)
    
    return oImage

## ICP-------------------------------------------------------------------------------------------
def read_txt_data(data_path):
    with open(data_path, 'r') as f:
        read_data = f.read()
    tocke = np.reshape(np.array(np.matrix(read_data[6:])), [np.int(np.float(read_data[:5])), 3])
    return tocke


def visualize(data, model):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r')
    ax.scatter(model[:, 0], model[:, 1], model[:, 2], c='b')
    plt.show()


class IterativeClosestPoint(object):
    def __init__(self, data, model, toga='true', maxIterations=100, tolerance=0.001):
        if data.shape[1] != model.shape[1]:
            raise 'Both point clouds must have the same number of dimensions!'

        self.pts1 = model
        self.pts2 = data #this one is transformed
        self.ptsT = 0
        self.T = 0
        self.R = 0
        self.mean_error = 0
        self.toga = toga
        self.iteration = 0
        self.maxIterations = maxIterations
        self.tolerance = tolerance

    def register(self):

        if self.toga == 'true':
            self.icp(self.pts1, self.pts2)
            # icp da ven transformacijsko matriko za poravnavo prvega parametra!
            self.ptTransform3d(self.pts2)
        return self.ptsT, self.T, self.R, self.mean_error


    def ptTransform3d(self, pts):
        x = np.asarray(pts[:, 0], dtype=np.float)
        y = np.asarray(pts[:, 1], dtype=np.float)
        z = np.asarray(pts[:, 2], dtype=np.float)
        T = np.asarray(self.T, dtype=np.float)

        r = np.vstack([x.flatten(), y.flatten(), z.flatten(), np.ones([x.size])])

        rT = np.dot(T, r)

        xT = rT[0] / rT[3]
        yT = rT[1] / rT[3]
        zT = rT[2] / rT[3]

        xT.shape = x.shape
        yT.shape = y.shape
        zT.shape = z.shape

        self.ptsT = np.vstack((xT, yT, zT)).T

    def icp(self, B, A):


        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m + 1, A.shape[0]))
        dst = np.ones((m + 1, B.shape[0]))
        src[:m, :] = np.copy(A.T)
        dst[:m, :] = np.copy(B.T)

        prev_error = 0

        for i in range(self.maxIterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.nearest_neighbor(src[:m, :].T, dst[:m, :].T)

            # compute the transformation between the current source and nearest destination points
            self.T, _, _ = self.best_fit_transform(src[:m, :].T, dst[:m, indices].T)

            # update the current source
            src = np.dot(self.T, src)

            # check error
            self.mean_error = np.mean(distances)
            if np.abs(prev_error - self.mean_error) < self.tolerance:
                break
            prev_error = self.mean_error

        # calculate final transformation and rotational matrix
        self.T, self.R, _ = self.best_fit_transform(A, src[:m, :].T)

    def nearest_neighbor(self, src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    def best_fit_transform(self, A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
          A: Nxm numpy array of corresponding points
          B: Nxm numpy array of corresponding points
        Returns:
          T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
          R: mxm rotation matrix
          t: mx1 translation vector
        '''

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R, centroid_A.T)

        # homogeneous transformation
        T = np.identity(m + 1)
        T[:m, :m] = R
        T[:m, m] = t
        
        return T, R, t


def transAffine3D(iScale = (1,1,1), iTrans = (0,0,0), iRot = (0,0,0), iShear = (0,0,0,0,0,0)):
    iRot0 = iRot[0]*np.pi/180
    iRot1 = iRot[1]*np.pi/180
    iRot2 = iRot[2]*np.pi/180
    
    oMatScale = np.array( ((iScale[0],0,0, 0),(0, iScale[1],0, 0), (0,0,iScale[2], 0), (0,0,0,1) ))   
    oMatTrans = np.array(((1,0,0,iTrans[0]), (0,1,0,iTrans[1]), (0,0,1,iTrans[2]), (0,0,0,1)))
    
    Rz = np.array(((1,0,0,0), (0,np.cos(iRot0),-np.sin(iRot0),0), \
                   (0,np.sin(iRot0),np.cos(iRot0),0) , (0,0,0,1)))
    Ry = np.array(((np.cos(iRot1),0,np.sin(iRot1),0), (0,1,0,0), \
                   (-np.sin(iRot1),0,np.cos(iRot1),0) , (0,0,0,1)))
    Rx = np.array(((np.cos(iRot2),-np.sin(iRot2),0,0), (np.sin(iRot2),np.cos(iRot2),0,0),\
                   (0,0,1,0) , (0,0,0,1)))
    
    oMatRot = np.dot(Rz, np.dot(Ry, Rx))
        
    oMatShear = np.array(((1,iShear[0],iShear[1],0), (iShear[2],1,iShear[5],0),\
                          (iShear[3],iShear[4],1,0), (0,0,0,1)))
    
    oMat3D = np.dot(oMatTrans,np.dot(oMatShear, np.dot(oMatRot, oMatScale))) #matricno mnozimo
    
    return oMat3D


def transform_data(model, data, search_step):
    mean_err_min = sys.maxsize
    for steps in range(0,-360,-search_step):
        Mat_rot = transAffine3D(iRot = (0,0,steps))
        data_out = np.dot(data, Mat_rot.transpose())

        data_out = data_out[:,0:3]
        model_out = model[:,0:3]

        icp = IterativeClosestPoint(data=data_out, model=model_out)
        register_points_icp, _, R_icp, mean_err = icp.register()
        z = np.arctan2(R_icp[1,0], R_icp[0,0])
        angleZ = np.degrees(z)
        # x = np.arctan2(R_icp[2,1], R_icp[2,2])
        # x_angle = np.degrees(x)
        # y = np.arctan2(-R_icp[2,0], np.sqrt(R_icp[2,1]**2 + R_icp[2,2]**2))
        # y_angle = np.degrees(y)
        # print(x_angle, y_angle, z_angle)

        if mean_err_min > mean_err:
            register_points_icp_best = register_points_icp
            mean_err_best = mean_err
            angleZ_best = angleZ + steps
            mean_err_min = mean_err

    return register_points_icp_best, np.abs(angleZ_best)

def prepare_sets(modelCoor, dataCoor):
    pointCoorX_ref,pointCoorY_ref,pointCoorZ_ref = modelCoor
    pointCoorX,pointCoorY,pointCoorZ = dataCoor

    model_in = np.dstack((pointCoorX_ref,pointCoorY_ref,pointCoorZ_ref, np.ones([np.size(pointCoorZ_ref)])))
    model_in = model_in[0]
    data_in = np.dstack((pointCoorX,pointCoorY,pointCoorZ, np.ones([np.size(pointCoorZ)])))
    data_in = data_in[0]

    # visualize(data_in, model_in) #visualise transformed and nontransformed data

    if np.shape(data_in)[0] < np.shape(model_in)[0]:
        index_lim = np.shape(data_in)[0]
    elif np.shape(data_in)[0] > np.shape(model_in)[0]:
        index_lim = np.shape(model_in)[0]

    model_in = model_in[np.random.randint(0, np.shape(model_in)[0], index_lim)]
    data_in = data_in[np.random.randint(0, np.shape(data_in)[0], index_lim)]

    return model_in, data_in
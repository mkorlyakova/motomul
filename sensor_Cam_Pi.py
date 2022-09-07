import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

import pickle
import os
import glob


def GetFrameIm( input_image_path, w=640, h=480):
    print(input_image_path)
    myFrame = Image.open(input_image_path)
    print(myFrame)
    img = np.array(myFrame)
    img =  cv2.rotate(img, cv2.ROTATE_180)

    return img
def findObject(bodyCascade, img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    body = bodyCascade.detectMultiScale(imgGray, 1.05, 3)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in body:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 10)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        myListArea.append(area)
        myListC.append([cx, cy])

    if len(myListArea) != 0:
        i = myListArea.index(max(myListArea))
        return img, [myListC[i], myListArea[i]]
    else:
        return img, [[0, 0], 0]

def imread(input_image_path, SIZE):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))

    return original_image

def calibrate_camera( columns, rows, dir_im):
    """

    :param columns: number of grids in x axis
    :param rows: number of grids in y axis
    :param dir_im: path contains the calibration images
    :return: write calibration file into basepath as calibration_pickle.p
    """

    objp = np.zeros((columns*rows,3), np.float32)
    objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(os.path.join(dir_im, 'image*.jpg'))
    print(images)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (columns,rows),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (columns,rows), corners, ret)
            cv2.imshow('input image',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()


    # calibrate the camera
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we don't use rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    destnation = os.path.join(dir_im,'calibration_pickle.p')
    pickle.dump( dist_pickle, open( destnation, "wb" ) )
    print("calibration data is written into: {}".format(destnation))

    return mtx, dist


def dist_chessboard(image, columns, rows):
    """

    """
    found_chessboard = [False]
    # Placeholder for corners
    found_corners = [None]

    found_chessboard,found_corners = cv2.findChessboardCorners(image, (columns, rows), flags=cv2.CALIB_CB_FAST_CHECK)
    cv2.drawChessboardCorners(image, (columns, rows), found_corners , found_chessboard)
    return image, found_chessboard,found_corners

#dir_im = '/home/mariya/PycharmProjects/base/data/19_08/cam0'
#mtx, dist = calibrate_camera( 7, 7, dir_im+'/')
#print(mtx)
#print(dist)
#calib =mtx#
#dist = dist#
calib = [[494.55223427 ,  0.  ,       307.3389561 ], [  0. ,        497.76620709 ,231.49800923], [  0.  ,         0.    ,       1.        ]]
dist = [[ 0.23533876, -0.42059634 ,-0.02012409, -0.01350207,  0.25499589]]
d  = 200./8.
def distance(dir_im, d, col =7,row=7):
    Z = []
    for name in range(len(list_im)):
        if name<10:
            s  = '0'+str(name)
        else:
            s = str(name)
        s = dir_im+'/image'+s+'.jpg'
        #print(s)
        image = GetFrameIm(s, w=640, h=480)
        image, found_,found_corners = dist_chessboard(image, col, row)
        print(found_)
        if found_:
            f_len = len(found_corners)
            points = np.array(found_corners)
            #print(points.shape)
            dd = []
            for i in range(row):
                for j in range(col-1):
                    dd +=[ ((calib[0][0] + calib[1][1])/2) * d /np.sum((points[i*row + j +1,0,:] - points[i*row + j,0,:])**2)**0.5 ]

            #print(dd)
            Z += [np.mean(dd)]
        else:
            Z += [0.0]
            #plt.imshow(image)
            #plt.show()

    Z_raw = np.array(Z)
    Z_clear = Z_raw[np.where(Z_raw!=0)[0]]
    return Z_raw, Z_clear,np.min(Z_clear),np.max(Z_clear),-np.min(Z_clear)+np.max(Z_clear)

REZ = []
d_0 = [300.,0.,300., 300.]
for k,n in enumerate(['0','1','2','3','4','5','6','7','8','9','10']):#)['0','1','2','3','4']):    #['','0','1','2']):

    #tele_path = '/home/mariya/PycharmProjects/base/data/2_08/test' + n + '.csv'
    #dir_im = '/home/mariya/PycharmProjects/base/data/2_08/cam' + n
    #dir_im = '/home/mariya/PycharmProjects/base/data/26_08_22/27_08_'+n
    dir_im = '/home/mariya/PycharmProjects/base/data/4_09/04_09_'+n

    list_im = os.listdir(dir_im)
    #print(list_im)

    try:
        Z_raw, Z_clear, Zmin,Zmax,Z_min_max = distance(dir_im, d, col =7,row=7)
        print(Z_clear.std(),Z_clear.mean())
        #print(Zmin,Zmax,Z_min_max )
        plt.subplot(1,2,1)
        plt.plot(Z_raw,'o', label = 'exp.'+str(k))
        REZ +=[[Zmin,Zmax,Z_min_max,np.abs(Z_min_max - d_0[k]),Z_clear.std(),Z_clear.mean() ]]
        plt.subplot(1, 2, 2)
        plt.plot(Z_clear-Z_clear.mean(), 'o', label='exp.' + str(k))
    except:
        pass
print(REZ)
plt.subplot(1, 2, 1)
plt.title('глубина по оценке камеры')
plt.xlabel('step')
plt.ylabel('Z')
plt.grid()
plt.subplot(1, 2, 2)
plt.title('отклонение от средней оценки камеры')
plt.xlabel('step')
plt.ylabel('Z')
plt.grid()
plt.legend()

plt.show()

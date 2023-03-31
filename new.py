import pyrealsense2 as rs

import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import pandas as pd

import time


def initialize_camera():
    # start the frames pipe
    p = rs.pipeline()
    conf = rs.config()
    conf.enable_stream(rs.stream.accel)
    conf.enable_stream(rs.stream.gyro)
    conf.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    conf.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    prof = p.start(conf)
    return p


def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])


def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])


def image_data(image):
    return np.asarray(image)

p = initialize_camera()
try:
    i = 0
    g = []
    a = []
    im = []
    dep = []
    #t0 = time.localtime()

    while i<30:# redaction ******* number of frame to work / change to True for unlimit frame read
        i += 1
        f = p.wait_for_frames()
        #tc = time.localtime()
        #print(f)
        accel = accel_data(f[3].as_motion_frame().get_motion_data())
        gyro = gyro_data(f[2].as_motion_frame().get_motion_data())
        depth = f.get_depth_frame()
        image = f.get_color_frame()

        D = np.asanyarray(depth.as_frame().get_data())

        I = np.asanyarray(image.as_frame().get_data())

        im += [I]
        dep += [D]


        a += [accel]
        g += [gyro]

        #print("accelerometer: ", accel)
        #print("gyro: ", gyro)

        #print(I.shape)
        #print(D.shape)
        print(i)
        if (not depth ) :
            pass
        else:
            if 0: ## write frame (for dataset make)
                try:
                    matplotlib.image.imsave('img/'+str(i)+'.png', I)
                    matplotlib.image.imsave('dep/' + str(i) + '.png', D/D.max())
                except:
                    pass
            else:
                #depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics
                plt.figure(figsize = (16,8))
                plt.subplot(1,2,1)
                plt.imshow(I)
                plt.subplot(1,2,2)
                plt.imshow(D/D.max())
                plt.show()
            print(D.max())
finally:
    p.stop()

print('accel: ',a)
print('gyro: ',g)
a = np.array(a)
g = np.array(g)

xyz =  [[a[:i,0].sum(),a[:i,1].sum(),a[:i,2].sum()] for i in range(a.shape[0])]
xyz = np.array(xyz)

plt.figure(figsize =(16,5))

plt.subplot(1,4,1)
plt.title('accel')
plt.plot(a[:,0], label = 'x')
plt.plot(a[:,1], label = 'y')
plt.plot(a[:,2], label = 'z')
plt.xlabel('t')
plt.legend()
plt.grid()

plt.subplot(1,4,2)
plt.title('gyro')
plt.plot(g[:,0], label = 'x')
plt.plot(g[:,1], label = 'y')
plt.plot(g[:,2], label = 'z')
plt.xlabel('t')
plt.grid()
plt.legend()

plt.subplot(1,4,3)
plt.title('in plane')
plt.plot(xyz[:,0],xyz[:,1], label = 'xy')
plt.xlabel('t')
plt.legend()
plt.grid()
plt.subplot(1,4,4)
plt.title('z')

plt.plot(xyz[:,2], label = 'z')
plt.xlabel('t')
plt.legend()
plt.grid()
plt.show()


new_ag = np.hstack((a,g,xyz))
## save track to .csv
df = pd.DataFrame(new_ag, columns=['ax','ay','az','gx','gy','gz','x','y','z'])
df.to_csv('new-ag.csv') ## change name for new track




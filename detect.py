
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class Track():

    def __init__(self,tele_path ,dir_im, list_im, cascad_dir ='/home/mariya/PycharmProjects/face_antispoofing_dev/venv/haarcascade_fullbody.xml', t = 0.3, calib =[-7.7344684631347650,- 1.8850986191406250,8.099344796142564]):
        self.dir_im = dir_im
        self.list_im = list_im

        self.x_velocity0 = 0
        self.y_velocity0 = 0
        self.z_velocity0 = 0

        self.x_velocity = 0
        self.y_velocity = 0
        self.z_velocity = 0

        if calib is not None:
            self.ax0 = calib[0]#-7.7344684631347650#-0.5915593461914058  #-7.142909116943359
            self.ay0 = calib[1]#- 1.8850986191406250#-0.6795223151855468 #-1.205576303955078#0.03462015600585936
            self.az0 = calib[2]#8.099344796142564# -0.9252152895395863 #9.02456008568215#0.09418789329933923
        else:
            self.ax0 = 0.#-0.5915593461914058  #-7.142909116943359
            self.ay0 = 0.#-0.6795223151855468 #-1.205576303955078#0.03462015600585936
            self.az0 = 0.# -0.9252152895395863 #9.02456008568215#0.09418789329933923

        self.gx0 =  -0.6914634146341464#
        self.gy0 =   0.7329268292682927#0.03462015600585936
        self.gz0 =  0.18414634146341452#0.09418789329933923

        self.gx = 0.
        self.gy = 0.
        self.gz =  0.

        self.ax = 0
        self.ay = 0
        self.az = 0

        self.x_ = 0
        self.y_ = 0
        self.z_ = 0

        self.us_z = 0
        self.n = 5
        with open(tele_path,'r') as f:
            s = f.read()

        self.list_tele =  s.split('\n')

        self.bodyCascade = cv2.CascadeClassifier(cascad_dir)
        self.myListArea = []
        self.myListC = []
        self.t = t

        self.tele_calibr()
        print(self.ax0, self.ay0, self.az0)
        #input('enter:')


    def tele_calibr(self):
        for i in range(self.n):
            s = self.list_tele[-i-2]
            s = s.split(':')
            print(s)
            num = int(s[0])
            t_ = float(s[1])
            self.us_z =  float(s[2])
            ax = float(s[5][:-5])*0.849847208
            ay = float(s[6][:-5])*0.849847208
            az = float(s[7][:-2])*0.849847208



            self.gx = float(s[10][:-5])
            self.gy = float(s[11][:-5])
            self.gz = float(s[12][:-2])


            self.ax0 += ax
            self.ay0 += ay
            self.az0 += az

        self.ax0 /= (self.n)
        self.ay0 /= (self.n)
        self.az0 /= (self.n)



    def GetFrameIm(self,input_image_path , w=640, h=480):
        print(input_image_path)
        myFrame = Image.open(input_image_path)
        print(myFrame)
        img = np.array(myFrame)
        img = image = cv2.rotate(img, cv2.ROTATE_180)


        return img

    def tele_logger(self,numb_frame):
        s = self.list_tele[numb_frame]
        s = s.split(':')
        print(s)
        num = int(s[0])
        t_ = float(s[1])
        self.us_z =  float(s[2])
        ax = float(s[5][:-5])*0.849847208
        ay = float(s[6][:-5])*0.849847208
        az = float(s[7][:-2])*0.849847208


        self.gx = float(s[10][:-5])
        self.gy = float(s[11][:-5])
        self.gz = float(s[12][:-2])

        if 1:
            self.ax = (ax-self.ax0)
            self.ay = (ay-self.ay0)
            self.az = (az-self.az0)
            self.x_velocity += self.ax * t_
            self.y_velocity += self.ay * t_
            self.z_velocity += self.az * t_

            self.x_ += (self.ax * t_) * t_/2
            self.y_ += (self.ay * t_) * t_/2
            self.z_ += (self.az * t_ ) * t_/2
            self.t = t_


    def findObject(self, img):

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        body = self.bodyCascade.detectMultiScale(imgGray, 1.05, 3)

        myFaceListC = []
        myFaceListArea = []

        for (x, y, w, h) in body:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 10)
            cx = x + w // 2
            cy = y + h // 2
            area = w * h
            self.myListArea.append(area)
            self.myListC.append([cx, cy])

        if len(self.myListArea) != 0:
            i = self.myListArea.index(max(self.myListArea))
            return img, [self.myListC[i], self.myListArea[i]]
        else:
            return img, [[0, 0], 0]

    def get_frame(self, numb = 50, plot_im = 0):
        img_list = []
        xyz_list = []
        for i in range( numb):
            img = self.GetFrameIm(self.dir_im+'/'+self.list_im[i])
            self.tele_logger(i )
            imgi,img_detect_area = self.findObject( img)
            print(img_detect_area)
            if plot_im:
                plt.imshow(imgi)
                plt.show()
            img_list.append(img_detect_area[0])
            xyz_list.append([self.x_velocity,self.y_velocity,self.z_velocity, self.x_,self.y_,self.z_, self.ax, self.ay, self.az, self.gx, self.gy, self.gz,self.t])
        return img_detect_area, xyz_list



def trackFace(myDrone, info, w, pid, pError):
    ## PID
    error = info[0][0] - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    print(speed)
    if info[0][0] != 0:
        myDrone.yaw_velocity = speed
    else:
        myDrone.for_back_velocity = 0
        myDrone.left_right_velocity = 0
        myDrone.up_down_velocity = 0
        myDrone.yaw_velocity = 0
        error = 0
    if myDrone.send_rc_control:
        myDrone.send_rc_control(myDrone.left_right_velocity,
                                myDrone.for_back_velocity,
                                myDrone.up_down_velocity,
                                myDrone.yaw_velocity)
    return error


def Setup(yolo=''):

    weights = os.path.sep.join([yolo, "yolov2-tiny.weights"])
    config = os.path.sep.join([yolo, "yolov2-tiny.cfg"])
    labelsPath = os.path.sep.join([yolo, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    neural_net = cv2.dnn.readNetFromDarknet(config, weights)

    ln = neural_net.getLayerNames()
    print(":",len(ln), ln)

    print('net',neural_net.getUnconnectedOutLayers())
    print(neural_net.getUnconnectedOutLayersNames())
    #ln = [neural_net.getUnconnectedOutLayersNames()]#ln[i[0] - 1] for i in neural_net.getUnconnectedOutLayers()]
    return neural_net, ln, LABELS

def Check(a,  b):
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    calibration = (a[1] + b[1]) / 2
    if 0 < dist < 0.25 * calibration:
        return True
    else:
        return False

def ImageProcess(image, net =None, ln= None, LABELS = None ):

    (H, W) = (None, None)
    frame = image.copy()

    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    print('blob',blob)
    plt.imshow(blob[0][0,:,:])
    plt.show()
    layerOutputs = net.forward(ln[-1])

    print('out',len(layerOutputs),len(layerOutputs[-1]))
    confidences = []
    outline = []
    for output in layerOutputs:

        #for detection in output:
            detection = output
            print((detection.shape))
            scores = detection[5:]
            print(scores, np.argmax(scores))
            maxi_class = np.argmax(scores)
            confidence = scores[maxi_class]
            if LABELS[maxi_class] == "person":
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    outline.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
    box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.3)
    if len(box_line) > 0:
        flat_box = box_line.flatten()
        pairs = []
        center = []
        status = []
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(False)
        for i in range(len(center)):
            for j in range(len(center)):
                close = Check(center[i], center[j])
                if close:
                    pairs.append([center[i], center[j]])
                    status[i] = True
                    status[j] = True
        index = 0
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            if status[index] == True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
            elif status[index] == False:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            index += 1
        for h in pairs:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
    return frame

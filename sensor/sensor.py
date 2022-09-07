import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from PIL import Image
import matplotlib.pyplot as plt
import os

import detect


def GetFrameIm(self, input_image_path, w=640, h=480):
    print(input_image_path)
    myFrame = Image.open(input_image_path)
    print(myFrame)
    img = np.array(myFrame)
    img = image = cv2.rotate(img, cv2.ROTATE_180)


def imread(input_image_path, SIZE):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))

    return original_image
n = '5'
tele_path = '/home/mariya/Рабочий стол/obninsk/26_08_22/test'+n+'.csv'
dir_im = '/home/mariya/Рабочий стол/obninsk/26_08_22/27_08_'+n

tele_path = '/home/mariya/Рабочий стол/obninsk/4_09/test'+n+'.csv'
dir_im = '/home/mariya/Рабочий стол/obninsk/4_09/04_09_'+n
list_im = os.listdir(dir_im)

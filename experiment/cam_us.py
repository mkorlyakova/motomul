from picamera import PiCamera
from time import sleep,time
import RPi.GPIO as GPIO
import smbus            #import SMBus module of I2C

from mpu6050 import mpu6050
sensor = mpu6050(0x68)
accelerometer_data = sensor.get_accel_data()
print(accelerometer_data)
input('enter')
#some MPU6050 Registers and their Address
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47

def MPU_Init():
    #write to sample rate register
    bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
    
    #Write to power management register
    bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
    
    #Write to Configuration register
    bus.write_byte_data(Device_Address, CONFIG, 0)
    
    #Write to Gyro configuration register
    bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)
    
    #Write to interrupt enable register
    bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
    #Accelero and Gyro value are 16-bit
        high = bus.read_byte_data(Device_Address, addr)
        low = bus.read_byte_data(Device_Address, addr+1)
    
        #concatenate higher and lower value
        value = ((high << 8) | low)
        
        #to get signed value from mpu6050
        if(value > 32768):
                value = value - 65536
        return value


bus = smbus.SMBus(1)    # or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x68   # MPU6050 device address


MPU_Init()

print (" Reading Data of Gyroscope and Accelerometer")
i = 0
while i<0:
    i = i+1
    #Read Accelerometer raw value
    acc_x = read_raw_data(ACCEL_XOUT_H)
    acc_y = read_raw_data(ACCEL_YOUT_H)
    acc_z = read_raw_data(ACCEL_ZOUT_H)
    
    #Read Gyroscope raw value
    gyro_x = read_raw_data(GYRO_XOUT_H)
    gyro_y = read_raw_data(GYRO_YOUT_H)
    gyro_z = read_raw_data(GYRO_ZOUT_H)
    
    #Full scale range +/- 250 degree/C as per sensitivity scale factor
    Ax = acc_x/16384.0
    Ay = acc_y/16384.0
    Az = acc_z/16384.0
    
    Gx = gyro_x/131.0
    Gy = gyro_y/131.0
    Gz = gyro_z/131.0
    

    print ("Gx=%.2f" %Gx, u'\u00b0'+ "/s", "\tGy=%.2f" %Gy, u'\u00b0'+ "/s", "\tGz=%.2f" %Gz, u'\u00b0'+ "/s", "\tAx=%.2f g" %Ax, "\tAy=%.2f g" %Ay, "\tAz=%.2f g" %Az)     



def setup_us(echo = 6,trigger = 26, tameout=2):
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    trigger = trigger
    echo = echo 
    GPIO.setup(echo,GPIO.IN, pull_up_down = GPIO.PUD_DOWN)
    GPIO.setup(trigger,GPIO.OUT)
    GPIO.output(trigger, GPIO.LOW)
    sleep(2)
    return echo, trigger
def run_sd(echo=6, trigger = 19):
    GPIO.output(trigger, GPIO.HIGH)
    sleep(0.00001)
    GPIO.output(trigger, GPIO.LOW)
    dist = [time()]
    k = 0
    while GPIO.input(echo) == 0:
        dist = [time()]
        k += 1
        if k > 100000:
            print(0)
            break
    k = 0    
    while GPIO.input(echo) == 1:
        
        
        dist  += [time()]
        k += 1
        if k > 100000:
            break
        
    
    return((dist[-1]-dist[0])*17150)

def short_take(camera, i=0,echo = 6,trigger = 19, cam=False):
    if cam:
        try:
            pass
            # запускаем предпросмотр сигнала с камеры поверх всех окон
            #camera.start_preview()

            # даем камере 3 секунды для автофокусировки
            

            # делаем снимок, и сохраняем его в файл
            #camera.capture('/home/pi/Desktop/CamSource/cam/image'+str(i)+'.jpg')

            # выключаем режим просмотра
            #camera.stop_preview()
            
        except:
             pass
    d = run_sd(echo=echo, trigger = trigger)
    print(d)
    
from io import BytesIO


camera = PiCamera(resolution=(640, 480), framerate=3)
# Set ISO to the desired value
camera.iso = 5
# Wait for the automatic gain control to settle

sleep(2)
# Now fix the values
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = g
# Finally, take several photos with the fixed settings
camera.capture_sequence(['/home/pi/Desktop/CamSource/cam/image%02d.jpg' % i for i in range(10)])


setup_us(echo = 14,trigger = 4)

#setup_us(echo = 6,trigger = 19)
#setup_us(echo = 5,trigger = 19) 
#short_take(echo = 6,trigger = 26)
for i in range(30):
    short_take(camera,i=i,echo = 14,trigger = 4, cam = True)
    accelerometer_data = sensor.get_accel_data()
    gyro_data = sensor.get_gyro_data()
    print(accelerometer_data,gyro_data )

    sleep(0.51)
    #short_take(echo = 6,trigger = 19)
    
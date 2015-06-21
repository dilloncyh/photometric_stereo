from time import sleep
import RPi.GPIO as GPIO
import picamera
import picamera.array
import glob


#set up camera
camera = picamera.PiCamera()
camera.hflip = True
camera.brightness = 50
camera.resolution = (500,500)
camera.framrate = 1
camera.iso = 200
camera.zoom = (0,0,1,1)
sleep(1.5)
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = g


#label LEDs
red1 = 18
yellow1 = 23
green1 = 24
red2 = 25
yellow2 = 8
green2 = 7

output_list = []

#setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.cleanup()
for i in output_list:
    GPIO.setup(i, GPIO.OUT)

# set all outputs to 'low'
for i in output_list:
    GPIO.setup(i, GPIO.OUT)


#capture sequential images of LEDs
filename='11-6-img_01' 

for i in range(len(output_list)):
    GPIO.output(output_list[i], GPIO.HIGH)
    sleep(1)
    camera.capture('%s%s%02d%s' %(filename, '#', i, '.png'))
    GPIO.output(output_list[i], GPIO.LOW)


#capture image with all lights on
for i in range(len(output_list)):
    GPIO.output(output_list[i], GPIO.HIGH)    
sleep(1)
camera.capture('%s%s%s%s' %(filename, '#', str(len(output_list)), '.png'))

#capture image with all lights off
for i in range(len(output_list)):
    GPIO.output(output_list[i], GPIO.LOW)
camera.capture('%s%s%s%s' %(filename, '#', str(len(output_list)+1), '.png'))

print "Acquisition finished"

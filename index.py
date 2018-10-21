#!/usr/bin/env python


#Importing modules
import numpy as np
import cv2
import time
import math
import os.path
from facerec.feature import Fisherfaces
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
import sys, os
import time
import getopt
import multiprocessing
import shutil

# local modules
from video import create_capture
from common import clock, draw_str, dot1, draw_str1


pi = 0			#change to 1 when running on pi and 0 when running on pc
skip = 1		#change the int value to skip frame(value = 1 when runing on pc, 5,6,7 when running on pi)
minFacesize = 190	#Minimum size of Face needed for start recognize
timeout = 1		#Timout value for closed eye to triger alarm



model = PredictableModel(Fisherfaces(), NearestNeighbor())
pathdir='sampleFace/'

if pi == 1:					#Setting GPIO Pins
	import RPi.GPIO as GPIO			#import GPIO module
	GPIO.setmode(GPIO.BCM)			#Set GPIO pin numbering method
	GPIO.setwarnings(False)			#Disable Warning
	GPIO.setup(10, GPIO.OUT)		#Set pin as output
	GPIO.setup(11, GPIO.OUT)		#Set pin as output
	GPIO.setup(12, GPIO.OUT)		#Set pin as output
	GPIO.setup(13, GPIO.OUT)		#Set pin as output
	GPIO.setup(15, GPIO.OUT)		#Set pin as output
	GPIO.setup(16, GPIO.OUT)		#Set pin as output
	GPIO.setup(17, GPIO.OUT)		#Set pin as output
	GPIO.setup(18, GPIO.OUT)		#Set pin as output
	GPIO.setup(19, GPIO.OUT)		#Set pin as output



	GPIO.output(10, GPIO.LOW)		#GPIO o/p HIGH
	GPIO.output(11, GPIO.HIGH)		#GPIO o/p HIGH
	GPIO.output(12, GPIO.HIGH)		#GPIO o/p HIGH
	GPIO.output(13, GPIO.HIGH)		#GPIO o/p HIGH
	GPIO.output(15, GPIO.HIGH)		#GPIO o/p HIGH
	GPIO.output(16, GPIO.HIGH)		#GPIO o/p HIGH
	GPIO.output(17, GPIO.HIGH)		#GPIO o/p HIGH
	GPIO.output(18, GPIO.HIGH)		#GPIO o/p HIGH


#=======================================================================
#Different function for detection
#======================================================================

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(minFacesize, minFacesize), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

#-------------



def beep(timelength):
    if pi==1:
	    GPIO.output(19, GPIO.HIGH)
	    time.sleep(timelength)
	    GPIO.output(19, GPIO.LOW)
    else:
	print 'buzzer', timelength



#-------------------------------
#Function for drawing rectangle
#-------------------------------
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def read_images(path, sz=(256,256)):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    folder_names = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)

                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y,folder_names]

#=======================================================================

if __name__ == '__main__':
    import sys, getopt


    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])

#-----------------------------------
#Selecting video source(file/camera)
#-----------------------------------

    try:
        video_src = video_src[0]
    except:


    	fn = raw_input("Enter Camera (1/0): ")	#Enable keyboard to enter value
        fn = int(fn)
        if fn == 1:
       		print "Camera 1 "
        	cam = create_capture(fn, fallback='synth:bg=lena.jpg:noise=0.05')

    	elif fn == 0:
        	print "Camera 0"
        	cam = create_capture(fn, fallback='synth:bg=lena.jpg:noise=0.05')

#------------------------
#Defining various cascade
#------------------------

    args = dict(args)
    cascade_fn = args.get('--cascade', "cascade/haarcascade_frontalface_default.xml")


    cascade = cv2.CascadeClassifier(cascade_fn)


#------------------------------
#Variables for counting & reset
#------------------------------

    begin=time.time()
    #dj = dlib.rectangle()

    face = 0
    x11=0

    x10=0
    x13=0

    x12=0

    sk = 1

    calibrate = 0


    recx = 1000
    person = 'unknown'
    folderempty = 1

    starttime = time.time()

    if not os.path.exists(pathdir): os.makedirs(pathdir)
    [X,y,subject_names] = read_images(pathdir)
    if not len(X)==0:
	folderempty = 0
	print 111111111111111111111111
	list_of_labels = list(xrange(max(y)+1))
	print y, list_of_labels, subject_names
	subject_dictionary = dict(zip(list_of_labels, subject_names))
	model.compute(X,y)
    recgnition=50
    rcgn = 0
    timout=0


#=======================================================================
#starting while loop
#-------------------
    while True:
        ret, img = cam.read()				#Readin image
	#img = rotateImage(img, 180)			#Active this if you are placing camera upside down
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	#converting to gray scale
        gray = cv2.equalizeHist(gray)			#Equalising Histogram

	#Variables
	x30 = 0
	face = 0
        t = clock()


        vis = img.copy()				#copying image
	dvis = img.copy()				#copying image


	dib,dig,dir = cv2.split(dvis)       		# Split image to BGR value
	dvis_rgb = cv2.merge([dir,dig,dib])		# Convert image to RGB

	dvis_rgb_sml = cv2.resize(dvis_rgb, (0,0), fx=0.5, fy=0.5)	#Resizing Image
	#print recgnition
	recgnimg = img.copy()
	recgngray = gray.copy()
	recgnfaces = detect(recgngray, cascade)

	for (rx,ry,rw,rh) in recgnfaces:

		face = 1
		sampleImage = gray[ry:rh, rx:rw]
		sampleImage = cv2.equalizeHist(sampleImage)
		sampleImage = cv2.resize(sampleImage, (256,256))
		#cv2.rectangle(sampleImage,(rx,ry),(rw,rh),(255,0,0),2)
		draw_rects(vis, recgnfaces, (0, 255, 255))
		#cv2.imshow('sample',sampleImage)
		#print "asdf", rx-rw
		if recgnition>1:

			recgnition-=1



			if folderempty==0:
				[ predicted_label, generic_classifier_output] = model.predict(sampleImage)
				if int(generic_classifier_output['distances']) <=  300:
					#cv2.putText(recgnimg,'This is : '+str(subject_dictionary[predicted_label]), (rx,ry), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
					if recx>=int(generic_classifier_output['distances']):
						recx = int(generic_classifier_output['distances'])
						person = str(subject_dictionary[predicted_label])
				#print person, int(generic_classifier_output['distances'])


		#print str(subject_dictionary[predicted_label]), int(generic_classifier_output['distances']), person, recx
			if person=='unknown':
				recgnition = 50
			draw_str1(vis, (rx, ry), person, (0,0,255))





	if sk%skip == 0:
		sk = 1



		if calibrate==1:		#Calibration fn
			if rcgn==1:
				rcgn = 0
				rcgn1 = 50
				startRCGN = time.time()
				countRCGN = 0
				name = raw_input('Hello user whats your name?\n Name:')
				if not os.path.exists(pathdir+name): os.makedirs(pathdir+name)


			if int(time.time()-startRCGN) <= 14:
				#print rcgn1
				rcgn1-=1
				facesRCGN = detect(gray, cascade)
				for (Rx,Ry,Rw,Rh) in facesRCGN:
					countRCGN +=1
					equ_imageRCGN = cv2.equalizeHist(gray[Ry:Rh, Rx:Rw])
					resized_imageRCGN = cv2.resize(equ_imageRCGN, (273, 273))
					if countRCGN%3 == 0:
						#print  pathdir+name+str(time.time()-startRCGN)+'.jpg'
						cv2.imwrite( pathdir+name+'/'+str(time.time()-startRCGN)+'.jpg', resized_imageRCGN );





		draw_str1(vis, (360, 40), 'Person: '+person, (0,0,255))
		if person == "No Face":
			if x13==0:
				x13=1
				x10=0
				x12=0
				if pi==1:
					GPIO.output(10, GPIO.LOW)

			as1=0


		elif person == "unknown":
			if x10==0:
				x11=3
				x10=1
				x12=0
				x13=0
				if pi==1:
					GPIO.output(10, GPIO.LOW)

		else:
			#GPIO.output(10, GPIO.LOW)
			if x12==0:
				x11=1
				x12=1
				x10=0
				x13=0
				if pi==1:
					GPIO.output(10, GPIO.HIGH)

		if x11>=0:
			#print "ok"
			#beep(0.3)
			x11 = x11-1



		if face==0 and pi==1:				#Turn of otuputs when face not detected
			#GPIO.output(10, GPIO.LOW)
			GPIO.output(11, GPIO.LOW)
			GPIO.output(12, GPIO.LOW)
			GPIO.output(13, GPIO.LOW)
			GPIO.output(15, GPIO.LOW)
			GPIO.output(17, GPIO.LOW)
			GPIO.output(18, GPIO.LOW)




		if face==0:
			timout+=1
			if timout>=50:
				recgnition=80
				timout=0
				recx=1000
				person = 'No Face'



	sk = sk+1

	cv2.imshow('facedetect', vis)					#show the result

	ch = 0xFFF & cv2.waitKey(5)
	if ch == 27:							#press 'Esc' key for exit
		break
	if ch == ord('c'):						#press key for start calibration
		calibrate = 1
		rcgn = 1

	if ch == ord('d'):
		print "Reset"
		if os.path.exists(pathdir): shutil.rmtree(pathdir)
		if not os.path.exists(pathdir): os.makedirs(pathdir)
		break





#=======================================================================
    cv2.destroyAllWindows()

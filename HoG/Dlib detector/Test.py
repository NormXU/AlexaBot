#!/usr/bin/env python
import dlib
import numpy as np
import cv2
from skimage import io


def stream_detector():
# Load Trained setector
	detector = dlib.simple_object_detector("Pedestriandetector.svm")
	cap = cv2.VideoCapture('demo.mp4')


	while(cap.isOpened()):
	    # Capture frame-by-frame
	    ret, frame = cap.read()
	    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	    dets = detector(frame)
	    
	    for d in dets:
	        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
	    
	    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	    # Display the resulting frame
	    cv2.imshow("frame",frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()







def single_detector():
	detector = dlib.simple_object_detector("Pedestriandetector.svm")
	win_det = dlib.image_window()
	win_det.set_image(detector)

	print("Showing detections on the images in the faces folder...")
	win = dlib.image_window()

	filename = 'resized_set/img-0725-1.jpg'
	img = io.imread(filename)

	dets = detector(img)
	print("Number of faces detected: {}".format(len(dets)))

	# Non-Maximum Suppression

	win.clear_overlay()
	win.set_image(img)
	win.add_overlay(dets)
	dlib.hit_enter_to_continue()



if __name__ == '__main__':
	stream_detector()
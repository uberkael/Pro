import cv2 as cv
import numpy as np
import Tracker
import Selector
import Utiles




if __name__ == "__main__":
	guardar = True
	if guardar:
		fourcc = cv.VideoWriter_fourcc(*"VP80")
		fps = 25
		resolu = (768, 576)
		out = cv.VideoWriter("Salida.avi", fourcc, fps, resolu)
	cap = cv.VideoCapture("Samples/vtest.avi")
	cv.namedWindow("Tracker", cv.WINDOW_NORMAL)
	cv.setWindowProperty("Tracker", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

	while cap.isOpened():
		ret, frame = cap.read()
		img = Tracker.tracker(frame)
		cv.imshow("Tracker", img)
		if guardar:
			out.write(img)
		# if (cv.waitKey(40) & 0xFF == ord('q')):
		if (cv.waitKey(1) & 0xFF == ord('q')):
			break

cv.waitKey(0)

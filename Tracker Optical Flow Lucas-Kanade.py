#######################################
# Tracker de movimiento con MeanShift #
#######################################

import cv2 as cv
import numpy as np
from imutils.video import FPS
import Utiles
import Config
import Selector


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Tracker Lucas-Kanade"
	# Config.Fullscreen(titulo)
	out = None
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida/{titulo}.avi", VidProp.fourcc,
							 VidProp.fps, VidProp.resolu)
	cap = cv.VideoCapture(Config.VidProp.source)
	fps = FPS().start()

	# Localización inicial
	x, y, w, h = 252, 218, 32, 89
	objetivo = (x, y, w, h)
	# params for ShiTomasi corner detection
	feature_params = dict(maxCorners=100,
	                      qualityLevel=0.3,
	                      minDistance=7,
	                      blockSize=7)
	# Parameters for lucas kanade optical flow
	lk_params = dict(winSize=(15, 15),
	                 maxLevel=2,
	                 criteria=(cv.TERM_CRITERIA_EPS |
	                           cv.TERM_CRITERIA_COUNT, 10, 0.03))
	# Create some random colors
	color = np.random.randint(0, 255, (100, 3))
	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
	p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
	p0 = np.array([[[270, 250]]], dtype=np.float32)


	counter = 0
	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			break
		# image, objetivos = tracker(image)
		# Utiles.dibuja_contornos(image, objetivos)

			frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		# Calcula el flujo optico
		p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
    	                                   **lk_params)
		# Select good points
		good_new = p1[st == 1]
		good_old = p0[st == 1]
		# draw the tracks
		for i, (new, old) in enumerate(zip(good_new, good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
			frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
		image = cv.add(frame, mask)
		# cv.imshow("mascara", mask)


		if Config.VidProp.show_fps:
			Utiles.dibuja_FPS(image, fps)
		cv.imshow(titulo, image)
		if Config.VidProp.guardar:
			Utiles.guardar(out, image)
		# if (cv.waitKey(40) & 0xFF == ord('q')):
		if (cv.waitKey(1) & 0xFF == 27):
			break
		if counter == 66:
			break
		else:
			counter += 1
	if Config.VidProp.guardar:
		out.release()
	cap.release()

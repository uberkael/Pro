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
	titulo = "Tracker Optical Flow Dense"
	# Config.Fullscreen(titulo)
	out = None
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida/{titulo}.avi", VidProp.fourcc,
							 VidProp.fps, VidProp.resolu)
	cap = cv.VideoCapture(Config.VidProp.source)
	fps = FPS().start()

	ret, frame1 = cap.read()
	prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1)
	hsv[..., 1] = 255


	counter = 0
	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			break
		# image, objetivos = tracker(image)
		# Utiles.dibuja_contornos(image, objetivos)

		ret, frame2 = cap.read()
		next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
		flow = cv.calcOpticalFlowFarneback(
						prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
		hsv[..., 0] = ang*180/np.pi/2
		hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
		image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


		if Config.VidProp.show_fps:
			Utiles.dibuja_FPS(image, fps)
		cv.imshow(titulo, image)
		if Config.VidProp.guardar:
			Utiles.guardar(out, image)
		# if (cv.waitKey(40) & 0xFF == ord('q')):
		if (cv.waitKey(1) & 0xFF == 27):
			break
		# if counter == 66: break
		else:
			counter += 1
	if Config.VidProp.guardar:
		out.release()
	cap.release()

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
	titulo = "Tracker CAM Shift"
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

	# Extrae un image y pone la region de interes
	ret, image = cap.read()
	roi = image[y:y+h, x:x+w]

	# Para seleccionar la region primero
	# img_final = cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
	# cv.imshow("CAM Shift", img_final)

	hsv_roi = cv.cvtColor(image, cv.COLOR_BGR2HSV)

	mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)),
					  np.array((180., 255., 255.)))
	roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
	cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

	# Criteria 10 iterations or move 1 px
	term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

	image = cv.rectangle(image, (x, y), (x+w, y+h), Config.UI.rojo, 2)
	cv.imshow(titulo, image)

	counter = 0
	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			break
		# image, objetivos = tracker(image)
		# Utiles.dibuja_contornos(image, objetivos)

		hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
		dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
		# Apply CamShift
		ret, objetivo = cv.CamShift(dst, objetivo, term_crit)
		# Dibuja en la imagen
		# x, y, w, h = objetivos
		# img_final = cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)
		pts = cv.boxPoints(ret)
		pts = np.int0(pts)
		image = cv.polylines(image, [pts], True, Config.UI.rojo, 2)

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

#################################################
# Implementacion de una unica torreta solitaria #
#################################################

import cv2 as cv
import numpy as np
from numpy.lib.function_base import append
import Tracker
import Selector
import Utiles
import Config


def objetivos_centros(objetivos):
	"Devuelve los centros de los objetivos"
	print(objetivos)
	centros = np.array()
	for o in objetivos:
		M = cv.moments(o)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		np.append(centros, [cX, cY])
	print(centros)
	return centros


if __name__ == "__main__":
	titulo = "Torreta"
	Config.fullscreen(titulo)
	guardar = True
	out = None
	if guardar:
		from Config import VideoProp
		out = cv.VideoWriter("Salida.avi", VideoProp.fourcc,
							VideoProp.fps, VideoProp.resolu)
	cap = cv.VideoCapture("Samples/vtest.avi")

	while cap.isOpened():
		ret, frame = cap.read()
		img, objetivos = Tracker.tracker(frame)
		cv.imshow(titulo, img)
		if guardar:
			out.write(img)
		# if (cv.waitKey(40) & 0xFF == ord('q')):
		if (cv.waitKey(1) & 0xFF == ord('q')):
			break

cv.waitKey(0)

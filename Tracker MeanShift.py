#######################################
# Tracker de movimiento con MeanShift #
#######################################

import cv2 as cv
import numpy as np
from imutils.video import FPS
import Utiles
import Config
import Selector


def actualiza_objetivo(objetivo, objetivos):
	# "Actualiza el objetivo si se mueve"
	# # No hay nuevos objetivos, nos quedamos con el que se detecto
	# if(len(objetivos) < 1):
	# 	return objetivo
	# # Elimina las detecciones del tracker con area muy distinta del objetivo
	# x, y, w, h = objetivo
	# area_objetivo = w * h
	# rectangulos = Utiles.genera_rectangulos(objetivos)
	# rectangulos = Utiles.elimina_rect_irelevantes(rectangulos, area_objetivo)
	# # Los objetivos no eran suficientemente similares en area
	# if(len(rectangulos) < 1):
	# 	return objetivo
	# # Cambia el objetivo por el mas cercano detectado por el tracker
	# distancia_max = max(w, h)
	# p_objetivo = Utiles.centro_rectangulo(objetivo)
	# objetivo_alt = Selector.objetivo_prioritario(p_objetivo, rectangulos,
	#                                           distancia_max)
	# if(len(objetivo_alt) > 0):
	# 	return objetivo_alt
	# # No habia un objetivo lo suficientemete cercano
	# return objetivo
	return


def tracker(image):
	"""
	Funcion principal, extrae objetivos de los objetos en movimiento
	Devuelve el frame pintado y una lista de contornos de objetivos
	"""
	return


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Tracker MeanShift"
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
	# Extrae un frame y pone la region de interes
	ret, frame = cap.read()
	roi = frame[y:y+h, x:x+w]
	hsv_roi = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)),
						np.array((180., 255., 255.)))
	roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
	cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
	# Criteria 10 iterations or move 1 px
	term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

	while cap.isOpened():
		ret, image = cap.read()
		if not ret: break
		# image, objetivos = tracker(image)
		# Utiles.dibuja_contornos(image, objetivos)

		# OpenCV GPU
		# frame = cv.UMat(frame)
		hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
		dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
		# Apply meanshift
		ret, objetivo = cv.meanShift(dst, objetivo, term_crit)
		# Dibuja en la imagen
		image = cv.rectangle(image, (x, y), (x+w, y+h), Config.UI.rojo, 2)
		x, y, w, h = objetivo



		if Config.VidProp.show_fps:
			Utiles.dibuja_FPS(image, fps)
		cv.imshow(titulo, image)
		if Config.VidProp.guardar:
			Utiles.guardar(out, image)
		# if (cv.waitKey(40) & 0xFF == ord('q')):
		if (cv.waitKey(1) & 0xFF == 27): break
	if Config.VidProp.guardar:
		out.release()
	cap.release()

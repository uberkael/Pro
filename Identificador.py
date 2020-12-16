##################################################################
# Identificador y discriminador de objetivos con DNN MobileNetv3 #
##################################################################

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.api import keras
from tensorflow.keras import datasets, layers, models
import Utiles
import Config
import Tracker


dims = (0, 0)

if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	import cv2 as cv # Solo para DEBUG
	titulo = "Identificador"
	# Config.Fullscreen(titulo)
	out = None
	# Para capturar la salida
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida {titulo}.avi", VidProp.fourcc,
							VidProp.fps, VidProp.resolu)
	# Abre el video y almacena las dimesiones
	cap = cv.VideoCapture("Samples/vtest.avi")
	dims = Utiles.dimensiones_video(cap)

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		img, objetivos = Tracker.tracker(frame)
		# Utiles.dibuja_contornos(img, objetivos)
		ROIs = Utiles.genera_ROIs(objetivos)
		img = Utiles.dibuja_ROIs(frame, ROIs)
		if img.shape[0] > 0:
			cv.imshow(titulo, img)
			if Config.VidProp.guardar:
				Utiles.guardar(out, img)
		# if (cv.waitKey(40) & 0xFF == ord('q')):
		if (cv.waitKey(1) & 0xFF == ord('q')):
			break
	cap.release()
	out.release()

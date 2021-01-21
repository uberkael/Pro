#########################################
# Detector y discriminador de objetivos #
#########################################
##############################
# You Only Look Once (YOLO)  #
##############################
# https: // gilberttanner.com/blog/yolo-object-detection-with-opencv
# https://www.pyimagesearch.com/2020/01/27/yolo-and-tiny-yolo-object-detection-on-the-raspberry-pi-and-movidius-ncs/

import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow.python.keras.api import keras
from tensorflow.keras import datasets, layers, models
from imutils.video import FPS
import Utiles
import Config
import Detector


def guardar_ROIs():
	"""Guarda las imagenes de las personas en la carpeta de salida
	Utiliza un closure como contador """
	contador = 0

	def closure(image, ROIs):
		nonlocal contador
		rects, probabilidades, tipos = ROIs
		for rect, probabilidad, tipo in zip(rects, probabilidades, tipos):
			cv.imwrite(
                f"Salida/Objetivos/Objetivo {contador}.png",
				Utiles.ROI_to_img(image, rect))
			contador += 1
	return closure


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Extractor Personas"
	dims = (0, 0)
	# Config.Fullscreen(titulo)
	out = None
	# Para capturar la salida
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida/{titulo}.avi",
							VidProp.fourcc, VidProp.fps, VidProp.resolu)
		# print(out.get(2)) # TODO
	# Abre el video y almacena las dimesiones
	cap = cv.VideoCapture(Config.VidProp.source)
	dims = Utiles.dimensiones_video(cap)

	# Crea la red neural
	modelo = Detector.genera_DNN()

	# Extrae layers principales de YoLo
	capas_conexion = Detector.capas_desconectadas(modelo)

	labels = Detector.genera_labels()
	colores = Detector.genera_colores(labels)

	fps = FPS().start()

	# Closure para guardar los objetivos en archivos
	save_ROIs = guardar_ROIs()

	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			print('Video file finished.')
			break

		ROIs = Detector.prediccion(image, modelo, capas_conexion, labels)


		save_ROIs(image, ROIs)

		if Config.VidProp.show_fps:
			Utiles.dibuja_FPS(image, fps)
		cv.imshow(titulo, image)

		if (cv.waitKey(1) & 0xFF == 27):
			break

		if Config.VidProp.guardar: Utiles.guardar(out, image)

	cap.release()
	if Config.VidProp.guardar: out.release()

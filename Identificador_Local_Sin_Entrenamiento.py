###########################################################
# Identificador univoco de obejetivos con red siamesa SNN #
###########################################################
# Utiliza Mobilenetv2 como red preentrenada
#https://www.pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/

import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow.python.keras.api import keras
from tensorflow.keras import datasets, layers, models
from imutils.video import FPS
import Utiles
import Config

from tensorflow.keras.datasets import mnist
import tensorflow.keras.applications.mobilenet as mobilenet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import imagenet_utils
from imutils import build_montages

import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


def capa_distancia_euclidea(vectors):
	(A, B) = vectors
	suma_cuadrada = K.sum(K.square(A - B), axis=1, keepdims=True)
	return K.sqrt(K.maximum(suma_cuadrada, K.epsilon()))


def genera_DNN():
	"Genera una Red Neuronal Profunda Preentrenada con los parametros de Config"
	return keras.applications.mobilenet.MobileNet()


def data_preprocesado():
	"Genera con los objetivos"
	path = "Entrada/Objetivos"
	return ImageDataGenerator(
		preprocessing_function=mobilenet.preprocess_input).flow_from_directory(
		directory=path, target_size=(224, 224), batch_size=10, shuffle=True)


def preproceso(frame):
	""" Devuelve una imagen preprocesada para ser procesada por la DNN"""
	tama = Config.DNN.MobileNet.img_size
	frame['image'] = tf.cast(frame['image'], tf.float32)
	frame['image'] = tf.image.resize(frame['image'], (tama, tama))
	frame['image'] = mobilenet.preprocess_input(frame['image'])
	# frame['label'] = tf.one_hot(tf.cast(frame['label'], tf.int32), 2)
	# frame['label'] = tf.cast(frame['label'], tf.float32)
	return frame['image']



if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Detector YoLo"
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
	modelo = genera_DNN()
	datos = data_preprocesado()

	# Extraemos cada clase
	for imagenes, etiquetas in datos:
		# imagenes, etiquetas = next(datos)
		print(list(datos.class_indices.keys()))
		predicciones = modelo.predict(imagenes)
		# print(f"{i} {Utiles.distancia(predicciones_a, predicciones_b)}")
		print(len(predicciones[0]))
		# print(imagenet_utils.decode_predictions(predicciones)[0])
		cv.imshow("Salida", imagenes[0])
		cv.waitKey(0)


	# Extrae layers principales de YoLo
	# capas_conexion = capas_desconectadas(modelo)

	# labels = genera_labels()
	# colores = genera_colores(labels)

	fps = FPS().start()

	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			print('Video file finished.')
			break

		cv.waitKey(0)
		quit()

		# ROIs = prediccion(image, modelo, capas_conexion, labels)

		# image = dibuja_ROIs(image, ROIs)

		if Config.VidProp.show_fps:
			Utiles.dibuja_FPS(image, fps)
		cv.imshow(titulo, image)

		if (cv.waitKey(1) & 0xFF == 27):
			break

		if Config.VidProp.guardar: Utiles.guardar(out, image)

	cap.release()
	if Config.VidProp.guardar: out.release()

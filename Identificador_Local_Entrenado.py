###########################################################
# Identificador univoco de obejetivos con red siamesa SNN #
###########################################################
# Utiliza Mobilenetv2 como red preentrenada
#https://www.pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/

import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import os
import random
import Utiles
import Config
from tensorflow.python.keras.api import keras
import tensorflow.keras.applications.mobilenet as mobilenet


def genera_DNN():
	"Genera una Red Neuronal Profunda Preentrenada con los parametros de Config"
	return keras.models.load_model('customMobileNet.h5')


def preproceso(imagen):
	""" Devuelve una imagen preprocesada para ser procesada por la DNN"""
	tama = Config.DNN.MobileNet.img_size
	imagen = tf.cast(imagen, tf.float32)
	imagen = tf.image.resize(imagen, (tama, tama))
	imagen = mobilenet.preprocess_input(imagen)
	return imagen


def prepara_imagen(archivo):
	"""Lee un archivo y devuelve:
		imagen si preprocesar para imprimir
		imagen lista para usar en la DNN"""
	img = cv.imread(archivo)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	return img, preproceso(img)


def muestra_imagen(axi, pos, img, etiqueta, distancia=None):
	"""Dibujo subplot con matplotlib"""
	axi[pos].imshow(img)
	axi[pos].axes.xaxis.set_ticks([])
	axi[pos].axes.yaxis.set_ticks([])
	axi[pos].title.set_text(etiqueta)
	if distancia:
		axi[pos].set_xlabel(f"Diferencia:\n{distancia}")
		if distancia > 1:
			axi[pos].xaxis.label.set_color('red')


if __name__ == "__main__":
	# DNN para detectar objetos
	modelo = genera_DNN()
	# Importa las etiquetas
	etiquetas = Config.DNN.etiquetas_entrenadas
	# Parametros de plot (graficos)
	plt.rcParams["axes.grid"] = False
	plot_num = 8
	fig, ax = plt.subplots(2, plot_num)
	axi = np.ravel(ax)
	# Directorio raiz de imagenes extraidas
	path = "Entrada/Objetivos/"
	# Seleccionamos imagenes aleatorias y comparamos su distancia
	for i in range(plot_num):
		# Seleccionamos dos imagen aleatorias
		directorio = random.choice(os.listdir(path))
		# Imagen A
		archivo_a = path + directorio + '/' + random.choice(
				os.listdir(path+directorio))
		img_a, img_a_proces = prepara_imagen(archivo_a)
		# Imagen B
		directorio = random.choice(os.listdir(path))
		archivo_b = path + directorio + '/' + random.choice(
                    os.listdir(path+directorio))
		img_b, img_b_proces = prepara_imagen(archivo_b)

		# Prediccion sobre las imagenes usando la red entrenada
		predicciones = modelo.predict(np.array([img_a_proces, img_b_proces]))
		# Mide la distancia entre ambos vectores
		distancia = Utiles.distancia(predicciones[0], predicciones[1])

		# Dibuja los graficos
		# A
		etiqueta = etiquetas[predicciones[0].argmax()]
		muestra_imagen(axi, i, img_a, etiqueta, distancia)
		# B
		etiqueta = etiquetas[predicciones[1].argmax()]
		muestra_imagen(axi, i+plot_num, img_b, etiqueta)
	plt.show()

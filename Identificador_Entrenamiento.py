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
	"Distancia euclidea como capa de una red neuronal"
	(A, B) = vectors
	suma_cuadrada = K.sum(K.square(A - B), axis=1, keepdims=True)
	return K.sqrt(K.maximum(suma_cuadrada, K.epsilon()))


def genera_DNN():
	"Genera una Red Neuronal Profunda Preentrenada con los parametros de Config"
	# Cargamos el modelo de keras
	modelo = keras.applications.mobilenet.MobileNet()
	x = modelo.layers[-6].output
	# Esto significa que al ser un modelo funcional se le pasa como argumento la
	# salida de las ateriores capas (x)
	output = keras.layers.Dense(units=10, activation='softmax')(x)
	# Crea un modelo funcional de red neural
	modelo = keras.Model(inputs=modelo.input, outputs=output)
	# Evitamos que se re-entrene todas las capas excepto las ultimas 23
	for layer in modelo.layers[:-23]:
		layer.trainable = False
	modelo.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
		loss='categorical_crossentropy',
		metrics=['accuracy'])
	return modelo

def entrena_DNN(modelo, datos):
	"Entrena el modelo con los datos para los objetivos"
	modelo.fit(x=datos, epochs=10, verbose=2)
	return modelo


def data_preprocesado():
	"Genera con los objetivos"
	path = "Entrada/Objetivos"
	return ImageDataGenerator(
		preprocessing_function=mobilenet.preprocess_input).flow_from_directory(
		directory=path, target_size=(224, 224), batch_size=10, shuffle=True)


def preproceso(frame):
	"""Devuelve una imagen preprocesada para ser procesada por la DNN"""
	tama = Config.DNN.MobileNet.img_size
	frame['image'] = tf.cast(frame['image'], tf.float32)
	frame['image'] = tf.image.resize(frame['image'], (tama, tama))
	frame['image'] = mobilenet.preprocess_input(frame['image'])
	return frame['image']



if __name__ == "__main__":
	datos = data_preprocesado()
	print(list(datos.class_indices.keys()))
	modelo = genera_DNN()
	modelo = entrena_DNN(modelo, datos)
	modelo.save('customMobileNet.h5')

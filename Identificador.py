##################################################################
# Identificador y discriminador de objetivos con DNN MobileNetv2 #
##################################################################

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.api import keras
from tensorflow.keras import datasets, layers, models
import Utiles
import Config
import Tracker


def genera_DNN():
	"Genera una Red Neuronal Profunda Preentrenada con los parametros de Config"
	img_shape = (Config.DNN.img_size, Config.DNN.img_size, 3)
	return keras.applications.MobileNetV2(input_shape=img_shape,
											weights='imagenet')


def genera_batches(frame, ROIs):
	"Genera imagenes desde las ROI preparadas para procesar por la DNN"
	images = []
	for roi in ROIs:
		img = Utiles.ROI_to_img(frame, roi)
		img = preproceso(img)
		images.append(img)
		# img = np.uint8(img)
	return np.array(images)


def predicciones(frame, ROIs):
	if (len(ROIs) > 0):
		img_batch = genera_batches(frame, ROIs)
		predicciones = modelo.predict(img_batch)
		# print(len(predicciones)) # n
		# print(predicciones.shape) # (n, 1000)
		# print(keras.applications.imagenet_utils.decode_predictions(predicciones))
		predicciones = keras.applications.imagenet_utils.decode_predictions(
					predicciones)
		Utiles.dibuja_predic(frame, ROIs, predicciones)


def preproceso(img):
	"""
	Funcion de preprocesado de imagenes para usar con MobileNetv2
	Modificado sin labels de format_example()
	https://colab.research.google.com/drive/1ZZXnCjFEOkp_KdNcNabd14yok0BAIuwS
	"""
	img = tf.cast(img, tf.float32)
	img = tf.image.resize(img, (Config.DNN.img_size, Config.DNN.img_size))
	# img = vgg16.preprocess_input(img)
	# img = keras.applications.mobilenet_v2.preprocess_input(img)
	# label = tf.one_hot(tf.cast(label, tf.int32), 2)
	# label = tf.cast(label, tf.float32)
	return img


def dibuja_ROIs(frame, ROIs):
	"Auxiliar, no usada en main, Genera una imagen con varias ROI"
	# Pintamos n ROI
	n = 10
	res = np.array([])
	i = 0
	for roi in ROIs:
		#  Pinta solo 5 imagnes
		if i == n: break
		img = Utiles.ROI_to_img(frame, roi)
		img = preproceso(img)
		img = np.uint8(img)
		res = np.hstack([res, img]) if len(res) > 1 else img
		i += 1
	# Completa con negro la imagen hasta la resolucion
	if i != n and len(res.shape) == 3:
		# print((Config.DNN.img_size*n)-res.shape[1])
		img = np.zeros(
			[Config.DNN.img_size, Config.DNN.img_size*n-res.shape[1], 3],
			dtype='uint8')
		res = np.hstack([res, img])
	return cv.UMat(res)


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	import cv2 as cv  # Solo para DEBUG, no como libreria
	titulo = "Identificador"
	dims = (0, 0)
	# Config.Fullscreen(titulo)
	out = None
	# Para capturar la salida
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida {titulo}.avi", VidProp.fourcc,
							VidProp.fps, VidProp.resolu)
		# print(out.get(2)) # TODO
	# Abre el video y almacena las dimesiones
	cap = cv.VideoCapture(Config.VidProp.source)
	dims = Utiles.dimensiones_video(cap)
	# Crea la red neural
	modelo = genera_DNN()
	# modelo.summary()

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		img, objetivos = Tracker.tracker(frame)
		# Utiles.dibuja_contornos(img, objetivos)
		ROIs = Utiles.genera_ROIs(objetivos)
		predicciones(frame, ROIs)
		cv.imshow(titulo, frame)
		if Config.VidProp.guardar:
			Utiles.guardar(out, frame)
		# img = dibuja_ROIs(frame, ROIs)
		# if img.get().shape[0] > 0:
		# 	cv.imshow(titulo, img)
		# 	if Config.VidProp.guardar:
		# 		Utiles.guardar(out, img)
		# if (cv.waitKey(40) & 0xFF == ord('q')):
		if (cv.waitKey(1) & 0xFF == 27):
			break
	if Config.VidProp.guardar: out.release()
	cap.release()

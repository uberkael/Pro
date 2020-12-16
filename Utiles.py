###################################
# Funciones de  proposito general #
###################################

import numpy as np
import cv2 as cv
import tensorflow as tf
import Config


def guardar(out, img):
	"Guarda un video"
	# Si no usa la GPU lo trasnforma en UMat para usarla
	# if(isinstance(img, (np.ndarray))):
	if(not isinstance(img, cv.UMat)):
		img = cv.UMat(img)
	# Extrae el size
	h, w, z = img.get().shape
	wc, hc = Config.VidProp.resolu
	# Si la resolucion del frame no es la requerida la cambia
	if wc != w or hc != h:
		print('resize')
		img = cv.resize(img, (wc, hc))
	# Escribe en el archivo
	out.write(img)

def dimensiones_video(cap):
	"Devuelve las dimensioones x e y de un video"
	return (cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))


def punto_centro(dim):
	"Devuelve el centro de unas dimensiones 2D"
	w, h = dim[0], dim[1]
	return (w//2, h//2)


def distancia(a, b):
	"Calcula la distancia eucidiana entre dos puntos"
	return np.linalg.norm(a-b)
	# dif = a - b
	# return math.sqrt(dif.x*dif.x + dif.y*dif.y)


def gen_p_aleatorios(num_puntos, dims):
	"Genera n puntos aleatorios entre unas dimensiones (mejor cuadrados)"
	return np.random.randint(0, min(dims[0], dims[1]), (num_puntos, 2))


def dibuja_contornos(frame, contornos):
	"""Dibuja
	El contorno,
	El centro del contorno,
	Un rectangulo alrededor,
	El centro del rectangulo
	"""
	for c in contornos:
		# Dibuja el contorno
		cv.drawContours(frame, [c], 0, Config.UI.rojo_claro, 1)
		# Dibujas el centro del contorno
		M = cv.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		cv.circle(frame, (cX, cY), 1, (255, 255, 0), 0)
		x, y, w, h = rectangulo_contenedor(c)
		cv.rectangle(frame, (x, y), (x+w, y+h), Config.UI.rojo_oscuro, 1)
		cv.circle(frame, (x+w//2, y+h//2), 2, Config.UI.cyan, 0)


def rectangulo_contenedor(contorno, margen=0):
	"Devuelve el rectangulo que contiene el contorno"
	# Calculamos el rectangulo que contiene el elemento
	peri = cv.arcLength(contorno, True)
	# Dibuja un rectangulo con su centro
	approx = cv.approxPolyDP(contorno, 0.02*peri, True)
	x, y, w, h = cv.boundingRect(approx)
	if margen:
		x += margen
		y += margen
		w += margen
		h += margen
	return x, y, w, h


def genera_ROIs(contornos):
	# TODO deberia agrandar las zonas
	"Devuelve todos los rectangulos que contienen a los contornos en una lista"
	rectangulos = []
	for c in contornos:
		x, y, w, h = rectangulo_contenedor(c)
		rectangulos.append([x, y, w, h])
	return np.array(rectangulos)


def ROI_to_img(frame, ROI):
	x, y, w, h = ROI
	return frame[y:y+h, x:x+w]


def dibuja_ROIs(frame, ROIs):
	"Genera una imagen con varias ROI"
	# Pintamos n ROI
	n = 10
	res = np.array([])
	i = 0
	for roi in ROIs:
		#  Pinta solo 5 imagnes
		if i == n: break
		img = ROI_to_img(frame, roi)
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
	return res


def preproceso(image):
	"""
	Funcion de preprocesado de imagenes para usar con MobileNet3
	Modificado sin labels de format_example()
	https://colab.research.google.com/drive/1ZZXnCjFEOkp_KdNcNabd14yok0BAIuwS
	"""
	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, (Config.DNN.img_size, Config.DNN.img_size))
	# image = vgg16.preprocess_input(image)
	# label = tf.one_hot(tf.cast(label, tf.int32), 2)
	# label = tf.cast(label, tf.float32)
	return image

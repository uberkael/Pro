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
	"Devuelve una imagen que contiene solo la Region de Interes"
	x, y, w, h = ROI
	return frame[y:y+h, x:x+w]


def dibuja_predic(frame, ROIs, predicciones):
	"Escribe la prediccion mas popular encima  del objetivo"
	if len(ROIs) != len(predicciones):
		print(f"Error, no iguales: {len(ROIs)} {len(predicciones)}")
		return
	#Recorre cada region
	for r, p in zip(ROIs, predicciones):
		# Extrae el texto de la prediccion
		text = p[0][1]
		x, y, w, h = r
		cv.putText(frame, text,
			(x+5, y-5), 0, 1, Config.UI.morado, 1, 16)
		cv.rectangle(frame, (x, y), (x+w, y+h), Config.UI.rojo_oscuro, 1)

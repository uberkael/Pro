import numpy as np
import cv2 as cv
import Config


def busca_centro(dim):
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
		# Calculamos el rectangulo que contiene el elemento
		peri = cv.arcLength(c, True)
		# Dibuja un rectangulo con su centro
		approx = cv.approxPolyDP(c, 0.02*peri, True)
		x, y, w, h = cv.boundingRect(approx)
		cv.rectangle(frame, (x, y), (x+w, y+h), Config.UI.rojo_oscuro, 1)
		cv.circle(frame, (x+w//2, y+h//2), 2, Config.UI.cyan, 0)

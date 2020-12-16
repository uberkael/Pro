#######################################################################
# Selector de orden de objetivos empezando por el centro de la imagen #
#######################################################################

import cv2 as cv
import numpy as np
import Utiles
import Config

num_obj_max = 0

def organiza_objetivos(dims, lista_p):
	"""Devuelve una lista de objetivos en ordenados segun su distancia mutua
	comenzando por el centro"""
	# DEBUG
	global num_obj_max
	tam = len(lista_p)
	if tam > num_obj_max:
		num_obj_max = tam
		print(num_obj_max)
	# DEBUG
	ctr_x, ctr_y = Utiles.punto_centro(dims)
	centro = np.array((ctr_x, ctr_y))

	orden_objetivos = []
	objetivo_anterior = centro
	for i in range(len(lista_p)):
		distancias = [Utiles.distancia(x, objetivo_anterior) for x in lista_p]
		menor = np.argmin(distancias)
		objetivo_anterior = lista_p[menor]
		orden_objetivos.append(lista_p[menor])
		lista_p = np.delete(lista_p, menor, axis=0)
	return np.array(orden_objetivos)


def dibuja_puntos(img, lista_p):
	"Dibuja los puntos de los objetivos ordeandos y su orden numerico"
	for i, punto in enumerate(lista_p):
		# cv.circle(img, (punto[0], punto[1]), 5, Config.UI.cyan2, 3)
		# cv.putText(img, str(i), (punto[0]+5, punto[1]-5), 0, 1, Config.UI.lima, 2)
		cv.putText(img, str(i),
			(punto[0]+5, punto[1]-5), 0, 1, Config.UI.rojo_claro, 1, 16)


def dibuja_path(img, lista_p):
	"Dibuja el path a seguir por la torreta entre distintos objetivos"
	anterior = np.array([])
	for punto in lista_p:
		if anterior.any():
			cv.arrowedLine(img, (anterior[0], anterior[1]),
							(punto[0], punto[1]), Config.UI.rojo, 2,
							tipLength=0.05)
		anterior = punto


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Selector"
	num_puntos = 30
	dims_img = (500, 500, 3)
	lista_p = Utiles.gen_p_aleatorios(num_puntos, (dims_img[0], dims_img[1]))
	img = np.zeros(dims_img, np.uint8)
	dims = (img.shape[0], img.shape[1])
	orden_objetivos = organiza_objetivos(dims, lista_p)
	dibuja_path(img, orden_objetivos)
	dibuja_puntos(img, orden_objetivos)
	cv.imshow(titulo, img)
	cv.waitKey(0)


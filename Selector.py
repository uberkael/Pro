#######################################################################
# Selector de orden de objetivos empezando por el centro de la imagen #
#######################################################################

import cv2 as cv
import numpy as np
import Utiles
import Config

num_obj_max = 0


def organiza_objetivos(p_actual, lista_p):
	"""Devuelve una lista de objetivos en ordenados segun su distancia mutua
	desde un punto"""
	orden_objetivos = []
	objetivo_anterior = p_actual
	for i in range(len(lista_p)):
		distancias = [Utiles.distancia(x, objetivo_anterior) for x in lista_p]
		menor = np.argmin(distancias)
		objetivo_anterior = lista_p[menor]
		orden_objetivos.append(lista_p[menor])
		lista_p = np.delete(lista_p, menor, axis=0)
	return np.array(orden_objetivos)


def objetivo_prioritario(punto, objetivos, distancia_max=0):
	"""Devuelve el objetivo prioritario segun la distancia a la posicion actual
	de la torreta"""
	if(len(objetivos) < 1):
		return None
	puntos_centro = Utiles.centros_rectangulos(objetivos)
	objetivo_anterior = punto
	distancias = [Utiles.distancia(x, objetivo_anterior) for x in puntos_centro]
	menor = np.argmin(distancias)
	# Si el punto mas cercano esta lo suficientemente cerca no se devuelve nada
	if distancia_max and np.min(distancias) > distancia_max:
		return np.array([])
	return objetivos[menor]
	# return objetivos[menor], puntos_centro[menor]


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Selector"
	num_puntos = 20
	w, h = Config.VidProp.resolu
	# dims_img = (500, 500, 3)
	dims_img = [h, w, 3]
	lista_p = Utiles.gen_p_aleatorios(num_puntos, (dims_img[0], dims_img[1]))
	img = np.zeros(dims_img, np.uint8)
	dims = (img.shape[1], img.shape[0])
	p_actual = Utiles.punto_centro(dims)
	orden_objetivos = organiza_objetivos(p_actual, lista_p)
	Utiles.dibuja_path(img, orden_objetivos)
	img = Utiles.dibuja_puntos(img, orden_objetivos)
	cv.imshow(titulo, img)
	cv.waitKey(0)
